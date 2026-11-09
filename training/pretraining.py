"""PHIAL pretraining: AdamW, linear warmup + cosine decay, gradient accumulation, AMP,
exact-resume checkpoints, held-out evaluation and CSV/JSONL/TensorBoard logging.

Nothing here loads pretrained weights: the model is built from PhialConfig and randomly initialised.
Native PyTorch only (no Trainer abstraction).
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model.phial_model import PhialConfig, PhialModel


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    max_steps: int = 1000                 # optimizer steps for the WHOLE run (fixes the LR schedule)
    warmup_steps: int = 50
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    micro_batch: int = 8
    grad_accum: int = 4
    eval_interval: int = 100
    eval_batches: int = 20                # number of micro-batches of held-out windows per evaluation
    ckpt_interval_steps: int = 200
    ckpt_interval_s: float = 900.0
    log_interval: int = 10
    amp_dtype: str = "float32"            # bfloat16 | float16 | float32 (ignored on CPU)
    seed: int = 1337
    grad_checkpoint: bool = False

    def __post_init__(self):
        if self.max_steps < 1 or self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise ValueError("need 1 <= max_steps and 0 <= warmup_steps <= max_steps")
        if self.micro_batch < 1 or self.grad_accum < 1:
            raise ValueError("micro_batch and grad_accum must be >= 1")
        if not (0 <= self.min_lr <= self.lr):
            raise ValueError("need 0 <= min_lr <= lr")
        if self.amp_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"bad amp_dtype {self.amp_dtype}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Fields that must not change between a checkpoint and its resumption (they define the run).
# micro_batch / grad_accum may change individually (different GPU) as long as their PRODUCT is equal.
RUN_DEFINING = ("max_steps", "warmup_steps", "lr", "min_lr", "weight_decay", "beta1", "beta2", "eps", "grad_clip", "seed")


# --------------------------------------------------------------------------------------
# Schedule, optimizer, data
# --------------------------------------------------------------------------------------
def lr_at(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Learning rate for the 0-based optimizer step `step`: linear warmup, then cosine decay to min_lr."""
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    prog = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig, device_type: str):
    """AdamW. Matrices (ndim >= 2) are decayed; norm gains / biases (ndim < 2) are not. Tied weights appear once."""
    seen, decay, no_decay = set(), [], []
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [dict(params=decay, weight_decay=cfg.weight_decay), dict(params=no_decay, weight_decay=0.0)]
    kw = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    if device_type == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kw)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(groups, **kw)


class TokenBin:
    """uint16 token stream on disk (documents separated by EOS). Random windows for training,
    deterministic windows for evaluation. Targets are the inputs shifted left by one."""

    def __init__(self, path, vocab_size: int):
        self.path, self.vocab_size = Path(path), int(vocab_size)
        self.mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n = int(len(self.mm))

    def _to_device(self, x: np.ndarray, y: np.ndarray, device):
        if int(x.max()) >= self.vocab_size or int(y.max()) >= self.vocab_size:
            raise ValueError(f"{self.path.name}: token id >= vocab_size {self.vocab_size}")
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        if device.type == "cuda":
            return xt.pin_memory().to(device, non_blocking=True), yt.pin_memory().to(device, non_blocking=True)
        return xt.to(device), yt.to(device)

    def _window(self, starts, T):
        x = np.stack([self.mm[i:i + T] for i in starts]).astype(np.int64)
        y = np.stack([self.mm[i + 1:i + T + 1] for i in starts]).astype(np.int64)
        return x, y

    def sample(self, B: int, T: int, rng: np.random.Generator, device):
        if self.n < T + 2:
            raise ValueError(f"{self.path.name} has only {self.n} tokens; need > block_size+1 = {T + 1}")
        starts = rng.integers(0, self.n - T - 1, B)
        return self._to_device(*self._window(starts, T), device)

    def n_windows(self, T: int) -> int:
        return (self.n - 1) // T

    def eval_starts(self, T: int, n_windows: int):
        """Non-overlapping windows spread evenly over the split (deterministic)."""
        total = self.n_windows(T)
        if total < 1:
            raise ValueError(f"{self.path.name} too small for block_size {T}")
        k = min(n_windows, total)
        return (np.floor(np.arange(k) * total / k).astype(np.int64)) * T


def make_autocast(device, amp_dtype: str):
    if device.type == "cuda" and amp_dtype != "float32":
        dt = getattr(torch, amp_dtype)
        return lambda: torch.autocast("cuda", dtype=dt)
    return lambda: contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, data: TokenBin, T: int, batch: int, n_windows: int, device, autocast_fn):
    """Held-out loss on deterministic non-overlapping windows. Eval mode, no gradients; train mode restored."""
    was_training = model.training
    model.eval()
    starts = data.eval_starts(T, n_windows)
    tot, cnt = 0.0, 0
    for i in range(0, len(starts), batch):
        x, y = data._to_device(*data._window(starts[i:i + batch], T), device)
        with autocast_fn():
            loss = model(x, y)[1]
        v = float(loss)
        if not math.isfinite(v):
            raise FloatingPointError("non-finite validation loss")
        tot += v * x.size(0)
        cnt += x.size(0)
    model.train(was_training)
    mean = tot / cnt
    return dict(loss=mean, ppl=math.exp(min(mean, 30.0)), windows=cnt, tokens=cnt * T)


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def atomic_save(obj, path: Path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)            # a crash mid-write never corrupts the previous checkpoint


def rng_bundle(np_rng: np.random.Generator):
    return dict(python=random.getstate(), numpy_batcher=np_rng.bit_generator.state, torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(b, np_rng: np.random.Generator):
    random.setstate(b["python"])
    np_rng.bit_generator.state = b["numpy_batcher"]
    torch.set_rng_state(b["torch_cpu"])
    if b.get("torch_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(b["torch_cuda"])
        except Exception:         # different GPU count on resume: harmless (dropout is 0 for pretraining)
            pass


def load_checkpoint(path):
    # weights_only=False: this is OUR checkpoint (contains RNG states / python objects); never load untrusted files this way.
    return torch.load(path, map_location="cpu", weights_only=False)


class RunLogger:
    def __init__(self, log_dir, tensorboard=True, wandb_project=None, run_name=None, config=None):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tb = self.wb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.dir / "tb"))
            except Exception as e:
                print(f"[logger] TensorBoard unavailable ({type(e).__name__}); continuing with CSV/JSONL only")
        if wandb_project:
            try:
                import wandb
                wandb.init(project=wandb_project, name=run_name, config=config, resume="allow")
                self.wb = wandb
            except Exception as e:
                print(f"[logger] W&B unavailable ({type(e).__name__}); continuing without it")

    def _csv(self, name, row):
        p = self.dir / name
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def log(self, kind: str, step: int, row: dict):
        row = {"step": step, **row}
        self._csv(f"{kind}_log.csv", row)
        with open(self.dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"kind": kind, **row}) + "\n")
        for k, v in row.items():
            if k != "step" and isinstance(v, (int, float)):
                if self.tb is not None:
                    self.tb.add_scalar(f"{kind}/{k}", v, step)
        if self.wb is not None:
            self.wb.log({f"{kind}/{k}": v for k, v in row.items() if k != "step" and isinstance(v, (int, float))}, step=step)

    def close(self):
        if self.tb is not None:
            self.tb.close()
        if self.wb is not None:
            self.wb.finish()


# --------------------------------------------------------------------------------------
# The training run
# --------------------------------------------------------------------------------------
def pretrain(ckpt_dir, log_dir, model_cfg: dict, tcfg: TrainConfig, train_bin, val_bin, tokenizer_cfg: dict, device,
             resume: bool = True, overwrite: bool = False, time_budget_s: Optional[float] = None,
             pause_at_step: Optional[int] = None, tensorboard: bool = True, wandb_project: Optional[str] = None,
             return_objects: bool = False, log_fn=print) -> dict:
    """Run (or resume) pretraining. Returns a summary dict whose `status` is one of:
    finished | paused_time_budget | paused_at_step | already_finished | interrupted."""
    ckpt_dir, log_dir = Path(ckpt_dir), Path(log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "ckpt_last.pt", ckpt_dir / "ckpt_best.pt"
    device = torch.device(device)
    model_cfg = {k: v for k, v in model_cfg.items() if k != "grad_checkpoint"}
    cfg_full = PhialConfig(**{**model_cfg, "grad_checkpoint": tcfg.grad_checkpoint})
    T = cfg_full.block_size
    tokens_per_step = tcfg.micro_batch * tcfg.grad_accum * T

    # ---- never silently restart from random init when a checkpoint exists --------------------------------
    ckpt = None
    if last_path.exists():
        if overwrite:
            log_fn(f"[pretrain] overwrite=True: discarding existing checkpoint(s) in {ckpt_dir}")
            for p in ckpt_dir.glob("ckpt_*.pt"):
                p.unlink()
            if log_dir.exists():
                shutil.rmtree(log_dir)
        elif not resume:
            raise FileExistsError(f"{last_path} exists. Use resume=True to continue it, or overwrite=True to discard it.")
        else:
            ckpt = load_checkpoint(last_path)
            if ckpt["model_cfg"] != model_cfg:
                raise ValueError(f"model config differs from checkpoint:\n  ckpt: {ckpt['model_cfg']}\n  now:  {model_cfg}")
            old = TrainConfig.from_dict(ckpt["train_cfg"])
            diffs = {k: (getattr(old, k), getattr(tcfg, k)) for k in RUN_DEFINING if getattr(old, k) != getattr(tcfg, k)}
            if old.micro_batch * old.grad_accum != tcfg.micro_batch * tcfg.grad_accum:
                diffs["micro_batch*grad_accum"] = (old.micro_batch * old.grad_accum, tcfg.micro_batch * tcfg.grad_accum)
            if diffs:
                raise ValueError(f"training config differs from checkpoint (would change the run): {diffs}")

    # ---- build everything ------------------------------------------------------------------------------
    torch.manual_seed(tcfg.seed)
    random.seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = PhialModel(cfg_full).to(device)          # RANDOM initialisation
    model.validate_inputs = False                    # ids are range-checked on the CPU side in TokenBin
    opt = build_optimizer(model, tcfg, device.type)
    use_scaler = device.type == "cuda" and tcfg.amp_dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast_fn = make_autocast(device, tcfg.amp_dtype)
    vocab = cfg_full.vocab_size
    train, val = TokenBin(train_bin, vocab), TokenBin(val_bin, vocab)
    np_rng = np.random.default_rng(tcfg.seed)
    step, tokens_seen, best_val, skipped = 0, 0, float("inf"), 0

    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["optimizer"])
        if use_scaler and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        step, tokens_seen = int(ckpt["step"]), int(ckpt["tokens_seen"])
        best_val, skipped = float(ckpt["best_val"]), int(ckpt.get("skipped_steps", 0))
        restore_rng(ckpt["rng"], np_rng)
        log_fn(f"[pretrain] RESUMED from {last_path.name}: step {step}/{tcfg.max_steps}, {tokens_seen:,} tokens, best_val {best_val:.4f}")
    else:
        log_fn(f"[pretrain] fresh run: random init, {model.num_parameters() / 1e6:.1f}M params, "
               f"{tokens_per_step:,} tokens/step, {tcfg.max_steps} steps = {tcfg.max_steps * tokens_per_step / 1e6:,.1f}M tokens")
    if step >= tcfg.max_steps:
        log_fn("[pretrain] run already complete")
        out = dict(status="already_finished", step=step, tokens_seen=tokens_seen, best_val=best_val)
        if return_objects:
            out.update(_model=model, _opt=opt)
        return out

    logger = RunLogger(log_dir, tensorboard, wandb_project, run_name=ckpt_dir.name, config=dict(model=model_cfg, train=tcfg.to_dict()))

    def save_last():
        atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                         scaler=scaler.state_dict() if use_scaler else None,
                         step=step, tokens_seen=tokens_seen, best_val=best_val, skipped_steps=skipped,
                         rng=rng_bundle(np_rng), model_cfg=model_cfg, train_cfg=tcfg.to_dict(),
                         tokenizer_cfg=tokenizer_cfg, saved_at=time.time()), last_path)

    def save_best(vloss):
        atomic_save(dict(model=model.state_dict(), model_cfg=model_cfg, tokenizer_cfg=tokenizer_cfg,
                         step=step, val_loss=vloss), best_path)

    params = [p for g in opt.param_groups for p in g["params"]]
    t_session, t_log, t_ckpt = time.time(), time.time(), time.time()
    steps_this_session, status = 0, "finished"
    loss_window, last_gn = [], float("nan")
    model.train()
    try:
        while step < tcfg.max_steps:
            lr = lr_at(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr, tcfg.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_acc = torch.zeros((), device=device)
            for _ in range(tcfg.grad_accum):
                x, y = train.sample(tcfg.micro_batch, T, np_rng, device)
                with autocast_fn():
                    loss = model(x, y)[1]
                scaler.scale(loss / tcfg.grad_accum).backward()
                loss_acc += loss.detach().float() / tcfg.grad_accum
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip))
            loss_v = float(loss_acc)
            if not math.isfinite(loss_v):
                raise FloatingPointError(f"non-finite training loss at step {step}: {loss_v}")
            if not math.isfinite(gnorm):
                if use_scaler:
                    skipped += 1             # GradScaler skips this update and lowers the scale (expected with fp16)
                else:
                    raise FloatingPointError(f"non-finite gradient norm at step {step}")
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            steps_this_session += 1
            tokens_seen += tokens_per_step
            loss_window.append(loss_v)
            last_gn = gnorm

            if step % tcfg.log_interval == 0 or step == tcfg.max_steps or step == 1:
                now = time.time()
                dt = max(now - t_log, 1e-9)
                n_in_window = len(loss_window)
                el = now - t_session
                sec_per_step = el / steps_this_session
                row = dict(train_loss=sum(loss_window) / n_in_window, lr=lr, grad_norm=last_gn, tokens_seen=tokens_seen,
                           tokens_per_s=n_in_window * tokens_per_step / dt,
                           gpu_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0,
                           elapsed_s=el, eta_min=(tcfg.max_steps - step) * sec_per_step / 60.0, skipped_steps=skipped)
                logger.log("train", step, row)
                log_fn(f"step {step:>6}/{tcfg.max_steps} | loss {row['train_loss']:.4f} | lr {lr:.2e} | gnorm {last_gn:.2f} | "
                       f"{row['tokens_per_s']:,.0f} tok/s | {row['gpu_mem_gb']:.1f} GB | ETA {row['eta_min']:.0f} min")
                loss_window, t_log = [], now

            if step % tcfg.eval_interval == 0 or step == tcfg.max_steps:
                ev = evaluate(model, val, T, tcfg.micro_batch, tcfg.eval_batches * tcfg.micro_batch, device, autocast_fn)
                logger.log("val", step, dict(val_loss=ev["loss"], val_ppl=ev["ppl"], val_tokens=ev["tokens"], tokens_seen=tokens_seen))
                log_fn(f"  [val] step {step} | loss {ev['loss']:.4f} | ppl {ev['ppl']:.2f} | {ev['tokens']:,} held-out tokens")
                if ev["loss"] < best_val:
                    best_val = ev["loss"]
                    save_best(best_val)

            done = step >= tcfg.max_steps
            stop = None
            if pause_at_step is not None and step >= pause_at_step and not done:
                stop = "paused_at_step"
            elif time_budget_s is not None and (time.time() - t_session) > time_budget_s and not done:
                stop = "paused_time_budget"
            if done or stop or step % tcfg.ckpt_interval_steps == 0 or (time.time() - t_ckpt) > tcfg.ckpt_interval_s:
                save_last()
                t_ckpt = time.time()
            if stop:
                status = stop
                break
    except KeyboardInterrupt:
        status = "interrupted"
        log_fn(f"[pretrain] interrupted. The last saved checkpoint (see {last_path}) is intact; re-run with resume=True.")
    finally:
        logger.close()

    summary = dict(status=status, step=step, max_steps=tcfg.max_steps, tokens_seen=tokens_seen, best_val=best_val,
                   skipped_steps=skipped, session_seconds=round(time.time() - t_session, 1), params=model.num_parameters(),
                   tokens_per_step=tokens_per_step)
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log_fn(f"[pretrain] {status} at step {step}/{tcfg.max_steps}")
    if return_objects:
        summary.update(_model=model, _opt=opt)
    return summary


def export_pretrained(ckpt_path, out_dir, tokenizer_dir=None):
    """Write a standalone PHIAL model folder (model_config.json + model_weights.pt [+ tokenizer]) from a checkpoint."""
    ck = load_checkpoint(ckpt_path)
    cfg = PhialConfig.from_dict({**ck["model_cfg"], "grad_checkpoint": False})
    m = PhialModel(cfg)
    m.load_state_dict(ck["model"], strict=True)
    m.save_pretrained(out_dir)
    if tokenizer_dir is not None:
        for f in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy(Path(tokenizer_dir) / f, Path(out_dir) / f)
    return dict(step=ck["step"], params=m.num_parameters(), val_loss=ck.get("val_loss", ck.get("best_val")))
