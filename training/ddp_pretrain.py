
import contextlib
import json
import math
import os
import random
import sys
import time

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP


sys.path.insert(0, os.environ["PHIAL_ROOT"])

import training.pretraining as PT

from model.phial_model import PhialConfig, PhialModel


# ------------------------------------------------------------
# SAFE CHECKPOINT SAVE
# ------------------------------------------------------------

def safe_atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    torch.save(obj, tmp)
    os.replace(tmp, path)


PT.atomic_save = safe_atomic_save


def main():

    a = json.load(open(sys.argv[1]))

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    lrank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(lrank)
    device = torch.device("cuda", lrank)

    dist.init_process_group("nccl")

    tc = PT.TrainConfig.from_dict(a["tcfg"])
    mc = a["model_cfg"]

    T = mc["block_size"]

    if tc.grad_accum % world != 0:
        raise ValueError(
            f"grad_accum {tc.grad_accum} not divisible "
            f"by world size {world}"
        )

    accum = tc.grad_accum // world

    # --------------------------------------------------------
    # CRITICAL FIX:
    # Create checkpoint and log directories before any save.
    # Rank 0 creates them; all ranks wait for completion.
    # --------------------------------------------------------

    ck_dir = Path(a["ckpt_dir"])
    log_dir = Path(a["log_dir"])

    if rank == 0:
        ck_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    last = ck_dir / "ckpt_last.pt"
    best = ck_dir / "ckpt_best.pt"

    tps = tc.micro_batch * tc.grad_accum * T

    ck = None

    # --------------------------------------------------------
    # CHECKPOINT RESUME / OVERWRITE
    # --------------------------------------------------------

    if last.exists():

        if a["overwrite"]:

            if rank == 0:
                for p in ck_dir.glob("ckpt_*.pt"):
                    p.unlink()

                import shutil
                shutil.rmtree(log_dir, ignore_errors=True)

                log_dir.mkdir(parents=True, exist_ok=True)

            dist.barrier()

        elif not a["resume"]:

            raise FileExistsError(
                f"{last} exists: resume=True to continue "
                "or overwrite=True to discard"
            )

        else:

            ck = PT.load_checkpoint(last)

            if ck["model_cfg"] != mc:
                raise ValueError(
                    "Model config differs from checkpoint"
                )

            old = PT.TrainConfig.from_dict(ck["train_cfg"])

            diff = {
                k: (getattr(old, k), getattr(tc, k))
                for k in PT.RUN_DEFINING
                if getattr(old, k) != getattr(tc, k)
            }

            if old.micro_batch * old.grad_accum != (
                tc.micro_batch * tc.grad_accum
            ):
                diff["tokens/step"] = (
                    old.micro_batch * old.grad_accum,
                    tc.micro_batch * tc.grad_accum
                )

            if diff:
                raise ValueError(
                    f"Training config differs from checkpoint: {diff}"
                )

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    torch.manual_seed(tc.seed)
    random.seed(tc.seed)
    np.random.seed(tc.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = PhialConfig(
        **{
            **mc,
            "grad_checkpoint": tc.grad_checkpoint
        }
    )

    model = PhialModel(cfg).to(device)
    model.validate_inputs = False

    opt = PT.build_optimizer(model, tc, "cuda")

    use_scaler = tc.amp_dtype == "float16"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler
    )

    actx = PT.make_autocast(device, tc.amp_dtype)

    train = PT.TokenBin(
        a["train_bin"],
        cfg.vocab_size
    )

    val = PT.TokenBin(
        a["val_bin"],
        cfg.vocab_size
    )

    # Different random batches for each rank.
    np_rng = np.random.default_rng([tc.seed, rank])

    step = 0
    seen = 0
    best_val = float("inf")
    skipped = 0

    # --------------------------------------------------------
    # RESTORE CHECKPOINT
    # --------------------------------------------------------

    if ck is not None:

        model.load_state_dict(
            ck["model"],
            strict=True
        )

        opt.load_state_dict(ck["optimizer"])

        if use_scaler and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])

        step = int(ck["step"])
        seen = int(ck["tokens_seen"])
        best_val = float(ck["best_val"])
        skipped = int(ck.get("skipped_steps", 0))

        rr = ck.get("rng_ranks")

        if rr is not None and len(rr) == world:
            np_rng.bit_generator.state = rr[rank]

        else:
            np_rng = np.random.default_rng(
                [tc.seed, rank, step]
            )

        if rank == 0:
            print(
                f"[ddp] RESUMED step {step}/{tc.max_steps}, "
                f"{seen:,} tokens, best_val {best_val:.4f}",
                flush=True
            )

    elif rank == 0:

        print(
            f"[ddp] fresh run on {world} GPUs: "
            f"{model.num_parameters()/1e6:.1f}M params, "
            f"{tps:,} tokens/step, {tc.max_steps} steps",
            flush=True
        )

    if step >= tc.max_steps:

        if rank == 0:
            print("[ddp] run already complete", flush=True)

        dist.destroy_process_group()
        return

    # --------------------------------------------------------
    # DDP WRAPPER
    # --------------------------------------------------------

    ddp = DDP(
        model,
        device_ids=[lrank]
    )

    logger = (
        PT.RunLogger(
            log_dir,
            True,
            a.get("wandb"),
            ck_dir.name,
            dict(
                model=mc,
                train=tc.to_dict()
            )
        )
        if rank == 0
        else None
    )

    params = [
        p
        for g in opt.param_groups
        for p in g["params"]
    ]

    # --------------------------------------------------------
    # CHECKPOINT FUNCTION
    # --------------------------------------------------------

    def save_last():

        states = [None] * world

        dist.all_gather_object(
            states,
            np_rng.bit_generator.state
        )

        if rank == 0:

            PT.atomic_save(
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    scaler=(
                        scaler.state_dict()
                        if use_scaler
                        else None
                    ),
                    step=step,
                    tokens_seen=seen,
                    best_val=best_val,
                    skipped_steps=skipped,
                    rng=PT.rng_bundle(np_rng),
                    rng_ranks=states,
                    model_cfg=mc,
                    train_cfg=tc.to_dict(),
                    tokenizer_cfg=a["tokenizer_cfg"],
                    saved_at=time.time()
                ),
                last
            )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    t_sess = time.time()
    t_log = time.time()
    t_ck = time.time()

    n_sess = 0
    win = []

    status = "finished"
    gn = float("nan")

    model.train()

    while step < tc.max_steps:

        lr = PT.lr_at(
            step,
            tc.warmup_steps,
            tc.max_steps,
            tc.lr,
            tc.min_lr
        )

        for g in opt.param_groups:
            g["lr"] = lr

        acc = torch.zeros(
            (),
            device=device
        )

        for m in range(accum):

            x, y = train.sample(
                tc.micro_batch,
                T,
                np_rng,
                device
            )

            sync_context = (
                ddp.no_sync()
                if m < accum - 1
                else contextlib.nullcontext()
            )

            with sync_context:

                with actx():
                    loss = ddp(x, y)[1]

                scaler.scale(
                    loss / accum
                ).backward()

            acc += loss.detach().float() / accum

        dist.all_reduce(
            acc,
            op=dist.ReduceOp.AVG
        )

        scaler.unscale_(opt)

        gn = float(
            torch.nn.utils.clip_grad_norm_(
                params,
                tc.grad_clip
            )
        )

        lv = float(acc)

        if not math.isfinite(lv):
            raise FloatingPointError(
                f"Non-finite loss at step {step}"
            )

        if not math.isfinite(gn):

            if use_scaler:
                skipped += 1

            else:
                raise FloatingPointError(
                    f"Non-finite grad norm at step {step}"
                )

        scaler.step(opt)
        scaler.update()

        opt.zero_grad(set_to_none=True)

        step += 1
        n_sess += 1
        seen += tps

        win.append(lv)

        # ----------------------------------------------------
        # TRAINING LOGS
        # ----------------------------------------------------

        if rank == 0 and (
            step % tc.log_interval == 0
            or step in (1, tc.max_steps)
        ):

            now = time.time()
            elapsed = now - t_sess

            row = dict(
                train_loss=sum(win) / len(win),
                lr=lr,
                grad_norm=gn,
                tokens_seen=seen,
                tokens_per_s=(
                    len(win) * tps
                    / max(now - t_log, 1e-9)
                ),
                gpu_mem_gb=(
                    torch.cuda.max_memory_allocated()
                    / 2**30
                ),
                elapsed_s=elapsed,
                eta_min=(
                    (tc.max_steps - step)
                    * elapsed
                    / n_sess
                    / 60
                ),
                skipped_steps=skipped
            )

            logger.log(
                "train",
                step,
                row
            )

            print(
                f"step {step:>6}/{tc.max_steps} | "
                f"loss {row['train_loss']:.4f} | "
                f"lr {lr:.2e} | "
                f"gnorm {gn:.2f} | "
                f"{row['tokens_per_s']:,.0f} tok/s "
                f"({world} GPUs) | "
                f"{row['gpu_mem_gb']:.1f} GB | "
                f"ETA {row['eta_min']:.0f} min",
                flush=True
            )

            win = []
            t_log = now

        elif rank != 0 and step % tc.log_interval == 0:

            win = []

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            step % tc.eval_interval == 0
            or step == tc.max_steps
        ):

            v = torch.zeros(
                1,
                device=device
            )

            if rank == 0:

                ev = PT.evaluate(
                    model,
                    val,
                    T,
                    tc.micro_batch,
                    tc.eval_batches * tc.micro_batch,
                    device,
                    actx
                )

                v[0] = ev["loss"]

                logger.log(
                    "val",
                    step,
                    dict(
                        val_loss=ev["loss"],
                        val_ppl=ev["ppl"],
                        val_tokens=ev["tokens"],
                        tokens_seen=seen
                    )
                )

                print(
                    f"  [val] step {step} | "
                    f"loss {ev['loss']:.4f} | "
                    f"ppl {ev['ppl']:.2f}",
                    flush=True
                )

            dist.broadcast(v, 0)

            if float(v) < best_val:

                best_val = float(v)

                if rank == 0:

                    PT.atomic_save(
                        dict(
                            model=model.state_dict(),
                            model_cfg=mc,
                            tokenizer_cfg=a["tokenizer_cfg"],
                            step=step,
                            val_loss=best_val
                        ),
                        best
                    )

        # ----------------------------------------------------
        # SAVE / TIME BUDGET
        # ----------------------------------------------------

        flag = torch.zeros(
            2,
            device=device
        )

        if rank == 0:

            done = step >= tc.max_steps

            stop = (
                not done
                and a["time_budget_s"] is not None
                and (time.time() - t_sess)
                > a["time_budget_s"]
            )

            flag[0] = float(
                done
                or stop
                or step % tc.ckpt_interval_steps == 0
                or (
                    time.time() - t_ck
                ) > tc.ckpt_interval_s
            )

            flag[1] = float(stop)

        dist.broadcast(flag, 0)

        if flag[0] > 0:

            save_last()
            t_ck = time.time()

        if flag[1] > 0:

            status = "paused_time_budget"
            break

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    if rank == 0:

        logger.close()

        summary = dict(
            status=status,
            step=step,
            max_steps=tc.max_steps,
            tokens_seen=seen,
            best_val=best_val,
            skipped_steps=skipped,
            session_seconds=round(
                time.time() - t_sess,
                1
            ),
            params=model.num_parameters(),
            tokens_per_step=tps,
            world_size=world
        )

        (log_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        print(
            f"[ddp] {status} at step "
            f"{step}/{tc.max_steps}",
            flush=True
        )

    dist.barrier()
    dist.destroy_process_group()


main()
