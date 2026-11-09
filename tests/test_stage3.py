"""Stage 3 tests: schedule, optimizer groups, real-data batching, training steps, checkpoint/resume, evaluation, sampling.

Training/resume tests run on CPU in float32 with a tiny PhialModel on the REAL tokenised train/val splits, so they are
deterministic and fast. They test mechanics; they say nothing about how good a large pretrained model is.
Set CTX before running:  CTX.update(train_bin=..., val_bin=..., vocab=..., tokenizer_cfg=...)
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
import torch

from inference.generate import (apply_repetition_penalty, generate, make_generator, sample_next, stream_generate,
                                top_k_filter, top_p_filter)
from model.phial_model import PhialConfig, PhialModel
from training.pretraining import (TokenBin, TrainConfig, build_optimizer, evaluate, export_pretrained, load_checkpoint,
                                  lr_at, make_autocast, pretrain)

CTX = {}
CPU = torch.device("cpu")


def _cfg():
    return dict(vocab_size=CTX["vocab"], block_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)


def _tcfg(**kw):
    base = dict(max_steps=12, warmup_steps=2, lr=3e-3, min_lr=3e-4, micro_batch=2, grad_accum=2, eval_interval=6,
                eval_batches=2, ckpt_interval_steps=3, log_interval=2, amp_dtype="float32", seed=7)
    base.update(kw)
    return TrainConfig(**base)


def _run(d, tcfg=None, **kw):
    d = Path(d)
    return pretrain(d / "ckpt", d / "logs", _cfg(), tcfg or _tcfg(), CTX["train_bin"], CTX["val_bin"], CTX["tokenizer_cfg"],
                    CPU, tensorboard=False, log_fn=lambda *_: None, **kw)


def _tiny_model(double=False, **kw):
    torch.manual_seed(0)
    m = PhialModel(PhialConfig(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, **kw))
    return (m.double() if double else m).eval()       # float64 so argmax ties cannot flip on rounding noise


# ------------------------------------------------------------------ schedule / optimizer --------------
def test_lr_schedule_warmup_cosine_min():
    W, N, LR, MN = 10, 100, 1e-3, 1e-4
    lrs = [lr_at(s, W, N, LR, MN) for s in range(N)]
    assert all(a < b for a, b in zip(lrs[:W - 1], lrs[1:W])), "warmup must increase"
    assert abs(lrs[W - 1] - LR) < 1e-12 and abs(lrs[W] - LR) < 1e-12, "peak at end of warmup"
    assert all(a >= b - 1e-15 for a, b in zip(lrs[W:], lrs[W + 1:])), "cosine must not increase"
    assert min(lrs) >= MN - 1e-15 and abs(lr_at(N, W, N, LR, MN) - MN) < 1e-12
    assert lr_at(0, 0, N, LR, MN) == LR, "no warmup -> start at peak"


def test_optimizer_groups_cover_every_param_once_and_decay_rules():
    m = PhialModel(PhialConfig(**_cfg()))
    opt = build_optimizer(m, _tcfg(), "cpu")
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters())), "each parameter exactly once (tied weight not duplicated)"
    dec, nodec = opt.param_groups
    assert dec["weight_decay"] > 0 and nodec["weight_decay"] == 0
    assert all(p.dim() >= 2 for p in dec["params"]) and all(p.dim() < 2 for p in nodec["params"])
    assert len(nodec["params"]) > 0, "norm gains must be in the no-decay group"


def test_trainconfig_validation():
    for bad in (dict(max_steps=0), dict(warmup_steps=99999), dict(micro_batch=0), dict(min_lr=1.0, lr=0.1), dict(amp_dtype="int8")):
        try:
            TrainConfig(**{**dict(max_steps=10), **bad})
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {bad}")


# ------------------------------------------------------------------ data ------------------------------
def test_tokenbin_batch_alignment_range_and_shape():
    tb = TokenBin(CTX["train_bin"], CTX["vocab"])
    rng = np.random.default_rng(0)
    x, y = tb.sample(4, 64, rng, CPU)
    assert x.shape == y.shape == (4, 64) and x.dtype == y.dtype == torch.long
    assert (y[:, :-1] == x[:, 1:]).all(), "targets must be inputs shifted left by one"
    assert int(x.max()) < CTX["vocab"] and int(x.min()) >= 0
    bad = TokenBin(CTX["train_bin"], 5)           # real token ids exceed 5 -> must refuse, not silently clip
    try:
        bad.sample(8, 64, np.random.default_rng(0), CPU)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range ids not detected")


def test_eval_windows_are_deterministic_nonoverlapping_in_range():
    tb = TokenBin(CTX["val_bin"], CTX["vocab"])
    s1, s2 = tb.eval_starts(64, 10), tb.eval_starts(64, 10)
    assert (s1 == s2).all() and len(s1) == len(set(s1.tolist()))
    assert s1.min() >= 0 and s1.max() + 65 <= tb.n and (np.diff(s1) >= 64).all()


# ------------------------------------------------------------------ training --------------------------
def test_one_step_updates_every_param_and_resets_grads():
    torch.manual_seed(7)
    init = PhialModel(PhialConfig(**_cfg())).state_dict()
    init = {k: v.clone() for k, v in init.items()}
    with tempfile.TemporaryDirectory() as d:
        out = _run(d, _tcfg(max_steps=1, warmup_steps=1, eval_interval=1), return_objects=True)
        assert out["status"] == "finished" and out["step"] == 1
        m, opt = out["_model"], out["_opt"]
        assert all(p.grad is None for p in m.parameters()), "grads must be reset after the step"
        changed = {k: not torch.equal(v, init[k]) for k, v in m.state_dict().items()}
        assert all(changed.values()), f"params that did not change: {[k for k, c in changed.items() if not c]}"
        assert all("exp_avg" in s and s["step"] >= 1 for s in opt.state.values()), "Adam state missing"
        assert all(torch.isfinite(p).all() for p in m.parameters())


def test_training_reduces_loss_on_real_data():
    with tempfile.TemporaryDirectory() as d:
        _run(d, _tcfg(max_steps=40, warmup_steps=5, eval_interval=40, ckpt_interval_steps=40, log_interval=1))
        rows = [json.loads(l) for l in open(Path(d) / "logs" / "log.jsonl")]
        tr = [r["train_loss"] for r in rows if r["kind"] == "train"]
        assert len(tr) >= 30, len(tr)
        first, last = float(np.mean(tr[:5])), float(np.mean(tr[-5:]))
        assert last < first - 0.05, f"loss did not decrease: {first:.3f} -> {last:.3f}"
        assert abs(tr[0] - math.log(CTX["vocab"])) < 1.0, "initial loss should be near ln(V)"
        vals = [r for r in rows if r["kind"] == "val"]
        assert vals and all(math.isfinite(v["val_loss"]) for v in vals)


def test_checkpoint_contents_and_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        _run(d, _tcfg(max_steps=4, warmup_steps=1, eval_interval=2, ckpt_interval_steps=2))
        ck = load_checkpoint(Path(d) / "ckpt" / "ckpt_last.pt")
        for key in ("model", "optimizer", "scaler", "step", "tokens_seen", "best_val", "rng", "model_cfg", "train_cfg", "tokenizer_cfg"):
            assert key in ck, f"checkpoint lacks {key}"
        assert ck["step"] == 4 and ck["tokens_seen"] == 4 * 2 * 2 * 64
        assert set(ck["rng"]) >= {"python", "numpy_batcher", "torch_cpu"}
        assert (Path(d) / "ckpt" / "ckpt_best.pt").exists()
        m = PhialModel(PhialConfig(**ck["model_cfg"]))
        m.load_state_dict(ck["model"], strict=True)           # restoration into a fresh model
        for k, v in m.state_dict().items():
            assert torch.equal(v, ck["model"][k])
        o = build_optimizer(m, TrainConfig.from_dict(ck["train_cfg"]), "cpu")
        o.load_state_dict(ck["optimizer"])
        got, ref = o.state_dict()["state"], ck["optimizer"]["state"]
        assert got.keys() == ref.keys() and all(torch.equal(got[i]["exp_avg"], ref[i]["exp_avg"]) for i in ref)
        info = export_pretrained(Path(d) / "ckpt" / "ckpt_best.pt", Path(d) / "export")
        m2 = PhialModel.from_pretrained(Path(d) / "export")
        assert m2.num_parameters() == info["params"]


def test_exact_resume_matches_uninterrupted_run():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _run(a)                                                   # uninterrupted: 12 steps
        r1 = _run(b, pause_at_step=5)                             # stop mid-run ...
        assert r1["status"] == "paused_at_step" and r1["step"] == 5
        r2 = _run(b)                                              # ... and resume from the checkpoint
        assert r2["status"] == "finished" and r2["step"] == 12
        A = load_checkpoint(Path(a) / "ckpt" / "ckpt_last.pt")
        B = load_checkpoint(Path(b) / "ckpt" / "ckpt_last.pt")
        assert A["tokens_seen"] == B["tokens_seen"]
        worst = max((A["model"][k] - B["model"][k]).abs().max().item() for k in A["model"])
        assert worst < 1e-6, f"resumed run diverged from uninterrupted run (max |dw| = {worst:.3e})"
        assert abs(A["best_val"] - B["best_val"]) < 1e-6


def test_never_silently_restarts_and_rejects_config_drift():
    with tempfile.TemporaryDirectory() as d:
        _run(d, _tcfg(max_steps=2, warmup_steps=1, eval_interval=2, ckpt_interval_steps=1))
        try:
            _run(d, _tcfg(max_steps=2, warmup_steps=1), resume=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing checkpoint was silently overwritten")
        try:
            _run(d, _tcfg(max_steps=2, warmup_steps=1, lr=1e-2))
        except ValueError:
            pass
        else:
            raise AssertionError("changed lr accepted on resume")
        again = _run(d, _tcfg(max_steps=2, warmup_steps=1))
        assert again["status"] == "already_finished"
        ow = _run(d, _tcfg(max_steps=2, warmup_steps=1), resume=False, overwrite=True)
        assert ow["status"] == "finished"                         # explicit overwrite is allowed


def test_evaluate_is_deterministic_eval_mode_no_grad_and_independent():
    m = PhialModel(PhialConfig(**_cfg())).train()
    before = [p.clone() for p in m.parameters()]
    val = TokenBin(CTX["val_bin"], CTX["vocab"])
    e1 = evaluate(m, val, 64, 2, 6, CPU, make_autocast(CPU, "float32"))
    e2 = evaluate(m, val, 64, 2, 6, CPU, make_autocast(CPU, "float32"))
    assert e1 == e2 and m.training, "deterministic; train mode restored"
    assert all(p.grad is None for p in m.parameters()), "evaluation must not create gradients"
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert abs(e1["loss"] - math.log(CTX["vocab"])) < 1.0 and abs(e1["ppl"] - math.exp(e1["loss"])) < 1e-6 * e1["ppl"]


# ------------------------------------------------------------------ sampling --------------------------
def test_top_k_and_top_p_filters():
    lg = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    assert torch.isfinite(top_k_filter(lg, 2)).sum() == 2 and torch.isfinite(top_k_filter(lg, 99)).sum() == 4
    assert torch.isfinite(top_p_filter(lg, 0.7)).sum() == 2
    assert torch.isfinite(top_p_filter(lg, 0.4)).sum() == 1 and torch.isfinite(top_p_filter(lg, 1.0)).sum() == 4
    assert torch.isfinite(top_p_filter(lg, 0.4))[0], "the most likely token must survive"
    sh = torch.tensor([0.1, 3.0, 0.2, 2.0])                      # unsorted input: positions must be preserved
    assert torch.isfinite(top_k_filter(sh, 2)).tolist() == [False, True, False, True]
    assert torch.isfinite(top_p_filter(sh, 0.5)).tolist() == [False, True, False, False]


def test_repetition_penalty():
    lg = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_repetition_penalty(lg, [0, 1, 1], 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0, 0.5] and lg.tolist() == [2.0, -2.0, 1.0, 0.5], "input not mutated"
    assert apply_repetition_penalty(lg, [0], 1.0) is lg


def test_greedy_matches_manual_argmax_and_cache_matches_uncached():
    m = _tiny_model(double=True)
    prompt = [5, 6, 7, 8]
    cached = generate(m, prompt, max_new_tokens=12, greedy=True, use_cache=True)["tokens"]
    plain = generate(m, prompt, max_new_tokens=12, greedy=True, use_cache=False)["tokens"]
    seq = list(prompt)
    for _ in range(12):
        with torch.no_grad():
            seq.append(int(m(torch.tensor([seq]))[0][0, -1].argmax()))
    assert cached == plain == seq[len(prompt):], (cached, plain, seq[len(prompt):])
    assert generate(m, prompt, max_new_tokens=12, greedy=True)["tokens"] == cached, "greedy must be deterministic"


def test_sampling_reproducible_and_topk1_is_greedy():
    m = _tiny_model()
    prompt = [3, 4, 5]
    a = generate(m, prompt, max_new_tokens=15, temperature=1.2, top_p=0.95, generator=make_generator(1))["tokens"]
    b = generate(m, prompt, max_new_tokens=15, temperature=1.2, top_p=0.95, generator=make_generator(1))["tokens"]
    c = generate(m, prompt, max_new_tokens=15, temperature=1.2, top_p=0.95, generator=make_generator(2))["tokens"]
    assert a == b and a != c and all(0 <= t < 97 for t in a + c)
    g = generate(m, prompt, max_new_tokens=10, greedy=True)["tokens"]
    k1 = generate(m, prompt, max_new_tokens=10, temperature=0.8, top_k=1, generator=make_generator(0))["tokens"]
    assert g == k1


def test_eos_max_length_context_limit_and_streaming():
    m = _tiny_model()
    prompt = [1, 2, 3]
    first = generate(m, prompt, max_new_tokens=1, greedy=True)["tokens"][0]
    r = generate(m, prompt, max_new_tokens=20, greedy=True, eos_id=first)
    assert r["tokens"] == [first] and r["finish_reason"] == "eos"
    r = generate(m, prompt, max_new_tokens=7, greedy=True)
    assert len(r["tokens"]) == 7 and r["finish_reason"] == "max_new_tokens"
    r = generate(m, list(range(1, 30)), max_new_tokens=20, greedy=True)       # block_size = 32, prompt = 29
    assert len(r["tokens"]) == 3 and r["finish_reason"] == "context_limit"
    info = {}
    streamed = list(stream_generate(m, prompt, max_new_tokens=9, greedy=True, info=info))
    assert streamed == generate(m, prompt, max_new_tokens=9, greedy=True)["tokens"] and info["finish_reason"] == "max_new_tokens"
    for bad in ([], list(range(40))):
        try:
            generate(m, bad, max_new_tokens=2)
        except ValueError:
            continue
        raise AssertionError("bad prompt accepted")


def test_generation_past_block_size_with_rolling_cache_and_mode_restored():
    m = _tiny_model(double=True, sliding_window=8, attention_sink=1)
    m.train()
    r = generate(m, [1, 2, 3], max_new_tokens=60, greedy=True)               # 63 positions > block_size 32
    assert len(r["tokens"]) == 60 and r["finish_reason"] == "max_new_tokens" and m.training
    assert r["tokens"] == generate(m, [1, 2, 3], max_new_tokens=60, greedy=True, use_cache=False)["tokens"]


# ------------------------------------------------------------------ runner ----------------------------
def run_all(verbose: bool = True):
    rows = []
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        t0 = time.time()
        try:
            fn()
            status, msg = "PASS", ""
        except unittest.SkipTest as e:
            status, msg = "SKIP", str(e)
        except Exception:
            status, msg = "FAIL", traceback.format_exc(limit=4)
        rows.append(dict(test=name, status=status, seconds=round(time.time() - t0, 2), message=msg))
        if verbose:
            print(f"[{status}] {name} ({rows[-1]['seconds']}s)" + (f"\n{msg}" if status == "FAIL" else ""))
    return rows
