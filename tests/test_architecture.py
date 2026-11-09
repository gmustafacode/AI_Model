"""Architecture / gradient / cache tests for PhialModel. Run with pytest or `run_all()`.

Exactness tests use float64 so that "equal" really means equal, not "close in fp32 noise".
"""
from __future__ import annotations

import itertools
import math
import tempfile
import time
import traceback
import unittest

import torch
import torch.nn.functional as F

from model.phial_model import (Attention, Block, PhialConfig, PhialModel, RMSNorm, apply_rope,
                               count_parameters, rope_cos_sin, sinusoidal_table)

BASE = dict(vocab_size=97, block_size=16, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, dropout=0.0)


def make(double=False, **kw) -> PhialModel:
    torch.manual_seed(0)
    m = PhialModel(PhialConfig(**{**BASE, **kw}))
    if double:
        m = m.double()
    return m.eval()


def rand_ids(B, T, V=97, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, T), generator=g)


COMBOS = [dict(pos_type=p, norm_type=n, ffn_type=f)
          for p, n, f in itertools.product(["rope", "learned", "sinusoidal"], ["rmsnorm", "layernorm"], ["swiglu", "gelu"])]


# ---------------------------------------------------------------- shapes ------------------------------
def test_embedding_shape():
    m = make()
    ids = rand_ids(3, 7)
    assert m.tok_emb(ids).shape == (3, 7, 32)


def test_attention_shape_and_multihead_split():
    m = make()
    attn = m.blocks[0].attn
    assert attn.wq.out_features == 4 * 8 and attn.wk.out_features == 2 * 8 and attn.wv.out_features == 2 * 8
    x = torch.randn(2, 6, 32)
    pos = torch.arange(6)
    cos, sin = rope_cos_sin(pos, m.inv_freq)
    y, cache = attn(x, cos, sin, pos, None, True)
    assert y.shape == (2, 6, 32)
    assert cache.k.shape == (2, 2, 6, 8) and cache.v.shape == (2, 2, 6, 8) and cache.pos.shape == (6,)


def test_block_shape():
    m = make()
    x = torch.randn(2, 6, 32)
    pos = torch.arange(6)
    cos, sin = rope_cos_sin(pos, m.inv_freq)
    y, _ = m.blocks[0](x, cos, sin, pos)
    assert y.shape == x.shape


def test_logits_shape_all_variants():
    ids = rand_ids(2, 9)
    for kw in COMBOS:
        logits, loss, caches = make(**kw)(ids)
        assert logits.shape == (2, 9, 97), kw
        assert loss is None and caches is None


def test_param_count_matches_analytic():
    for kw in COMBOS:
        for tie in (True, False):
            m = make(tie_embeddings=tie, **kw)
            assert m.num_parameters() == count_parameters(m.cfg), (kw, tie, m.num_parameters(), count_parameters(m.cfg))


def test_tied_embeddings():
    t, u = make(tie_embeddings=True), make(tie_embeddings=False)
    assert t.lm_head.weight is t.tok_emb.weight
    assert u.lm_head.weight is not u.tok_emb.weight
    assert u.num_parameters() - t.num_parameters() == 97 * 32


def test_swiglu_is_roughly_parameter_matched_to_gelu():
    for d in (512, 768, 1024):
        sw = PhialConfig(n_embd=d, n_head=8, ffn_type="swiglu").resolved_ffn_hidden * 3 * d
        ge = PhialConfig(n_embd=d, n_head=8, ffn_type="gelu").resolved_ffn_hidden * 2 * d
        assert abs(sw - ge) / ge < 0.04, (d, sw, ge)


# ---------------------------------------------------------------- causality / masks -------------------
def test_causal_no_future_leak_all_positions():
    ids = rand_ids(2, 12)
    t = 7
    for kw in COMBOS[::2] + [dict(pos_type="rope", sliding_window=4, attention_sink=1)]:
        m = make(double=True, **kw)
        ids2 = ids.clone()
        ids2[:, t:] = (ids2[:, t:] + 5) % 97          # change every token at index >= t
        a, b = m(ids)[0], m(ids2)[0]
        assert (a[:, :t] - b[:, :t]).abs().max().item() < 1e-12, f"future leak: {kw}"
        assert (a[:, t:] - b[:, t:]).abs().max().item() > 1e-10, f"sanity: later positions must change: {kw}"


def test_sliding_window_blocks_far_past():
    m = make(double=True, n_layer=1, sliding_window=4, attention_sink=0)
    ids = rand_ids(1, 12)
    ids2 = ids.clone()
    ids2[:, 0] = (ids2[:, 0] + 3) % 97
    d = (m(ids)[0] - m(ids2)[0]).abs().amax(dim=(0, 2))
    assert d[4:].max().item() < 1e-12, "token 0 must not reach positions >= window"
    assert d[1:4].min().item() > 1e-10, "token 0 must reach positions inside the window"


def test_attention_sink_stays_visible():
    ids = rand_ids(1, 12)
    m = make(double=True, n_layer=1, sliding_window=4, attention_sink=1)
    i0 = ids.clone(); i0[:, 0] = (i0[:, 0] + 3) % 97
    i1 = ids.clone(); i1[:, 1] = (i1[:, 1] + 3) % 97
    d0 = (m(ids)[0] - m(i0)[0]).abs().amax(dim=(0, 2))
    d1 = (m(ids)[0] - m(i1)[0]).abs().amax(dim=(0, 2))
    assert d0[6:].min().item() > 1e-10, "sink token must remain visible far away"
    assert d1[5:].max().item() < 1e-12, "non-sink token must fall out of the window"


# ---------------------------------------------------------------- RoPE / RMSNorm ----------------------
def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    D = 16
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    q = torch.randn(1, 1, 1, D, dtype=torch.float64)
    k = torch.randn(1, 1, 1, D, dtype=torch.float64)

    def score(m, n):
        cq, sq = rope_cos_sin(torch.tensor([m]), inv)
        ck, sk = rope_cos_sin(torch.tensor([n]), inv)
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()

    base = score(5, 2)
    for shift in (1, 17, 300):
        assert abs(score(5 + shift, 2 + shift) - base) < 1e-9
    assert abs(score(5, 3) - base) > 1e-6, "score must change when the relative distance changes"


def test_rope_preserves_norm():
    D = 16
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    x = torch.randn(2, 3, 9, D, dtype=torch.float64)
    cos, sin = rope_cos_sin(torch.arange(9), inv)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-12)


def test_rmsnorm_matches_formula_and_is_stable():
    n = RMSNorm(8)
    with torch.no_grad():
        n.weight.copy_(torch.linspace(0.5, 1.5, 8))
    x = torch.randn(4, 8)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + n.eps) * n.weight
    assert torch.allclose(n(x), ref, atol=1e-6)
    big = (torch.randn(4, 8) * 1e3).to(torch.bfloat16)
    tiny = (torch.randn(4, 8) * 1e-4).to(torch.bfloat16)
    zero = torch.zeros(4, 8, dtype=torch.bfloat16)
    for v in (big, tiny, zero):
        assert torch.isfinite(n(v)).all()


def test_sinusoidal_table_properties():
    t = sinusoidal_table(10, 8)
    assert t.shape == (10, 8) and t.abs().max() <= 1.0
    assert torch.allclose(t[0, 0::2], torch.zeros(4)) and torch.allclose(t[0, 1::2], torch.ones(4))


# ---------------------------------------------------------------- KV cache ----------------------------
def _stepwise(m, ids, prefill):
    out, caches, pos = [], None, 0
    logits, _, caches = m(ids[:, :prefill], caches=None, start_pos=0, use_cache=True)
    out.append(logits)
    pos = prefill
    for t in range(prefill, ids.size(1)):
        logits, _, caches = m(ids[:, t:t + 1], caches=caches, start_pos=pos, use_cache=True)
        out.append(logits)
        pos += 1
    return torch.cat(out, dim=1), caches


def test_kv_cache_matches_uncached_and_cache_stays_bounded():
    cases = [("rope", None, 0, 14), ("rope", 4, 0, 24), ("rope", 4, 2, 24),
             ("learned", None, 0, 14), ("sinusoidal", None, 0, 14), ("learned", 5, 1, 14)]
    for pos_type, w, s, T in cases:
        m = make(double=True, pos_type=pos_type, sliding_window=w, attention_sink=s)
        ids = rand_ids(2, T)
        full = m(ids)[0]
        for prefill in (1, 5):
            step, caches = _stepwise(m, ids, prefill)
            err = (full - step).abs().max().item()
            assert err < 1e-9, f"cache mismatch {pos_type} w={w} s={s} prefill={prefill}: {err}"
            if w is not None:
                assert all(c.length <= s + w for c in caches), "cache must stay bounded by sink+window"


# ---------------------------------------------------------------- gradients ---------------------------
def test_gradient_flow_every_parameter():
    ids, tgt = rand_ids(3, 10), rand_ids(3, 10, seed=2)
    for kw in COMBOS + [dict(pos_type="rope", tie_embeddings=False)]:
        m = make(**kw)
        m.train()
        _, loss, _ = m(ids, tgt)
        assert torch.isfinite(loss)
        m.zero_grad()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"{name} disconnected ({kw})"
            assert torch.isfinite(p.grad).all(), f"{name} NaN/Inf grad ({kw})"
            assert p.grad.abs().sum() > 0, f"{name} zero grad ({kw})"


def test_loss_ignore_index_and_fully_masked():
    m = make()
    ids, tgt = rand_ids(2, 8), rand_ids(2, 8, seed=3)
    tgt[0, :3] = -100
    tgt[1, 5:] = -100
    logits, loss, _ = m(ids, tgt)
    valid = tgt != -100
    ref = F.cross_entropy(logits[valid], tgt[valid])
    assert abs(loss.item() - ref.item()) < 1e-6
    _, loss0, _ = m(ids, torch.full_like(ids, -100))
    assert loss0.item() == 0.0
    m.train(); m.zero_grad()
    m(ids, torch.full_like(ids, -100))[1].backward()


def test_initial_loss_is_near_log_vocab():
    m = make(vocab_size=512, n_embd=64, n_head=4, n_kv_head=4, block_size=32)
    ids, tgt = rand_ids(8, 32, V=512), rand_ids(8, 32, V=512, seed=5)
    loss = m(ids, tgt)[1].item()
    assert abs(loss - math.log(512)) < 0.2, (loss, math.log(512))


def test_grad_checkpoint_equivalence():
    ids, tgt = rand_ids(2, 10), rand_ids(2, 10, seed=2)
    res = []
    for ck in (False, True):
        m = make(double=True, grad_checkpoint=ck)
        m.train()
        loss = m(ids, tgt)[1]
        m.zero_grad(); loss.backward()
        res.append((loss.item(), [p.grad.clone() for p in m.parameters()]))
    assert abs(res[0][0] - res[1][0]) < 1e-10
    assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(res[0][1], res[1][1]))


def test_cuda_autocast_fp16_finite():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA")
    m = make(n_embd=64, n_head=4, n_kv_head=2).cuda().train()
    ids, tgt = rand_ids(4, 16).cuda(), rand_ids(4, 16, seed=2).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        logits, loss, _ = m(ids, tgt)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


# ---------------------------------------------------------------- misc --------------------------------
def test_save_and_load_roundtrip():
    m = make(pos_type="learned", norm_type="layernorm", ffn_type="gelu")
    ids = rand_ids(2, 9)
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = PhialModel.from_pretrained(d).eval()
    assert m2.cfg.to_dict() == m.cfg.to_dict()
    assert torch.equal(m(ids)[0], m2(ids)[0])


def test_input_validation_errors():
    m = make()
    bad = [(rand_ids(1, 4).float(), "float ids"), (rand_ids(1, 17), "too long"),
           (torch.full((1, 3), 97, dtype=torch.long), "id out of range"),
           (torch.empty(1, 0, dtype=torch.long), "empty")]
    for x, why in bad:
        try:
            m(x)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {why}")


# ---------------------------------------------------------------- runner ------------------------------
def run_all(verbose: bool = True):
    """Run every test_* function; return a list of dict rows. Real pass/fail, no swallowing."""
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
            status, msg = "FAIL", traceback.format_exc(limit=3)
        rows.append(dict(test=name, status=status, seconds=round(time.time() - t0, 2), message=msg))
        if verbose:
            print(f"[{status}] {name} ({rows[-1]['seconds']}s)" + (f"\n{msg}" if status == "FAIL" else ""))
    return rows
