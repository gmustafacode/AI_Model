"""PHIAL text generation: greedy, temperature, top-k, top-p, repetition penalty, EOS/stop ids,
KV-cached decoding and token streaming. Works on any PhialModel; independent of the training notebook.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Union

import torch


def make_generator(seed: int, device="cpu") -> torch.Generator:
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(seed)
    return g


def apply_repetition_penalty(logits: torch.Tensor, seen_ids: Sequence[int], penalty: float) -> torch.Tensor:
    """CTRL-style: positive logits of already-seen tokens are divided by `penalty`, negative ones multiplied."""
    if penalty == 1.0 or len(seen_ids) == 0:
        return logits
    if penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    idx = torch.tensor(sorted(set(int(i) for i in seen_ids)), device=logits.device, dtype=torch.long)
    out = logits.clone()
    v = out[idx]
    out[idx] = torch.where(v > 0, v / penalty, v * penalty)
    return out


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k < 1:
        raise ValueError("top_k must be >= 1")
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of top tokens whose cumulative probability reaches p (always at least one)."""
    if not (0.0 < p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    sl, si = torch.sort(logits, descending=True)
    probs = torch.softmax(sl, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = (cum - probs) >= p          # drop a token if the mass BEFORE it already reaches p
    remove[0] = False
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, si, sl.masked_fill(remove, float("-inf")))
    return out


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
                repetition_penalty: float = 1.0, seen_ids: Sequence[int] = (), greedy: bool = False,
                generator: Optional[torch.Generator] = None) -> int:
    """logits: (V,) -> next token id. Order: repetition penalty -> (greedy | temperature -> top-k -> top-p -> sample)."""
    if logits.dim() != 1:
        raise ValueError(f"logits must be (V,), got {tuple(logits.shape)}")
    logits = logits if logits.dtype == torch.float64 else logits.float()
    logits = apply_repetition_penalty(logits, seen_ids, repetition_penalty)
    if greedy or temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None and top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all():
        raise FloatingPointError("non-finite sampling distribution")
    if generator is not None and generator.device != probs.device:
        return int(torch.multinomial(probs.to(generator.device), 1, generator=generator))
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def stream_generate(model, prompt_ids: List[int], max_new_tokens: int = 128, temperature: float = 1.0,
                    top_k: Optional[int] = None, top_p: Optional[float] = None, repetition_penalty: float = 1.0,
                    eos_id: Union[None, int, Sequence[int]] = None, greedy: bool = False, use_cache: bool = True,
                    generator: Optional[torch.Generator] = None, info: Optional[dict] = None) -> Iterator[int]:
    """Yield new token ids one by one. `info['finish_reason']` is set when the generator ends:
    eos | max_new_tokens | context_limit. A stop token IS yielded (callers usually strip it when decoding)."""
    info = info if info is not None else {}
    cfg = model.cfg
    device = next(model.parameters()).device
    if len(prompt_ids) == 0:
        raise ValueError("empty prompt")
    unbounded = cfg.pos_type == "rope" and cfg.sliding_window is not None      # rolling cache: no hard length limit
    if not unbounded and len(prompt_ids) > cfg.block_size:
        raise ValueError(f"prompt has {len(prompt_ids)} tokens > block_size {cfg.block_size}; truncate it first")
    cap = None if unbounded else cfg.block_size - len(prompt_ids)
    stops = set() if eos_id is None else ({int(eos_id)} if isinstance(eos_id, int) else {int(i) for i in eos_id})
    was_training = model.training
    model.eval()
    try:
        gen: List[int] = []
        seq = list(prompt_ids)
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        logits, _, caches = model(ids, use_cache=use_cache)
        pos = len(seq)
        info["finish_reason"] = "max_new_tokens"
        for i in range(max_new_tokens):
            if cap is not None and i >= cap:
                info["finish_reason"] = "context_limit"
                break
            nxt = sample_next(logits[0, -1], temperature, top_k, top_p, repetition_penalty, seq, greedy, generator)
            gen.append(nxt)
            seq.append(nxt)
            yield nxt
            if nxt in stops:
                info["finish_reason"] = "eos"
                break
            if i == max_new_tokens - 1:
                break
            if use_cache:
                step_in = torch.tensor([[nxt]], dtype=torch.long, device=device)
                logits, _, caches = model(step_in, caches=caches, start_pos=pos, use_cache=True)
            else:
                logits = model(torch.tensor([seq], dtype=torch.long, device=device))[0]
            pos += 1
    finally:
        model.train(was_training)


def generate(model, prompt_ids: List[int], **kw) -> dict:
    info: dict = {}
    toks = list(stream_generate(model, prompt_ids, info=info, **kw))
    return dict(tokens=toks, finish_reason=info.get("finish_reason"), prompt_len=len(prompt_ids))


def generate_text(model, tok, prompt: str, add_bos: bool = False, **kw) -> dict:
    """Convenience wrapper for the BASE model (plain continuation, no chat template).
    The pretraining corpus is `doc tokens + <|eos|>` with no BOS, so BOS is off by default."""
    ids = tok.encode(prompt, add_bos=add_bos)
    kw.setdefault("eos_id", tok.eos_id)
    out = generate(model, ids, **kw)
    out["text"] = tok.decode(out["tokens"], skip_special=True)
    return out
