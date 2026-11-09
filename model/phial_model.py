"""PHIAL AI decoder-only Transformer (native PyTorch).

Configurable: positions (rope | learned | sinusoidal), norm (rmsnorm | layernorm),
FFN (swiglu | gelu), grouped-query attention, sliding window + attention sink,
KV cache with eviction, optional activation checkpointing, optional tied embeddings.

All randomly initialised. No pretrained weights are ever loaded here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_HALF = (torch.float16, torch.bfloat16)


def _up(x: torch.Tensor) -> torch.Tensor:
    """Upcast half precision to fp32 for numerically sensitive ops; keep fp32/fp64 as is."""
    return x.float() if x.dtype in _HALF else x


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class PhialConfig:
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None      # None -> n_head (plain multi-head attention)
    n_embd: int = 768
    dropout: float = 0.0
    pos_type: str = "rope"               # rope | learned | sinusoidal
    norm_type: str = "rmsnorm"           # rmsnorm | layernorm
    ffn_type: str = "swiglu"             # swiglu | gelu
    ffn_hidden: Optional[int] = None     # None -> parameter-matched default (see resolved_ffn_hidden)
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    sliding_window: Optional[int] = None  # None -> full causal attention
    attention_sink: int = 0               # number of always-visible leading tokens (needs sliding_window)
    grad_checkpoint: bool = False
    init_std: float = 0.02

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} not divisible by n_head {self.n_head}")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head {self.n_head} not divisible by n_kv_head {self.n_kv_head}")
        if self.pos_type not in ("rope", "learned", "sinusoidal"):
            raise ValueError(f"bad pos_type {self.pos_type}")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"bad norm_type {self.norm_type}")
        if self.ffn_type not in ("swiglu", "gelu"):
            raise ValueError(f"bad ffn_type {self.ffn_type}")
        if self.pos_type == "rope" and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.attention_sink < 0:
            raise ValueError("attention_sink must be >= 0")
        if self.attention_sink > 0 and self.sliding_window is None:
            raise ValueError("attention_sink requires sliding_window")
        if self.pos_type != "rope" and self.sliding_window is not None:
            # absolute positions cannot go past block_size, so a rolling window adds nothing there
            pass

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def resolved_ffn_hidden(self) -> int:
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        if self.ffn_type == "swiglu":
            # 3 matrices of width h  ~  2 matrices of width 4d  ->  h = 8d/3, rounded up to a multiple of 64
            return int(math.ceil(8 * self.n_embd / 3 / 64) * 64)
        return 4 * self.n_embd

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PhialConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def count_parameters(cfg: PhialConfig) -> int:
    """Analytic parameter count (all Linear layers are bias-free)."""
    d, V, hd = cfg.n_embd, cfg.vocab_size, cfg.head_dim
    h = cfg.resolved_ffn_hidden
    total = V * d
    if cfg.pos_type == "learned":
        total += cfg.block_size * d
    attn = d * (cfg.n_head * hd) + 2 * d * (cfg.n_kv_head * hd) + (cfg.n_head * hd) * d
    ffn = (3 if cfg.ffn_type == "swiglu" else 2) * d * h
    norm = d if cfg.norm_type == "rmsnorm" else 2 * d
    total += cfg.n_layer * (attn + ffn + 2 * norm) + norm
    if not cfg.tie_embeddings:
        total += V * d
    return total


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g, computed in fp32 for half-precision inputs."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = _up(x)
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (out * self.weight.to(out.dtype)).to(x.dtype)


def make_norm(cfg: PhialConfig) -> nn.Module:
    return RMSNorm(cfg.n_embd) if cfg.norm_type == "rmsnorm" else nn.LayerNorm(cfg.n_embd)


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """positions: (T,) long -> cos, sin of shape (T, head_dim) (half-split layout)."""
    freqs = positions.to(inv_freq.dtype)[:, None] * inv_freq[None, :]   # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)                            # (T, D)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). Rotates pairs (x_i, x_{i+D/2}) by position-dependent angles."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


def sinusoidal_table(max_len: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(max_len, dim)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class KVCache:
    """Per-layer cache. k, v: (B, Hk, Tk, D). pos: (Tk,) absolute positions of cached keys."""
    __slots__ = ("k", "v", "pos")

    def __init__(self, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor):
        self.k, self.v, self.pos = k, v, pos

    @property
    def length(self) -> int:
        return self.k.size(2)


class Attention(nn.Module):
    def __init__(self, cfg: PhialConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.n_kv_head, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.rep = self.n_head // self.n_kv_head
        d = cfg.n_embd
        self.wq = nn.Linear(d, self.n_head * self.hd, bias=False)
        self.wk = nn.Linear(d, self.n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(d, self.n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(self.n_head * self.hd, d, bias=False)
        self.wo._is_residual_proj = True
        self.dropout = cfg.dropout

    def _mask(self, qpos: torch.Tensor, kpos: torch.Tensor) -> torch.Tensor:
        """Boolean (Tq, Tk); True = may attend. Causal, optional sliding window, optional sink."""
        allowed = kpos[None, :] <= qpos[:, None]
        w = self.cfg.sliding_window
        if w is not None:
            near = (qpos[:, None] - kpos[None, :]) < w
            if self.cfg.attention_sink > 0:
                near = near | (kpos[None, :] < self.cfg.attention_sink)
            allowed = allowed & near
        return allowed

    def forward(self, x, cos, sin, positions, cache: Optional[KVCache] = None, use_cache: bool = False):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None and cache.length > 0:
            k_all = torch.cat([cache.k, k], dim=2)
            v_all = torch.cat([cache.v, v], dim=2)
            kpos = torch.cat([cache.pos, positions])
        else:
            k_all, v_all, kpos = k, v, positions

        need_mask = (cache is not None and cache.length > 0) or self.cfg.sliding_window is not None
        k_att = k_all.repeat_interleave(self.rep, dim=1) if self.rep > 1 else k_all
        v_att = v_all.repeat_interleave(self.rep, dim=1) if self.rep > 1 else v_all
        y = F.scaled_dot_product_attention(
            q, k_att, v_att,
            attn_mask=self._mask(positions, kpos) if need_mask else None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=not need_mask,
        )
        y = self.wo(y.transpose(1, 2).reshape(B, T, self.n_head * self.hd))

        new_cache = None
        if use_cache:
            w, s = self.cfg.sliding_window, self.cfg.attention_sink
            if w is not None and k_all.size(2) > s + w:
                # evict AFTER attending: keep sink + last w keys; exactly what any later query can still see
                k_all = torch.cat([k_all[:, :, :s], k_all[:, :, -w:]], dim=2)
                v_all = torch.cat([v_all[:, :, :s], v_all[:, :, -w:]], dim=2)
                kpos = torch.cat([kpos[:s], kpos[-w:]])
            new_cache = KVCache(k_all, v_all, kpos)
        return y, new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: PhialConfig):
        super().__init__()
        d, h = cfg.n_embd, cfg.resolved_ffn_hidden
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        self.w_down._is_residual_proj = True

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GeluFFN(nn.Module):
    def __init__(self, cfg: PhialConfig):
        super().__init__()
        d, h = cfg.n_embd, cfg.resolved_ffn_hidden
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        self.w_down._is_residual_proj = True

    def forward(self, x):
        return self.w_down(F.gelu(self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: PhialConfig):
        super().__init__()
        self.norm1, self.attn = make_norm(cfg), Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.ffn = SwiGLU(cfg) if cfg.ffn_type == "swiglu" else GeluFFN(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, positions, cache=None, use_cache=False):
        a, new_cache = self.attn(self.norm1(x), cos, sin, positions, cache, use_cache)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_cache

    def forward_train(self, x, cos, sin, positions):
        return self.forward(x, cos, sin, positions)[0]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class PhialModel(nn.Module):
    def __init__(self, cfg: PhialConfig):
        super().__init__()
        self.cfg = cfg
        self.validate_inputs = True   # token-range check forces a GPU sync; training loops may set False
        d = cfg.n_embd
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.block_size, d) if cfg.pos_type == "learned" else None
        if cfg.pos_type == "sinusoidal":
            # scaled to the same magnitude as token embeddings so neither dominates at init
            self.register_buffer("sin_table", sinusoidal_table(cfg.block_size, d) * cfg.init_std, persistent=False)
        if cfg.pos_type == "rope":
            inv = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
            self.register_buffer("inv_freq", inv, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg)               # final norm is ALWAYS present
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for m in self.modules():                   # GPT-2 style residual scaling
            if getattr(m, "_is_residual_proj", False):
                nn.init.normal_(m.weight, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())   # parameters() de-duplicates tied weights

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                caches: Optional[List[KVCache]] = None, start_pos: int = 0,
                use_cache: bool = False, ignore_index: int = -100):
        if idx.dim() != 2 or idx.dtype != torch.long:
            raise ValueError(f"idx must be a (B,T) long tensor, got {tuple(idx.shape)} {idx.dtype}")
        B, T = idx.shape
        if T < 1:
            raise ValueError("empty input sequence")
        cfg = self.cfg
        end = start_pos + T
        if (cfg.pos_type != "rope" or cfg.sliding_window is None) and end > cfg.block_size:
            raise ValueError(f"position {end} exceeds block_size {cfg.block_size}")
        if self.validate_inputs and (int(idx.max()) >= cfg.vocab_size or int(idx.min()) < 0):
            raise ValueError("token id out of range")
        positions = torch.arange(start_pos, end, device=idx.device)

        x = self.tok_emb(idx)
        cos = sin = None
        if cfg.pos_type == "learned":
            x = x + self.pos_emb(positions)
        elif cfg.pos_type == "sinusoidal":
            x = x + self.sin_table[positions].to(x.dtype)
        else:
            cos, sin = rope_cos_sin(positions, self.inv_freq)
        x = self.drop(x)

        new_caches: Optional[List[KVCache]] = [] if use_cache else None
        for i, blk in enumerate(self.blocks):
            if cfg.grad_checkpoint and self.training and not use_cache and caches is None:
                x = checkpoint(blk.forward_train, x, cos, sin, positions, use_reentrant=False)
            else:
                x, c = blk(x, cos, sin, positions, None if caches is None else caches[i], use_cache)
                if use_cache:
                    new_caches.append(c)
        logits = self.lm_head(self.norm_f(x))

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(f"targets {tuple(targets.shape)} != idx {tuple(idx.shape)}")
            lf = _up(logits)
            total = F.cross_entropy(lf.reshape(-1, lf.size(-1)), targets.reshape(-1),
                                    ignore_index=ignore_index, reduction="sum")
            n_valid = (targets != ignore_index).sum().clamp(min=1)
            loss = total / n_valid     # 0 (finite) if every target is masked
        return logits, loss, new_caches

    # ---- persistence -------------------------------------------------------------------
    def save_pretrained(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model_config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        torch.save(self.state_dict(), d / "model_weights.pt")

    @classmethod
    def from_pretrained(cls, directory, map_location="cpu") -> "PhialModel":
        d = Path(directory)
        cfg = PhialConfig.from_dict(json.loads((d / "model_config.json").read_text()))
        model = cls(cfg)
        sd = torch.load(d / "model_weights.pt", map_location=map_location)
        model.load_state_dict(sd, strict=True)
        return model
