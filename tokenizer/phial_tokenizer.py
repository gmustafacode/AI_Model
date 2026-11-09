"""PHIAL tokenizers: byte-level BPE (trained on the PHIAL corpus) and a byte-level baseline.

Special-token ids are FIXED and identical for both tokenizers, so no downstream stage needs magic numbers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",   # <|end|> closes one chat turn
]
PAD, BOS, EOS, UNK, SYSTEM, USER, ASSISTANT, END = range(8)
N_SPECIAL = len(SPECIAL_TOKENS)


class _Base:
    pad_id, bos_id, eos_id, unk_id = PAD, BOS, EOS, UNK
    system_id, user_id, assistant_id, end_id = SYSTEM, USER, ASSISTANT, END

    @staticmethod
    def escape_specials(text: str) -> str:
        """Neutralise special-token look-alikes inside untrusted text so users cannot inject chat markers."""
        return text.replace("<|", "< |")


class ByteTokenizer(_Base):
    """Baseline: ids 0..7 = specials, 8..263 = raw bytes. Lossless for any text."""

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + 256

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [b + N_SPECIAL for b in text.encode("utf-8")]
        return ([BOS] if add_bos else []) + ids + ([EOS] if add_eos else [])

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = bytearray()
        for i in ids:
            if i >= N_SPECIAL:
                out.append(i - N_SPECIAL)
            elif not skip_special:
                out.extend(SPECIAL_TOKENS[i].encode("utf-8"))
        return out.decode("utf-8", errors="replace")


class PhialTokenizer(_Base):
    """Wrapper over a HuggingFace `tokenizers` byte-level BPE."""

    def __init__(self, tok):
        self._tok = tok
        for i, name in enumerate(SPECIAL_TOKENS):
            got = tok.token_to_id(name)
            if got != i:
                raise RuntimeError(f"special token {name} has id {got}, expected {i}")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_frequency: int = 2, show_progress: bool = True):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=show_progress,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text).ids
        return ([BOS] if add_bos else []) + ids + ([EOS] if add_eos else [])

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self._tok.encode_batch(texts)]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(d / "tokenizer.json"))
        (d / "tokenizer_config.json").write_text(json.dumps({
            "type": "byte_level_bpe", "vocab_size": self.vocab_size,
            "special_tokens": {n: i for i, n in enumerate(SPECIAL_TOKENS)},
        }, indent=2))

    @classmethod
    def load(cls, directory) -> "PhialTokenizer":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(Path(directory) / "tokenizer.json")))
