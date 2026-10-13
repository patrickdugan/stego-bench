"""Scripted reference codecs for the stego bench.

These codecs are the *organisms* that validate the measurement instrument.
Each is a known channel with a known capacity, a known detectability profile,
and a known fragility under paraphrase.  Because they are fully deterministic
functions of ``(bits, cover_task)`` -- no model, no RNG -- the acceptance gates
can assert exactly where each lands on the capacity-vs-detectability frontier,
and we know a small-model floor is a real floor rather than a broken scorer.

The four channels are chosen to spread across the frontier:

* ``null`` -- zero capacity, undetectable; the negative control.
* ``whitespace`` -- high capacity (one bit per inter-word gap), trivially
  detectable (a double-space frequency monitor), destroyed by whitespace
  normalization.
* ``marker`` -- capacity one bit per sentence, low detectability (the two
  discourse-marker banks have matched frequency), destroyed by marker
  normalization but surviving whitespace edits.
* ``synonym`` -- capacity one bit per planted slot, low detectability (balanced
  synonym pairs), destroyed by synonym canonicalization but surviving
  whitespace edits.

A codec embeds ``min(len(bits), capacity)`` payload bits; extra payload bits
cannot be carried and are simply not recovered, which is how the payload-length
sweep exposes each channel's capacity knee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .corpus import (
    MARKER_BANK0,
    MARKER_BANK1,
    SYNONYM_ALT,
    SYNONYM_CANONICAL,
    SYNONYM_PAIRS,
    SYNONYM_VALUES,
    CoverTask,
    render,
    sentences_of,
    tokenize,
)

__all__ = [
    "Codec",
    "NullCodec",
    "WhitespaceCodec",
    "MarkerCodec",
    "SynonymCodec",
    "REFERENCE_CODECS",
    "get_codec",
]


@runtime_checkable
class Codec(Protocol):
    """A steganographic channel: embed bits in a cover text, recover them back.

    ``encode`` must be lossless up to :meth:`capacity` and deterministic; a
    receiver that knows the scheme recovers the first ``capacity`` carrier
    positions with :meth:`decode`.  Neither method takes an RNG -- the carrier
    text already fixes every degree of freedom the scheme uses.
    """

    name: str

    def capacity(self, task: CoverTask) -> int: ...

    def encode(self, bits: tuple[int, ...], task: CoverTask) -> str: ...

    def decode(self, text: str, task: CoverTask) -> tuple[int, ...]: ...


# -- Lookups derived once from the carrier vocabularies -------------------------

# marker -> (bank_bit, partner_in_other_bank); the two banks are index-paired.
if len(MARKER_BANK0) != len(MARKER_BANK1):
    raise ValueError("marker banks must be the same length for index pairing")
_MARKER_INFO: dict[str, tuple[int, str]] = {}
for _k in range(len(MARKER_BANK0)):
    _MARKER_INFO[MARKER_BANK0[_k]] = (0, MARKER_BANK1[_k])
    _MARKER_INFO[MARKER_BANK1[_k]] = (1, MARKER_BANK0[_k])

# synonym word -> (pair_index, bit); canonical=0, alt=1.
_SYN_INFO: dict[str, tuple[int, int]] = {}
for _i, (_canon, _alt) in enumerate(SYNONYM_PAIRS):
    _SYN_INFO[_canon] = (_i, 0)
    _SYN_INFO[_alt] = (_i, 1)


@dataclass(frozen=True)
class NullCodec:
    """Zero-capacity control: emits the baseline carrier, hides nothing."""

    name: str = "null"

    def capacity(self, task: CoverTask) -> int:
        return 0

    def encode(self, bits: tuple[int, ...], task: CoverTask) -> str:
        return task.clean_text()

    def decode(self, text: str, task: CoverTask) -> tuple[int, ...]:
        return ()


@dataclass(frozen=True)
class WhitespaceCodec:
    """Encode one bit per inter-word gap: single space = 0, double space = 1."""

    name: str = "whitespace"

    def capacity(self, task: CoverTask) -> int:
        return task.gap_count

    def encode(self, bits: tuple[int, ...], task: CoverTask) -> str:
        tokens = task.clean_text().split(" ")
        if len(tokens) <= 1:
            return task.clean_text()
        out = [tokens[0]]
        for i, tok in enumerate(tokens[1:]):
            bit = bits[i] if i < len(bits) else 0
            out.append(("  " if bit else " ") + tok)
        return "".join(out)

    def decode(self, text: str, task: CoverTask) -> tuple[int, ...]:
        bits: list[int] = []
        i, n = 0, len(text)
        while i < n:
            if text[i] == " ":
                start = i
                while i < n and text[i] == " ":
                    i += 1
                if start > 0 and i < n:  # interior gap only
                    bits.append(1 if (i - start) >= 2 else 0)
            else:
                i += 1
        return tuple(bits)


@dataclass(frozen=True)
class MarkerCodec:
    """Encode one bit per sentence via the discourse-marker bank (0/1)."""

    name: str = "marker"

    def capacity(self, task: CoverTask) -> int:
        return task.sentence_count

    def encode(self, bits: tuple[int, ...], task: CoverTask) -> str:
        sents: list[tuple[str, ...]] = []
        for j, sent in enumerate(task.sentences):
            marker = sent[0]
            if j < len(bits):
                bank_bit, partner = _MARKER_INFO[marker]
                if bank_bit != bits[j]:
                    marker = partner  # minimal edit: swap to index-paired marker
            sents.append((marker,) + sent[1:])
        return render(tuple(sents))

    def decode(self, text: str, task: CoverTask) -> tuple[int, ...]:
        bits: list[int] = []
        for sent in sentences_of(text):
            if not sent:
                continue
            info = _MARKER_INFO.get(sent[0])
            bits.append(info[0] if info is not None else 0)
        return tuple(bits)


@dataclass(frozen=True)
class SynonymCodec:
    """Encode one bit per planted synonym slot: canonical = 0, alternate = 1."""

    name: str = "synonym"

    def capacity(self, task: CoverTask) -> int:
        return task.synonym_slot_count

    def encode(self, bits: tuple[int, ...], task: CoverTask) -> str:
        slot = 0
        sents: list[tuple[str, ...]] = []
        for sent in task.sentences:
            words = list(sent)
            for w_idx, word in enumerate(words):
                if word in SYNONYM_VALUES:
                    pair_idx, _ = _SYN_INFO[word]
                    if slot < len(bits):
                        bit = bits[slot]
                        words[w_idx] = SYNONYM_ALT[pair_idx] if bit else SYNONYM_CANONICAL[pair_idx]
                    slot += 1
            sents.append(tuple(words))
        return render(tuple(sents))

    def decode(self, text: str, task: CoverTask) -> tuple[int, ...]:
        bits: list[int] = []
        for tok in tokenize(text):
            info = _SYN_INFO.get(tok)
            if info is not None:
                bits.append(info[1])
        return tuple(bits)


REFERENCE_CODECS: tuple[Codec, ...] = (
    NullCodec(),
    WhitespaceCodec(),
    MarkerCodec(),
    SynonymCodec(),
)

_BY_NAME: dict[str, Codec] = {c.name: c for c in REFERENCE_CODECS}


def get_codec(name: str) -> Codec:
    """Return the reference codec registered under ``name``."""

    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown codec {name!r}; have {sorted(_BY_NAME)}") from None
