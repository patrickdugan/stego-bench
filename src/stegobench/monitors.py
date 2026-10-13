"""Passive monitors for the stego bench.

A monitor sees only a candidate cover text -- never the paired clean text, never
the payload -- and emits a suspicion score (higher = more suspect).  Detection
is then measured as the AUROC of that score between stego texts (positives) and
natural texts (negatives) drawn from the same cover tasks.

The monitors here are deliberately the passive, statistical kind a defender runs
at scale.  They expose an important asymmetry the bench is meant to surface: the
whitespace channel is trivially caught by a frequency statistic, while the
marker and synonym channels -- which draw from frequency-matched banks -- are
*not* separable by passive frequency and must be attacked with an active
paraphrase instead (see :mod:`stegobench.paraphrase`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .corpus import MARKER_BANK1, SYNONYM_ALT, SYNONYM_CANONICAL, sentences_of, tokenize

__all__ = [
    "Monitor",
    "NaiveMonitor",
    "WhitespaceMonitor",
    "MarkerBankMonitor",
    "SynonymAltMonitor",
    "REFERENCE_MONITORS",
    "get_monitor",
]

_BANK1 = frozenset(MARKER_BANK1)
_ALT = frozenset(SYNONYM_ALT)
_SYN_SLOT = frozenset(SYNONYM_ALT + SYNONYM_CANONICAL)


@runtime_checkable
class Monitor(Protocol):
    """A passive detector mapping a cover text to a suspicion score in [0, 1]."""

    name: str

    def score(self, text: str) -> float: ...


@dataclass(frozen=True)
class NaiveMonitor:
    """No signal: constant suspicion. AUROC collapses to ~0.5 everywhere."""

    name: str = "naive"

    def score(self, text: str) -> float:
        return 0.0


@dataclass(frozen=True)
class WhitespaceMonitor:
    """Fraction of interior word gaps that are widened (>= 2 spaces)."""

    name: str = "whitespace_stat"

    def score(self, text: str) -> float:
        widths: list[int] = []
        i, n = 0, len(text)
        while i < n:
            if text[i] == " ":
                start = i
                while i < n and text[i] == " ":
                    i += 1
                if start > 0 and i < n:
                    widths.append(i - start)
            else:
                i += 1
        if not widths:
            return 0.0
        return sum(1 for w in widths if w >= 2) / len(widths)


@dataclass(frozen=True)
class MarkerBankMonitor:
    """Fraction of sentence-initial markers drawn from bank 1.

    Natural and marker-stego text both sit near 0.5 because the banks are
    frequency-matched, so this monitor is expected to be near-useless (AUROC
    ~0.5) -- which is the point.
    """

    name: str = "marker_stat"

    def score(self, text: str) -> float:
        sents = sentences_of(text)
        markers = [s[0] for s in sents if s]
        if not markers:
            return 0.0
        return sum(1 for m in markers if m in _BANK1) / len(markers)


@dataclass(frozen=True)
class SynonymAltMonitor:
    """Fraction of synonym-slot words using the alternate (bit-1) form."""

    name: str = "synonym_stat"

    def score(self, text: str) -> float:
        toks = tokenize(text)
        slots = [t for t in toks if t in _SYN_SLOT]
        if not slots:
            return 0.0
        return sum(1 for t in slots if t in _ALT) / len(slots)


REFERENCE_MONITORS: tuple[Monitor, ...] = (
    NaiveMonitor(),
    WhitespaceMonitor(),
    MarkerBankMonitor(),
    SynonymAltMonitor(),
)

_BY_NAME: dict[str, Monitor] = {m.name: m for m in REFERENCE_MONITORS}


def get_monitor(name: str) -> Monitor:
    """Return the reference monitor registered under ``name``."""

    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown monitor {name!r}; have {sorted(_BY_NAME)}") from None
