"""Paraphrase defenses for the stego bench.

A paraphrase is an active defense: the monitor does not merely *score* a text,
it *rewrites* it before passing it on, destroying any channel it does not
preserve.  The scripted defenses here are each minimal and targeted so the
survival matrix reads cleanly -- each targeted paraphrase destroys exactly its
own channel while leaving the others' carriers textually intact:

* ``whitespace_normalize`` collapses space runs -> destroys the whitespace
  channel, preserves markers and synonyms.
* ``marker_normalize`` canonicalizes every sentence-initial marker -> destroys
  the marker channel, preserves spacing and synonyms.
* ``synonym_canonicalize`` maps every alternate synonym to its canonical form
  -> destroys the synonym channel, preserves spacing and markers.
* ``full_normalize`` applies all three -> destroys all scripted channels.

"Survival" is then the chance-corrected recovery that a channel retains after a
given paraphrase is applied between encode and decode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .corpus import MARKERS, SYNONYM_ALT, SYNONYM_PAIRS

__all__ = [
    "Paraphrase",
    "IdentityParaphrase",
    "WhitespaceNormalize",
    "MarkerNormalize",
    "SynonymCanonicalize",
    "FullNormalize",
    "REFERENCE_PARAPHRASES",
    "get_paraphrase",
]

_MARKER_SET = frozenset(MARKERS)
_CANONICAL_MARKER = "Also"
_ALT_TO_CANON: dict[str, str] = {alt: canon for canon, alt in SYNONYM_PAIRS}
_SENTENCE_START = re.compile(r"(^|\.\s+)(\w+)")
_WORD = re.compile(r"\b\w+\b")


@runtime_checkable
class Paraphrase(Protocol):
    """An active defense rewriting a cover text before the receiver sees it."""

    name: str

    def apply(self, text: str) -> str: ...


@dataclass(frozen=True)
class IdentityParaphrase:
    """No-op: the undefended baseline against which survival is measured."""

    name: str = "identity"

    def apply(self, text: str) -> str:
        return text


@dataclass(frozen=True)
class WhitespaceNormalize:
    """Collapse every run of spaces to a single space (and trim)."""

    name: str = "whitespace_normalize"

    def apply(self, text: str) -> str:
        return re.sub(r"[ ]+", " ", text).strip()


@dataclass(frozen=True)
class MarkerNormalize:
    """Rewrite each sentence-initial marker to a single canonical marker."""

    name: str = "marker_normalize"

    def apply(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            lead, word = m.group(1), m.group(2)
            if word in _MARKER_SET:
                return lead + _CANONICAL_MARKER
            return m.group(0)

        return _SENTENCE_START.sub(repl, text)


@dataclass(frozen=True)
class SynonymCanonicalize:
    """Map every alternate synonym back to its canonical (bit-0) form."""

    name: str = "synonym_canonicalize"

    def apply(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            return _ALT_TO_CANON.get(m.group(0), m.group(0))

        return _WORD.sub(repl, text)


@dataclass(frozen=True)
class FullNormalize:
    """Apply whitespace, marker, and synonym normalization in sequence."""

    name: str = "full_normalize"

    def apply(self, text: str) -> str:
        text = SynonymCanonicalize().apply(text)
        text = MarkerNormalize().apply(text)
        text = WhitespaceNormalize().apply(text)
        return text


REFERENCE_PARAPHRASES: tuple[Paraphrase, ...] = (
    IdentityParaphrase(),
    WhitespaceNormalize(),
    MarkerNormalize(),
    SynonymCanonicalize(),
    FullNormalize(),
)

_BY_NAME: dict[str, Paraphrase] = {p.name: p for p in REFERENCE_PARAPHRASES}


def get_paraphrase(name: str) -> Paraphrase:
    """Return the reference paraphrase registered under ``name``."""

    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown paraphrase {name!r}; have {sorted(_BY_NAME)}") from None
