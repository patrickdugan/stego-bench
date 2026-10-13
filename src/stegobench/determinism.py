"""Determinism primitives shared across the steganography benchmark.

The stego bench inherits the discipline of the RTG kernel (see
the RTG benchmark's event kernel): every artifact is a pure function of its config and
seeds, canonical JSON is the only serialization, and named RNG streams are
derived by hashing a tuple of string parts.  Nothing in :mod:`stegobench`
is allowed to touch a model library, a clock, or the network -- a stego channel
is only interesting if its measured capacity and detectability are reproducible
to the byte.

These helpers are deliberately a local copy rather than an import from the RTG
kernel: the two benches are independent sealed instruments and must not share
mutable state or create a dependency edge between them.
"""

from __future__ import annotations

import hashlib
import json
from random import Random
from typing import Any

__all__ = ["canon", "digest", "seed_int", "stream"]


def canon(obj: Any) -> str:
    """Return the canonical JSON encoding of ``obj``.

    Keys are sorted and separators are compact, so two structurally equal
    objects always serialize to identical bytes regardless of construction
    order.  This is the only serialization used for hashing.
    """

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(*parts: str) -> str:
    """Return the hex sha256 of ``parts`` joined on ``"|"``.

    The join character cannot appear in a seed part without being escaped by
    the caller; in practice parts are short ASCII identifiers and integers
    rendered as strings, so the join is unambiguous for our use.
    """

    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def seed_int(*parts: str) -> int:
    """Derive a stable 64-bit integer seed from ``parts``.

    Used to seed named RNG streams so that, e.g., the payload stream and the
    cover-selection stream for a given task never accidentally share state.
    """

    return int(digest(*parts)[:16], 16)


def stream(*parts: str) -> Random:
    """Return a fresh :class:`random.Random` seeded from ``parts``."""

    return Random(seed_int(*parts))
