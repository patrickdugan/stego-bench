"""Exhaustive two-bit demonstration of cover matching versus keyed leakage.

This finite illustration is not a verifier for free-form LLM outputs.
S and K are independent fair bits. A and P are held fixed.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from math import log2
from typing import Callable

BITS = (0, 1)
Encoder = Callable[[int, int], str]


def keyed_cover(secret: int, key: int) -> str:
    """The no-key cover distribution is uniform; K reveals the secret."""
    return ("Yes.", "Certainly.")[secret ^ key]


def secret_blind_renderer(secret: int, key: int) -> str:
    """Fixed canonical output, independent of both inputs."""
    return '{"answer":"yes"}'


def secret_selected_field(secret: int, key: int) -> str:
    """Canonical serialization alone permits a secret-selected field."""
    return '{"answer":' + str(secret) + '}'


def whitespace_side_channel(secret: int, key: int) -> str:
    """A single secret-dependent byte changes the observation."""
    return "Yes." + (" " if secret else "")


def conditional_output_law(
    encoder: Encoder, secret: int, known_key: int | None = None
) -> dict[str, Fraction]:
    """Exact P(Y | S=s [,K=k]), averaging uniformly over unknown K."""
    if secret not in BITS or (known_key is not None and known_key not in BITS):
        raise ValueError("secret and key must be bits")
    keys = BITS if known_key is None else (known_key,)
    counts = Counter(encoder(secret, key) for key in keys)
    return {out: Fraction(n, len(keys)) for out, n in counts.items()}


def is_noninterfering(encoder: Encoder, receiver_knows_key: bool = True) -> bool:
    """Exhaustively check equality of the two secret-conditioned laws."""
    key_views = BITS if receiver_knows_key else (None,)
    return all(
        conditional_output_law(encoder, 0, k)
        == conditional_output_law(encoder, 1, k)
        for k in key_views
    )


def _entropy(law: dict[str, Fraction]) -> float:
    return -sum(float(p) * log2(float(p)) for p in law.values() if p)


def _half_mixture(left: dict[str, Fraction], right: dict[str, Fraction]
                  ) -> dict[str, Fraction]:
    return {
        y: (left.get(y, Fraction(0)) + right.get(y, Fraction(0))) / 2
        for y in left.keys() | right.keys()
    }


def mutual_information_bits(encoder: Encoder, receiver_knows_key: bool = True) -> float:
    """Compute I(S;Y|K) or I(S;Y) under independent uniform S and K."""
    key_views = BITS if receiver_knows_key else (None,)
    total = 0.0
    for key in key_views:
        law0 = conditional_output_law(encoder, 0, key)
        law1 = conditional_output_law(encoder, 1, key)
        total += _entropy(_half_mixture(law0, law1))
        total -= (_entropy(law0) + _entropy(law1)) / 2
    return total / len(key_views)


if __name__ == "__main__":
    for name, encoder in (
        ("keyed_cover", keyed_cover),
        ("secret_blind_renderer", secret_blind_renderer),
        ("secret_selected_field", secret_selected_field),
        ("whitespace_side_channel", whitespace_side_channel),
    ):
        print(
            f"{name:26s} "
            f"I(S;Y)={mutual_information_bits(encoder, False):.1f} "
            f"I(S;Y|K)={mutual_information_bits(encoder, True):.1f} "
            f"zero_leakage_with_key={is_noninterfering(encoder, True)}"
        )
