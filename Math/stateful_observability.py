"""STEGO-STATE-01: exact finite checks of receiver-relative transcript leakage.

Audits two fixed probability tables. No text/image encoding, model calls,
network access, or learned transmitter is provided. Columns are (secret,
first observation, second observation); repeated rows are equally weighted.
The example concerns measurement limits, not a deployed steganography result.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import product
import json
from math import log2
from typing import Hashable, Sequence

Row = tuple[int, ...]
Rows = Sequence[Row]
Columns = tuple[int, ...]

# A fixed joint distribution. Each individual observation is independent of
# the uniform secret; the two-observation transcript determines it.
DEPENDENT_ROWS: tuple[Row, ...] = (
    (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),
)
# A null distribution with independent secret and observations. Its marginal
# distribution over complete observable transcripts matches DEPENDENT_ROWS.
INDEPENDENT_ROWS: tuple[Row, ...] = tuple(product((0, 1), repeat=3))


def _validate(rows: Rows, columns: Columns) -> None:
    if not rows:
        raise ValueError("at least one row is required")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("rows must have the same positive width")
    if any(not isinstance(c, int) or c < 0 or c >= width for c in columns):
        raise ValueError("column outside table")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate column in observation")


def law(rows: Rows, columns: Columns) -> dict[tuple[int, ...], Fraction]:
    """Exact marginal law of the selected columns under uniform row weights."""
    _validate(rows, columns)
    counts = Counter(tuple(row[c] for c in columns) for row in rows)
    return {key: Fraction(count, len(rows)) for key, count in counts.items()}


def conditional_law(rows: Rows, observed: Columns, given: Columns,
                    values: tuple[int, ...]) -> dict[tuple[int, ...], Fraction]:
    _validate(rows, observed)
    _validate(rows, given)
    if len(given) != len(values):
        raise ValueError("conditioning columns and values must have equal lengths")
    subset = tuple(row for row in rows
                   if tuple(row[c] for c in given) == values)
    if not subset:
        raise ValueError("conditioning event has zero probability")
    return law(subset, observed)


def total_variation(left: dict[Hashable, Fraction],
                    right: dict[Hashable, Fraction]) -> Fraction:
    if (sum(left.values()) != 1 or sum(right.values()) != 1
            or any(x < 0 for x in (*left.values(), *right.values()))):
        raise ValueError("inputs must be normalized nonnegative distributions")
    return sum((abs(left.get(k, Fraction(0)) - right.get(k, Fraction(0)))
                for k in left.keys() | right.keys()), Fraction(0)) / 2


def entropy(rows: Rows, columns: Columns) -> float:
    return -sum(float(p) * log2(float(p)) for p in law(rows, columns).values())


def information(rows: Rows, target: Columns, observed: Columns,
                given: Columns = ()) -> float:
    """Conditional mutual information in bits, from the complete finite law.

    This is a population calculation on explicitly tabulated variables.
    It is not an estimator for high-dimensional model transcripts.
    """
    if set(target) & set(observed) or (set(target) | set(observed)) & set(given):
        raise ValueError("target, observations, and conditioning must be disjoint")
    _validate(rows, target + observed + given)
    value = (entropy(rows, target + given) + entropy(rows, observed + given)
             - entropy(rows, given) - entropy(rows, target + observed + given))
    return 0.0 if abs(value) < 1e-12 else value


def bayes_accuracy(rows: Rows, target: Columns, observed: Columns) -> Fraction:
    """Exact best target-identification probability with the selected access."""
    _validate(rows, target + observed)
    groups: dict[tuple[int, ...], Counter] = {}
    for row in rows:
        view = tuple(row[c] for c in observed)
        label = tuple(row[c] for c in target)
        groups.setdefault(view, Counter())[label] += 1
    return Fraction(sum(max(counts.values()) for counts in groups.values()), len(rows))


def reference() -> dict:
    first = information(DEPENDENT_ROWS, (0,), (1,))
    second = information(DEPENDENT_ROWS, (0,), (2,))
    joint = information(DEPENDENT_ROWS, (0,), (1, 2))
    conditional_second = information(DEPENDENT_ROWS, (0,), (2,), (1,))
    presence_tv = total_variation(law(DEPENDENT_ROWS, (1, 2)),
                                 law(INDEPENDENT_ROWS, (1, 2)))
    clean_zero = information(INDEPENDENT_ROWS, (0,), (1, 2))
    assert (first, second, joint, conditional_second, clean_zero) == (0, 0, 1, 1, 0)
    assert presence_tv == 0
    return {
        "experiment": "STEGO-STATE-01 finite reference",
        "evidence": "two explicit finite probability tables, exhaustive calculation",
        "model_call": False,
        "network": False,
        "dependent_table_rows": len(DEPENDENT_ROWS),
        "independent_table_rows": len(INDEPENDENT_ROWS),
        "information_bits": {
            "secret_first": first,
            "secret_second": second,
            "secret_full_transcript": joint,
            "secret_second_given_first": conditional_second,
            "null_secret_full_transcript": clean_zero,
        },
        "optimal_secret_recovery": {
            "first_only": str(bayes_accuracy(DEPENDENT_ROWS, (0,), (1,))),
            "second_only": str(bayes_accuracy(DEPENDENT_ROWS, (0,), (2,))),
            "full_transcript": str(bayes_accuracy(DEPENDENT_ROWS, (0,), (1, 2))),
            "null_full_transcript": str(bayes_accuracy(INDEPENDENT_ROWS, (0,), (1, 2))),
        },
        "presence_detection": {
            "observable_law_total_variation": str(presence_tv),
            "optimal_equal_prior_classification_accuracy": str((1 + presence_tv) / 2),
            "population_auroc_any_deterministic_transcript_score": 0.5,
        },
        "limits": "Distributional identity holds only for this specified null, access and prior. "
                  "Payload recovery and channel-presence classification are distinct tasks. "
                  "No empirical model steganography result or universal detector is claimed.",
    }


if __name__ == "__main__":
    print(json.dumps(reference(), sort_keys=True, indent=2))
