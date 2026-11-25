"""Defensive measurement invariants for the finite transcript audit."""
from fractions import Fraction
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stateful_observability import (
    DEPENDENT_ROWS, INDEPENDENT_ROWS, bayes_accuracy, conditional_law,
    information, law, reference, total_variation,
)


class StatefulObservabilityTests(unittest.TestCase):
    def test_each_marginal_is_secret_independent(self):
        for col in (1, 2):
            self.assertEqual(conditional_law(DEPENDENT_ROWS, (col,), (0,), (0,)),
                             conditional_law(DEPENDENT_ROWS, (col,), (0,), (1,)))

    def test_joint_conditional_laws_are_disjoint(self):
        a = conditional_law(DEPENDENT_ROWS, (1, 2), (0,), (0,))
        b = conditional_law(DEPENDENT_ROWS, (1, 2), (0,), (1,))
        self.assertEqual(total_variation(a, b), 1)

    def test_conditional_chain_rule(self):
        self.assertEqual(information(DEPENDENT_ROWS, (0,), (1, 2)),
                         information(DEPENDENT_ROWS, (0,), (1,)) +
                         information(DEPENDENT_ROWS, (0,), (2,), (1,)))
        self.assertEqual(information(DEPENDENT_ROWS, (0,), (2,), (1,)), 1)

    def test_marginal_zero_does_not_imply_joint_zero(self):
        self.assertEqual(information(DEPENDENT_ROWS, (0,), (1,)), 0)
        self.assertEqual(information(DEPENDENT_ROWS, (0,), (2,)), 0)
        self.assertEqual(information(DEPENDENT_ROWS, (0,), (1, 2)), 1)

    def test_detection_and_secret_recovery_are_separate(self):
        self.assertEqual(law(DEPENDENT_ROWS, (1, 2)), law(INDEPENDENT_ROWS, (1, 2)))
        self.assertEqual(bayes_accuracy(DEPENDENT_ROWS, (0,), (1, 2)), 1)
        self.assertEqual(bayes_accuracy(INDEPENDENT_ROWS, (0,), (1, 2)), Fraction(1, 2))

    def test_single_view_recovery_is_chance(self):
        for col in (1, 2):
            self.assertEqual(bayes_accuracy(DEPENDENT_ROWS, (0,), (col,)), Fraction(1, 2))

    def test_independent_table_has_no_transcript_leakage(self):
        self.assertEqual(information(INDEPENDENT_ROWS, (0,), (1, 2)), 0)
        self.assertEqual(information(INDEPENDENT_ROWS, (0,), (2,), (1,)), 0)

    def test_invalid_access_and_zero_probability_events(self):
        with self.assertRaises(ValueError):
            law(DEPENDENT_ROWS, (3,))
        with self.assertRaises(ValueError):
            law(DEPENDENT_ROWS, (1, 1))
        with self.assertRaises(ValueError):
            law((), (0,))
        with self.assertRaises(ValueError):
            conditional_law(DEPENDENT_ROWS, (1,), (0,), (9,))
        with self.assertRaises(ValueError):
            information(DEPENDENT_ROWS, (0,), (0,))
        with self.assertRaises(ValueError):
            total_variation({0: Fraction(2)}, {0: Fraction(1)})

    def test_frozen_receipt(self):
        expected = json.loads((Path(__file__).parent / "results/stateful_observability.json").read_text())
        self.assertEqual(reference(), expected)


if __name__ == "__main__":
    unittest.main()
