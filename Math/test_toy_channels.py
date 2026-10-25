"""Checks exhaustively enumerated toy channels. Requires only stdlib."""
import unittest
from toy_channels import (
    conditional_output_law, is_noninterfering, keyed_cover,
    mutual_information_bits, secret_blind_renderer,
    secret_selected_field, whitespace_side_channel,
)


class FiniteNoninterferenceTests(unittest.TestCase):
    def test_cover_distribution_matches_without_key(self):
        self.assertEqual(
            conditional_output_law(keyed_cover, 0),
            conditional_output_law(keyed_cover, 1),
        )
        self.assertTrue(is_noninterfering(keyed_cover, False))
        self.assertEqual(mutual_information_bits(keyed_cover, False), 0.0)

    def test_key_recovers_full_hidden_bit(self):
        for s in (0, 1):
            for k in (0, 1):
                y = 0 if keyed_cover(s, k) == "Yes." else 1
                self.assertEqual(y ^ k, s)
        self.assertFalse(is_noninterfering(keyed_cover, True))
        self.assertEqual(mutual_information_bits(keyed_cover, True), 1.0)

    def test_secret_blind_renderer(self):
        for knows_key in (False, True):
            self.assertTrue(is_noninterfering(secret_blind_renderer, knows_key))
            self.assertEqual(mutual_information_bits(secret_blind_renderer, knows_key), 0.0)

    def test_canonical_json_still_leaks_selected_content(self):
        self.assertFalse(is_noninterfering(secret_selected_field, True))
        self.assertEqual(mutual_information_bits(secret_selected_field, True), 1.0)

    def test_one_extra_output_byte_is_a_side_channel(self):
        self.assertFalse(is_noninterfering(whitespace_side_channel, True))
        self.assertEqual(mutual_information_bits(whitespace_side_channel, True), 1.0)


if __name__ == "__main__":
    unittest.main()
