"""Offline fake-gateway checks for the external Hyperspace probe."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stego_r"))
from pod_probe import run_probe, validate_base_url  # noqa: E402


class PodProbeTests(unittest.TestCase):
    def test_url_security(self):
        self.assertEqual(validate_base_url("http://localhost:8080/v1/"),
                         "http://localhost:8080/v1")
        self.assertEqual(validate_base_url("https://pod.example/v1"), "https://pod.example/v1")
        for url in ("http://pod.example/v1", "http://user:pw@localhost/v1",
                    "http://localhost:8080/v1?key=secret", "file:///v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_base_url(url)

    def test_preflight_and_completion_receipt(self):
        data = [
            b'{"data":[{"id":"qwen3.6-27b"}]}',
            json.dumps({"model": "qwen3.6-27b", "choices": [{
                "message": {"content": "1 2 3"}, "finish_reason": "stop"
            }], "usage": {"completion_tokens": 4}}).encode(),
        ]
        with patch("pod_probe.urllib.request.urlopen",
                   side_effect=[io.BytesIO(v) for v in data]) as mock:
            with tempfile.TemporaryDirectory() as tmp:
                transcript = Path(tmp) / "sample.json"
                transcript.write_bytes(b'{"fingerprint":"abc"}\n')
                receipt = run_probe("http://127.0.0.1:8080/v1", "qwen3.6-27b",
                                    "secret-key", repetitions=1, transcript=transcript)
            self.assertEqual(receipt["transcript_sha256"],
                             hashlib.sha256(b'{"fingerprint":"abc"}\n').hexdigest())
            self.assertEqual(receipt["samples"][0]["completion_tokens"], 4)
            self.assertEqual(receipt["samples"][0]["reply_chars"], 5)
            self.assertIsNotNone(receipt["latency_median_s"])
            self.assertNotIn("secret-key", json.dumps(receipt))
            req1 = mock.call_args_list[0].args[0]
            req2 = mock.call_args_list[1].args[0]
            self.assertEqual(req1.get_header("Authorization"), "Bearer secret-key")
            self.assertEqual(req2.get_method(), "POST")
            self.assertEqual(json.loads(req2.data)["model"], "qwen3.6-27b")

    def test_missing_model_stops_before_generation(self):
        with patch("pod_probe.urllib.request.urlopen",
                   return_value=io.BytesIO(b'{"data":[{"id":"another-model"}]}')) as mock:
            with self.assertRaises(RuntimeError):
                run_probe("http://localhost:8080/v1", "qwen3.6-27b", "key", repetitions=3)
            self.assertEqual(mock.call_count, 1)

    def test_preflight_only_does_not_generate(self):
        with patch("pod_probe.urllib.request.urlopen",
                   return_value=io.BytesIO(b'{"data":[{"id":"qwen3.6-27b"}]}')) as mock:
            result = run_probe("http://localhost:8080/v1", "qwen3.6-27b", "key", repetitions=0)
            self.assertEqual(result["samples"], [])
            self.assertIsNone(result["latency_median_s"])
            self.assertEqual(mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
