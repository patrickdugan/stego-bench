"""External-only Hyperspace Pod gateway preflight and end-to-end latency probe.

Uses the public OpenAI-compatible /v1/models and /v1/chat/completions interface.
Measures gateway-visible latency, not libp2p activation throughput or which node
executed a request. Kept in stego_r to preserve the sealed scorer's no-I/O rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def validate_base_url(url: str) -> str:
    url = url.rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.path.endswith("/v1")):
        raise ValueError("pod gateway URL must end in /v1 and contain no credentials or query")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("remote gateways must use HTTPS")
    return url


def _json_request(url: str, key: str, payload: dict | None = None, timeout: float = 180.0) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, headers=headers,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            result = json.load(res)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"pod gateway returned HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("pod gateway connection failed") from None
    if not isinstance(result, dict):
        raise RuntimeError("gateway returned a non-object JSON response")
    return result


def run_probe(base_url: str, model: str, key: str, repetitions: int = 3,
              max_tokens: int = 64, transcript: Path | None = None,
              deployment: str = "pod") -> dict:
    """Validate exact model ID, issue bounded requests, return secret-free receipt."""
    base_url = validate_base_url(base_url)
    if not model or not key:
        raise ValueError("model and API key are required")
    if not 0 <= repetitions <= 100 or not 1 <= max_tokens <= 8192:
        raise ValueError("repetitions must be 0..100 and max_tokens 1..8192")

    catalog = _json_request(base_url + "/models", key, timeout=15)
    data = catalog.get("data")
    if not isinstance(data, list):
        raise RuntimeError("invalid /v1/models response")
    ids = sorted(x["id"] for x in data if isinstance(x, dict) and isinstance(x.get("id"), str))
    if model not in ids:
        raise RuntimeError(f"requested model {model!r} missing from pod catalog: {ids[:12]}")

    samples = []
    for _ in range(repetitions):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "List the integers from 1 to 16 in order."}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        start = time.perf_counter()
        response = _json_request(base_url + "/chat/completions", key, payload)
        elapsed = time.perf_counter() - start
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("pod gateway sent an invalid chat completion") from None
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise RuntimeError("pod gateway returned non-text completion content")
        usage = response.get("usage") or {}
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if not isinstance(tokens, int) or tokens < 0:
            tokens = None
        samples.append({
            "latency_s": round(elapsed, 6),
            "completion_tokens": tokens,
            "tokens_per_s_e2e": round(tokens / elapsed, 3) if tokens is not None and elapsed else None,
            "reply_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "reply_chars": len(content),
            "finish_reason": choice.get("finish_reason"),
            "returned_model": response.get("model"),
        })
    report = {
        "schema": "hyperspace-pod-probe/v1",
        "deployment_label": deployment,
        "base_url": base_url,
        "requested_model": model,
        "advertised": True,
        "repetitions": repetitions,
        "max_tokens": max_tokens,
        "latency_median_s": round(statistics.median(s["latency_s"] for s in samples), 6)
                            if samples else None,
        "samples": samples,
    }
    if transcript is not None:
        report["transcript_sha256"] = hashlib.sha256(transcript.read_bytes()).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="HYPERSPACE_POD_API_KEY")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--deployment", default="pod")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--transcript", type=Path, help="bind receipt to exact transcript bytes")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    key = os.environ.get(args.api_key_env)
    if not key:
        p.error(f"set {args.api_key_env} before probing the pod")
    report = run_probe(
        args.base_url, args.model, key,
        repetitions=0 if args.preflight_only else args.repetitions,
        max_tokens=args.max_tokens,
        deployment=args.deployment,
        transcript=args.transcript,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pod model {args.model}: {report['repetitions']} requests; report {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
