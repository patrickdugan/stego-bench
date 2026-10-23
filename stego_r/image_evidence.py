"""Audit-grade blind evidence checks for the external Suite 4 image runner.

Never sends network requests. Validates saved prompts, images, identities, and
the exact preregistered case roster before sealing a blind decoder panel.
The seal is not a cryptographic signature; it detects accidental or deliberate
post-seal file mutation as long as the original seal is independently retained.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from stego_r.image_capture_core import (
    CLEAN_PROMPT, DECODE_PROMPT, ENCODE_PROMPT, MONITOR_PROMPT,
    extract_final, extract_svg, parse_monitor_score, validate_svg,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_call(response: dict, spec: dict, prompt: str, png: bytes | None, *, context: str) -> None:
    """Compare captured model-call evidence with the frozen exact prompt/PNG."""
    if not isinstance(response, dict):
        raise ValueError(f"{context}: missing recorded response")
    checks: dict[str, Any] = {
        "requested_model": spec["model"],
        "max_tokens": int(spec.get("max_tokens", 2048)),
        "temperature": float(spec.get("temperature", 0)),
        "prompt_sha256": sha(prompt.encode("utf-8")),
        "input_png_sha256": sha(png) if png is not None else None,
    }
    for key, expected in checks.items():
        if response.get(key, object()) != expected:
            raise ValueError(f"{context}: saved {key} differs from preregistered request")
    expected_backend = spec.get("expected_returned_model", spec["model"])
    if response.get("returned_model") != expected_backend:
        raise ValueError(
            f"{context}: backend reported {response.get('returned_model')!r}, "
            f"expected {expected_backend!r}; pin the model alias before planning"
        )
    if not isinstance(response.get("text"), str):
        raise ValueError(f"{context}: missing raw model text")


def verify_blind(workdir: Path, config: dict, endpoints: dict, manifest: dict) -> dict:
    """Return hash-only witness for EVERY planned capture and blind model call.

    Audited response identity and absence of source leakage are based on the
    runner's recorded request hashes and its fixed message structure. External
    model providers and later human rule auditing remain separate trust edges.
    """
    expected = {c["case_id"] for c in manifest["cases"]}
    if len(expected) != len(manifest["cases"]):
        raise ValueError("duplicate planned case IDs")
    raw_dir = workdir / "raw"
    evidence = {}
    for c in manifest["cases"]:
        cid = c["case_id"]
        failure_path = raw_dir / (cid + ".failure.json")
        raw_path = raw_dir / (cid + ".json")
        prompt = ENCODE_PROMPT.format(message=c["payload"]) if c["stego"] else CLEAN_PROMPT
        spec = endpoints["models"][c["encoder_model"]]
        if failure_path.exists():
            if raw_path.exists():
                raise ValueError(f"{cid}: both encoded output and failure present")
            failure = read_json(failure_path)
            if any(failure.get(k) != c[k] for k in ("case_id", "encoder_model", "stego", "payload")):
                raise ValueError(f"{cid}: failed-case record differs from manifest")
            verify_call(failure.get("encoder_call"), spec, prompt, None, context=f"{cid}:encoder")
            if failure.get("encoder_text_sha256") != sha(failure["encoder_call"]["text"].encode()):
                raise ValueError(f"{cid}: failed-case encoder text hash changed")
            if failure.get("failure_kind") not in ("invalid_svg", "render_error"):
                raise ValueError(f"{cid}: invalid failure kind")
            evidence[cid] = {
                "kind": failure["failure_kind"],
                "raw_sha256": file_sha(failure_path),
                "encoder_text_sha256": failure["encoder_text_sha256"],
            }
            continue
        if not raw_path.exists():
            raise ValueError(f"{cid}: blind capture incomplete, missing encoder output")
        raw = read_json(raw_path)
        if any(raw.get(k) != c[k] for k in ("case_id", "encoder_model", "stego", "payload")):
            raise ValueError(f"{cid}: captured record differs from manifest")
        svg_path = workdir / "images" / (cid + ".svg")
        png_path = workdir / "images" / (cid + ".png")
        if not svg_path.exists() or not png_path.exists():
            raise ValueError(f"{cid}: missing SVG or PNG for blinded case")
        svg = svg_path.read_text(encoding="utf-8")
        png = png_path.read_bytes()
        validate_svg(svg, c["payload"])
        if file_sha(svg_path) != raw.get("svg_sha256") or sha(png) != raw.get("image_sha256"):
            raise ValueError(f"{cid}: generated image bytes changed")
        verify_call(raw.get("encoder_call"), spec, prompt, None, context=f"{cid}:encoder")
        if extract_svg(raw["encoder_call"]["text"]) != svg:
            raise ValueError(f"{cid}: actual encoder SVG differs from rendered SVG")
        if set(raw.get("decodes", {})) != set(config["models"]):
            raise ValueError(f"{cid}: incomplete keyless decoder panel")
        if set(raw.get("monitors", {})) != set(config["blind_monitors"]):
            raise ValueError(f"{cid}: incomplete blind monitor panel")
        decoder_evidence = {}
        for model in config["models"]:
            d = raw["decodes"][model]
            verify_call(d.get("response"), endpoints["models"][model], DECODE_PROMPT, png,
                        context=f"{cid}:decoder:{model}")
            answer, valid = extract_final(d["response"]["text"])
            if d.get("answer") != answer or d.get("format_valid") is not valid:
                raise ValueError(f"{cid}: decoder answer differs from preserved raw response")
            decoder_evidence[model] = {
                "answer_sha256": sha(answer.encode("utf-8")),
                "format_valid": valid,
                "raw_text_sha256": sha(d["response"]["text"].encode("utf-8")),
            }
        monitor_evidence = {}
        for mon in config["blind_monitors"]:
            d = raw["monitors"][mon]
            verify_call(d.get("response"), endpoints["monitors"][mon], MONITOR_PROMPT, png,
                        context=f"{cid}:monitor:{mon}")
            score = parse_monitor_score(d["response"]["text"])
            if d.get("score") != score:
                raise ValueError(f"{cid}: monitor score differs from preserved response")
            monitor_evidence[mon] = {
                "score": score,
                "raw_text_sha256": sha(d["response"]["text"].encode("utf-8")),
            }
        evidence[cid] = {
            "kind": "rendered_png", "raw_sha256": file_sha(raw_path),
            "svg_sha256": file_sha(svg_path), "png_sha256": file_sha(png_path),
            "decoder_answers": decoder_evidence, "blind_monitors": monitor_evidence,
        }
    return {
        "schema_version": "1.1",
        "fingerprint": manifest["fingerprint"],
        "manifest_sha256": file_sha(workdir / "manifest.json"),
        "endpoints_sha256": manifest["endpoints_sha256"],
        "cases": evidence,
        "phase": "all_blind_panels_complete_before_rule_disclosure",
    }


def seal_blind(workdir: Path, config: dict, endpoints: dict, manifest: dict) -> dict:
    """Create an immutable-by-convention blind seal after all model captures."""
    lock_path = workdir / "blind_lock.json"
    receipt = verify_blind(workdir, config, endpoints, manifest)
    if lock_path.exists():
        if read_json(lock_path) != receipt:
            raise ValueError("blind lock no longer matches original captured evidence")
        return receipt
    if list((workdir / "raw").glob("*.oracles.json")):
        raise ValueError("cannot seal blind calls after oracle disclosures have started")
    lock_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def require_blind_lock(workdir: Path, config: dict, endpoints: dict, manifest: dict) -> dict:
    """Forbid oracle and finalized metrics unless the blind lock still matches."""
    lock_path = workdir / "blind_lock.json"
    if not lock_path.exists():
        raise ValueError("blind lock required: run 'seal' after full blind capture")
    receipt = verify_blind(workdir, config, endpoints, manifest)
    if read_json(lock_path) != receipt:
        raise ValueError("blind capture changed after seal; cannot proceed to oracle or scoring")
    return receipt


def review_template(workdir: Path, config: dict, endpoints: dict, manifest: dict) -> dict:
    """Generate an UNAPPROVED template; no model gets pre-certified."""
    lock = require_blind_lock(workdir, config, endpoints, manifest)
    cases = {}
    for c in manifest["cases"]:
        if not c["stego"] or lock["cases"][c["case_id"]]["kind"] != "rendered_png":
            continue
        cases[c["case_id"]] = {
            "image_sha256": lock["cases"][c["case_id"]]["png_sha256"],
            "reviewer_id": "",
            "pixel_carrier_verified": None,
            "source_literal_absent": None,
            "method_family": "",
            "verified_decoders": {m: False for m in config["models"]},
            "rule_text": "",
            "payload_free_key_audited": False,
        }
    return {
        "fingerprint": manifest["fingerprint"],
        "blind_lock_sha256": file_sha(workdir / "blind_lock.json"),
        "cases": cases,
        "audit_state": "UNREVIEWED",
    }
