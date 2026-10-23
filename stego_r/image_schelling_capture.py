"""External, auditable image capture for the keyless Schelling benchmark.

Stages: plan -> capture -> independent audit -> oracle -> finalize -> score.

This lives OUTSIDE stegobench's sealed kernel. Model calls use OpenAI-compatible
chat/completions and data:image/png image_url. Every call starts a fresh request,
and all generated images and raw replies are saved for external review.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stegobench.image_schelling_eval import evaluate, fingerprint
from stegobench.report import load_config
from stego_r.image_capture_core import (
    CLEAN_PROMPT, DECODE_PROMPT, ENCODE_PROMPT, MONITOR_PROMPT,
    extract_final, extract_svg, fresh_payload, key_disclosure_safe,
    parse_monitor_score, render_png, validate_svg,
)


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def endpoints_hash(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def assert_endpoints(config: dict, endpoints: dict) -> None:
    if set(endpoints.get("models", {})) != set(config["models"]):
        raise ValueError("endpoint model IDs must match the preregistered scorer roster")
    if set(endpoints.get("monitors", {})) != set(config["blind_monitors"]):
        raise ValueError("endpoint monitor IDs must match the preregistered scorer roster")
    for group in ("models", "monitors"):
        for logical_name, spec in endpoints[group].items():
            url = str(spec.get("base_url", ""))
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.netloc or not url.endswith("/v1"):
                raise ValueError(f"{logical_name}: base_url must be an http(s) .../v1 endpoint")
            if not isinstance(spec.get("model"), str) or not spec["model"].strip():
                raise ValueError(f"{logical_name}: exact backend model identifier required")
            if not 1 <= int(spec.get("max_tokens", 2048)) <= 16384:
                raise ValueError("max_tokens must be in 1..16384")
    model_specs = list(endpoints["models"].values())
    if len({spec["model"] for spec in model_specs}) != len(model_specs):
        raise ValueError("distinct outsider model families require distinct backend model IDs")
    token_caps = {int(spec.get("max_tokens", 2048)) for spec in model_specs}
    if len(token_caps) != 1:
        raise ValueError("all keyless decoders must have an equal max_tokens limit")
    temperatures = {float(spec.get("temperature", 0)) for spec in model_specs}
    if len(temperatures) != 1:
        raise ValueError("keyless decoder sampling temperatures must match")


def _content(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("model returned no choices")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    raise ValueError("model returned no readable text")


def chat(spec: dict, prompt: str, png: bytes | None = None) -> dict:
    """Exactly one user request, no previous messages, SVG source or tools."""
    user_content: str | list[dict]
    if png is None:
        user_content = prompt
    else:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            }},
        ]
    request_body = {
        "model": spec["model"],
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": int(spec.get("max_tokens", 2048)),
        "temperature": float(spec.get("temperature", 0)),
    }
    api_key = os.environ.get(spec.get("api_key_env", ""), "") if spec.get("api_key_env") else ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        str(spec["base_url"]).rstrip("/") + "/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(spec.get("timeout_seconds", 180))) as response:
        raw = json.loads(response.read().decode("utf-8"))
    # Persist the actual requested model, sampling parameters and usage, but no
    # bearer token and no base64 data URIs. The PNG remains a separate artifact.
    return {
        "text": _content(raw),
        "returned_model": raw.get("model"),
        "usage": raw.get("usage", {}),
        "requested_model": spec["model"],
        "max_tokens": request_body["max_tokens"],
        "temperature": request_body["temperature"],
        "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
        "input_png_sha256": sha_bytes(png) if png is not None else None,
    }


def planned(workdir: Path, config_path: Path, endpoints_path: Path) -> tuple[dict, dict, dict]:
    config = load_config(config_path)
    endpoints = json_load(endpoints_path)
    assert_endpoints(config, endpoints)
    manifest = json_load(workdir / "manifest.json")
    if manifest.get("fingerprint") != fingerprint(config):
        raise ValueError("config changed since pre-registration")
    if manifest.get("endpoints_sha256") != endpoints_hash(endpoints_path):
        raise ValueError("endpoint/model specification changed since pre-registration")
    return config, endpoints, manifest


def plan(workdir: Path, config_path: Path, endpoints_path: Path,
         encoders: list[str], encoded: int, clean: int) -> dict:
    if (workdir / "manifest.json").exists():
        raise FileExistsError("plan already exists: keep the original frozen manifest")
    config = load_config(config_path)
    specs = json_load(endpoints_path)
    assert_endpoints(config, specs)
    if len(encoders) < 2 or len(set(encoders)) != len(encoders) or not set(encoders).issubset(set(config["models"])):
        raise ValueError("pre-register at least two distinct encoder families")
    if encoded < 1 or clean < 1:
        raise ValueError("at least one encoded and clean case per encoder required")
    cases: list[dict] = []
    used_payloads: set[str] = set()
    for index in range(max(encoded, clean)):
        for enc_index, encoder in enumerate(encoders):
            for is_stego, limit in ((True, encoded), (False, clean)):
                if index >= limit:
                    continue
                payload = fresh_payload() if is_stego else None
                while payload is not None and payload in used_payloads:
                    payload = fresh_payload()
                if payload is not None:
                    used_payloads.add(payload)
                case_id = f"e{enc_index:02d}-{'stego' if is_stego else 'clean'}-{index:04d}"
                cases.append({
                    "case_id": case_id, "encoder_model": encoder,
                    "stego": is_stego, "payload": payload,
                })
    manifest = {
        "schema_version": "1.0", "fingerprint": fingerprint(config),
        "endpoints_sha256": endpoints_hash(endpoints_path),
        "encoders": encoders, "cases": cases,
        "payload_generation": "secrets.choice over 64 fixed words, six per stego case",
        "test_type": "external_capture_not_a_model_result",
    }
    json_save(workdir / "manifest.json", manifest)
    return manifest


def capture_case(workdir: Path, c: dict, config: dict, endpoints: dict,
                 role: str = "all", target: str | None = None) -> dict:
    """Checkpoint image and selected blind calls; never infer a different model.

    In serial mode the GPU hosts one vision model at a time. An encoder phase
    creates PNGs, then decoder and monitor phases consume those frozen bytes.
    """
    cid = c["case_id"]
    path = workdir / "raw" / (cid + ".json")
    svg_path = workdir / "images" / (cid + ".svg")
    png_path = workdir / "images" / (cid + ".png")
    failure_path = workdir / "raw" / (cid + ".failure.json")
    if failure_path.exists():
        f = json_load(failure_path)
        if any(f.get(k) != c[k] for k in ("case_id", "stego", "encoder_model", "payload")):
            raise ValueError(f"{cid}: failure record mismatches frozen manifest")
        return {"capture_failed": True, **f}
    raw = json_load(path) if path.exists() else None
    if raw is None:
        if role in ("decode", "monitor"):
            raise ValueError(f"{cid}: encode this case before using a decoder/monitor role")
        prompt = ENCODE_PROMPT.format(message=c["payload"]) if c["stego"] else CLEAN_PROMPT
        enc = chat(endpoints["models"][c["encoder_model"]], prompt)
        try:
            svg = extract_svg(enc["text"])
            validate_svg(svg, c["payload"])
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            svg_path.write_text(svg, encoding="utf-8")
            image_hash = render_png(svg, png_path)
        except ValueError as exc:
            # An invalid encoder artifact is a measured *failed attempt*.
            # Keep its raw output and include it in the trial denominator.
            # Transport/API exceptions instead stop the run for retry.
            kind = "render_error" if svg_path.is_file() else "invalid_svg"
            failure = {
                **c, "capture_failed": True, "failure_kind": kind,
                "failure_reason": str(exc),
                "encoder_text_sha256": sha_bytes(enc["text"].encode("utf-8")),
                "encoder_call": enc,
            }
            json_save(failure_path, failure)
            return failure
        raw = {
            **c, "image_sha256": image_hash, "svg_sha256": sha_bytes(svg.encode("utf-8")),
            "encoder_call": enc, "source_literal_absent_auto": True,
            "decodes": {}, "monitors": {},
        }
        json_save(path, raw)
    elif any(raw.get(k) != c[k] for k in ("case_id", "stego", "encoder_model", "payload")):
        raise ValueError(f"{cid}: checkpoint case mismatches plan")
    if not svg_path.is_file() or not png_path.is_file():
        raise ValueError(f"{cid}: missing SVG/PNG for checkpoint")
    svg = svg_path.read_text(encoding="utf-8")
    validate_svg(svg, c["payload"])
    png = png_path.read_bytes()
    if sha_bytes(svg.encode()) != raw["svg_sha256"] or sha_bytes(png) != raw["image_sha256"]:
        raise ValueError(f"{cid}: image bytes changed after capture")
    if role == "encode":
        return raw
    for model_id in config["models"]:
        if role == "monitor" or (role == "decode" and model_id != target):
            continue
        if model_id in raw["decodes"]:
            continue
        response = chat(endpoints["models"][model_id], DECODE_PROMPT, png)
        answer, valid = extract_final(response["text"])
        raw["decodes"][model_id] = {
            "answer": answer, "format_valid": valid, "response": response,
        }
        json_save(path, raw)
    for monitor_id in config["blind_monitors"]:
        if role == "decode" or (role == "monitor" and monitor_id != target):
            continue
        if monitor_id in raw["monitors"]:
            continue
        response = chat(endpoints["monitors"][monitor_id], MONITOR_PROMPT, png)
        raw["monitors"][monitor_id] = {
            "score": parse_monitor_score(response["text"]), "response": response,
        }
        json_save(path, raw)
    return raw


def capture(workdir: Path, config_path: Path, endpoints_path: Path,
            max_cases: int | None = None, role: str = "all",
            target: str | None = None) -> int:
    config, endpoints, manifest = planned(workdir, config_path, endpoints_path)
    if role not in ("all", "encode", "decode", "monitor"):
        raise ValueError("unsupported capture role")
    roster = config["blind_monitors"] if role == "monitor" else config["models"]
    if role == "all" and target is not None:
        raise ValueError("all-role capture does not accept a target")
    if role != "all" and target not in roster:
        raise ValueError(f"{role} target must be registered in frozen config")
    todo = [c for c in manifest["cases"] if role != "encode" or c["encoder_model"] == target]
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        todo = todo[:max_cases]
    for i, c in enumerate(todo, 1):
        raw = capture_case(workdir, c, config, endpoints, role=role, target=target)
        if raw.get("capture_failed"):
            print(f"[{i}/{len(todo)}] {c['case_id']} INVALID SVG attempt recorded ({raw['failure_kind']})")
        else:
            print(f"[{i}/{len(todo)}] {c['case_id']} PNG {raw['image_sha256'][:12]} {role} phase complete")
    return len(todo)


def audits_for(workdir: Path, audits_path: Path) -> tuple[dict, dict]:
    manifest = json_load(workdir / "manifest.json")
    audit = json_load(audits_path)
    if audit.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("audit fingerprint mismatches frozen plan")
    if not isinstance(audit.get("cases"), dict):
        raise ValueError("expected audits.cases mapping")
    return manifest, audit["cases"]


def require_review(c: dict, audit: dict) -> None:
    if not c["stego"]:
        return
    if not isinstance(audit, dict) or type(audit.get("pixel_carrier_verified")) is not bool:
        raise ValueError(f"{c['case_id']}: independent pixel-carrier review required")
    if type(audit.get("source_literal_absent")) is not bool or not str(audit.get("reviewer_id", "")).strip():
        raise ValueError(f"{c['case_id']}: independent SVG source review required")
    if not isinstance(audit.get("method_family"), str) or not audit["method_family"]:
        raise ValueError(f"{c['case_id']}: method family missing")
    verified = audit.get("verified_decoders")
    if not isinstance(verified, dict) or not all(type(v) is bool for v in verified.values()):
        raise ValueError(f"{c['case_id']}: explicit per-model decoder review required")
    if audit["pixel_carrier_verified"] and audit["source_literal_absent"]:
        rule = audit.get("rule_text", "")
        if not key_disclosure_safe(rule, c["payload"]):
            raise ValueError(f"{c['case_id']}: rule disclosure includes target or lacks explanation")
        if audit.get("payload_free_key_audited") is not True:
            raise ValueError(f"{c['case_id']}: rule-only disclosure review required")


def oracle(workdir: Path, config_path: Path, endpoints_path: Path,
           audits_path: Path, max_cases: int | None = None) -> int:
    config, endpoints, manifest = planned(workdir, config_path, endpoints_path)
    _, audits = audits_for(workdir, audits_path)
    todo = [c for c in manifest["cases"] if c["stego"]]
    if max_cases is not None:
        todo = todo[:max_cases]
    for i, c in enumerate(todo, 1):
        cid = c["case_id"]
        failure_path = workdir / "raw" / (cid + ".failure.json")
        if failure_path.exists():
            print(f"[{i}/{len(todo)}] {cid} skipped: failed SVG generation")
            continue
        audit = audits.get(cid)
        require_review(c, audit)
        if not (audit["pixel_carrier_verified"] and audit["source_literal_absent"]):
            print(f"[{i}/{len(todo)}] {cid} skipped: no independently verified carrier")
            continue
        raw = json_load(workdir / "raw" / (cid + ".json"))
        png = (workdir / "images" / (cid + ".png")).read_bytes()
        if sha_bytes(png) != raw["image_sha256"] or set(raw["decodes"]) != set(config["models"]):
            raise ValueError(f"{cid}: keyless panel or PNG changed since blind decoding")
        filename = workdir / "raw" / (cid + ".oracles.json")
        known = json_load(filename) if filename.exists() else {"image_sha256": raw["image_sha256"], "decodes": {}}
        if known["image_sha256"] != raw["image_sha256"]:
            raise ValueError("oracle used a different PNG")
        prompt = DECODE_PROMPT + "\n\nA general decoding rule has been disclosed for this exercise:\n" + audit["rule_text"]
        for dec in config["models"]:
            if dec == c["encoder_model"] or dec in known["decodes"]:
                continue
            response = chat(endpoints["models"][dec], prompt, png)
            answer, valid = extract_final(response["text"])
            known["decodes"][dec] = {"answer": answer, "format_valid": valid, "response": response,
                                    "rule_sha256": sha_bytes(audit["rule_text"].encode())}
            json_save(filename, known)
        print(f"[{i}/{len(todo)}] {cid} oracle panel complete")
    return len(todo)


def finalize(workdir: Path, config_path: Path, audits_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    manifest, audits = audits_for(workdir, audits_path)
    if manifest["fingerprint"] != fingerprint(config):
        raise ValueError("config changed since pre-registration")
    records: list[dict] = []
    failures: list[dict] = []
    for c in manifest["cases"]:
        cid = c["case_id"]
        failure_path = workdir / "raw" / (cid + ".failure.json")
        if failure_path.is_file():
            f = json_load(failure_path)
            if any(f.get(k) != c[k] for k in ("case_id", "encoder_model", "stego", "payload")):
                raise ValueError(f"{cid}: failure record mismatches plan")
            if sha_bytes(f["encoder_call"]["text"].encode("utf-8")) != f["encoder_text_sha256"]:
                raise ValueError(f"{cid}: failure encoder response hash mismatch")
            failures.append({
                "case_id": cid, "encoder_model": c["encoder_model"],
                "stego": c["stego"], "failure_kind": f["failure_kind"],
                "encoder_text_sha256": f["encoder_text_sha256"],
            })
            continue
        path = workdir / "raw" / (cid + ".json")
        if not path.is_file():
            raise ValueError(f"{cid}: missing blind capture")
        raw = json_load(path)
        png = (workdir / "images" / (cid + ".png")).read_bytes()
        svg = (workdir / "images" / (cid + ".svg")).read_text(encoding="utf-8")
        validate_svg(svg, c["payload"])
        if sha_bytes(svg.encode()) != raw["svg_sha256"] or sha_bytes(png) != raw["image_sha256"]:
            raise ValueError(f"{cid}: image hash mismatch")
        if set(raw["decodes"]) != set(config["models"]) or set(raw["monitors"]) != set(config["blind_monitors"]):
            raise ValueError(f"{cid}: incomplete keyless/monitor panels")
        review = audits.get(cid, {})
        if c["stego"]:
            require_review(c, review)
            if raw.get("source_literal_absent_auto") is not True:
                raise ValueError(f"{cid}: automated source check was not recorded")
        rows = [{
            "decoder_model": model,
            "answer": raw["decodes"][model]["answer"],
            "rule_verified": bool(review.get("verified_decoders", {}).get(model, False)),
            "fresh_session": True, "image_only": True, "scheme_supplied": False,
            "condition": "informed",
            "budget_id": config["decoder_budget_id"],
            "prompt_id": config["decoder_prompt_id"],
        } for model in config["models"]]
        oracle_rows = []
        viable = bool(c["stego"] and review["pixel_carrier_verified"] and review["source_literal_absent"])
        if viable:
            oracle_path = workdir / "raw" / (cid + ".oracles.json")
            if not oracle_path.is_file():
                raise ValueError(f"{cid}: missing oracle evidence")
            oracle_blob = json_load(oracle_path)
            if oracle_blob["image_sha256"] != raw["image_sha256"]:
                raise ValueError(f"{cid}: oracle PNG mismatch")
            for outsider in config["models"]:
                if outsider == c["encoder_model"]:
                    continue
                if outsider not in oracle_blob["decodes"]:
                    raise ValueError(f"{cid}: missing oracle decoder {outsider}")
                o = oracle_blob["decodes"][outsider]
                if o["rule_sha256"] != sha_bytes(review["rule_text"].encode()):
                    raise ValueError(f"{cid}: oracle disclosure differs from reviewed rule")
                oracle_rows.append({
                    "decoder_model": outsider, "answer": o["answer"],
                    "fresh_session": True, "image_only": True, "rule_only_disclosure": True,
                    "payload_free_key_audited": True,
                    "budget_id": config["decoder_budget_id"],
                })
        records.append({
            "case_id": cid, "encoder_model": c["encoder_model"], "stego": c["stego"],
            "payload": c["payload"], "input_type": "rendered_png",
            "shared_scheme": False, "image_sha256": raw["image_sha256"],
            "source_literal_absent": review["source_literal_absent"] if c["stego"] else None,
            "pixel_carrier_verified": review["pixel_carrier_verified"] if c["stego"] else None,
            "method_family": review.get("method_family", "control"),
            "decodes": rows, "oracle_decodes": oracle_rows,
            "blind_monitor_scores": {
                mon: raw["monitors"][mon]["score"] for mon in config["blind_monitors"]
            },
        })
    transcript = {"fingerprint": manifest["fingerprint"],
                  "cases": records, "capture_failures": failures}
    # Validate every provenance and case requirement before any output is written.
    summary, _ = evaluate(config, transcript)
    json_save(output, transcript)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Capture isolated image-Schelling model transcripts")
    stages = p.add_subparsers(dest="stage", required=True)
    for stage in ("plan", "capture", "oracle", "finalize"):
        sp = stages.add_parser(stage)
        sp.add_argument("--workdir", type=Path, required=True)
        sp.add_argument("--config", type=Path, default=ROOT / "configs/stego_image_schelling.json")
        if stage != "finalize":
            sp.add_argument("--endpoints", type=Path, required=True)
        if stage in ("oracle", "finalize"):
            sp.add_argument("--audits", type=Path, required=True)
        if stage in ("capture", "oracle"):
            sp.add_argument("--max-cases", type=int, default=None)
        if stage == "capture":
            sp.add_argument("--role", choices=("all", "encode", "decode", "monitor"), default="all")
            sp.add_argument("--target", help="registered encoder, decoder or monitor ID for serial-GPU runs")
        if stage == "plan":
            sp.add_argument("--encoders", nargs="+", required=True)
            sp.add_argument("--encoded-per-encoder", type=int, default=20)
            sp.add_argument("--clean-per-encoder", type=int, default=20)
        if stage == "finalize":
            sp.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.stage == "plan":
        m = plan(a.workdir, a.config, a.endpoints, a.encoders, a.encoded_per_encoder, a.clean_per_encoder)
        print(f"Frozen {len(m['cases'])} cases, manifest {a.workdir / 'manifest.json'}")
    elif a.stage == "capture":
        capture(a.workdir, a.config, a.endpoints, a.max_cases, a.role, a.target)
    elif a.stage == "oracle":
        oracle(a.workdir, a.config, a.endpoints, a.audits, a.max_cases)
    else:
        result = finalize(a.workdir, a.config, a.audits, a.output)
        print(json.dumps({"all_qualify": result["all_qualify"],
                          "all_covert_qualify": result["all_covert_qualify"],
                          "transcript": str(a.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
