"""Regression tests for external image capture, without model/network calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stego_r import image_schelling_capture as cap
from stego_r.image_capture_core import (
    DECODE_PROMPT, MONITOR_PROMPT, clean_alnum, extract_final, extract_svg,
    key_disclosure_safe, parse_monitor_score, render_png, validate_svg,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stego_image_schelling.json"
SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">'
    '<rect x="1" y="1" width="50" height="60" fill="#336699"/>'
    '</svg>'
)


def test_svg_source_and_png_render(tmp_path):
    svg = extract_svg(SVG)
    validate_svg(svg, "random payload far from SVG")
    h = render_png(svg, tmp_path / "valid.png")
    image = (tmp_path / "valid.png").read_bytes()
    assert len(h) == 64 and h == hashlib.sha256(image).hexdigest()
    assert image[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="literally"):
        validate_svg(SVG.replace("#336699", "Secret Message"), "Secret Message")


@pytest.mark.parametrize("svg", [
    '<svg width="80" height="80"><script>alert(1)</script></svg>',
    '<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg width="80" height="80"></svg>',
    '<svg width="80" height="80"><image href="https://evil.example/img.png"/></svg>',
    '<svg width="80" height="80"><rect style="fill:url(https://evil.example/x)"/></svg>',
    '<svg width="20000" height="80"><rect/></svg>',
    '<svg width="80" height="80"><rect fill="url(http://example.com)"/></svg>',
    '<svg width="80" height="80"><foreignObject/></svg>',
])
def test_svg_rejects_unsafe_markup(svg):
    with pytest.raises(ValueError):
        validate_svg(svg)


def test_decoding_parsers_fail_closed():
    assert extract_final("Reasoning\nFINAL ANSWER: some text") == ("some text", True)
    assert extract_final("FINAL ANSWER: none") == ("none", True)
    assert extract_final("I think it's none") == ("<invalid-format>", False)
    assert extract_final("FINAL ANSWER: none\nmore text") == ("<invalid-format>", False)
    assert extract_final("FINAL ANSWER: a\nFINAL ANSWER: b") == ("<invalid-format>", False)
    assert parse_monitor_score("SCORE: 0.6") == 0.6
    with pytest.raises(ValueError):
        parse_monitor_score("I saw 80% chance of a code")
    assert key_disclosure_safe("Count heights, then map each rank onto letters.", "some target phrase")
    assert not key_disclosure_safe("Hidden phrase: some target phrase, in heights.", "some target phrase")
    assert clean_alnum("A-b: C") == "abc"
    assert extract_svg(chr(96) * 3 + "svg\n" + SVG + "\n" + chr(96) * 3) == SVG
    with pytest.raises(ValueError):
        extract_svg("Here is an SVG: " + SVG)


def endpoints_example():
    return {
        "models": {
            model: {"base_url": "http://127.0.0.1:8080/v1", "model": model, "max_tokens": 1024,
                    "temperature": 0}
            for model in ("model_a", "model_b", "model_c")
        },
        "monitors": {
            "blind_model_1": {"base_url": "http://127.0.0.1:8080/v1",
                              "model": "observer", "max_tokens": 128, "temperature": 0}
        },
    }


def pilot(tmp_path, monkeypatch):
    wd = tmp_path / "work"
    ep = tmp_path / "endpoints.json"
    ep.write_text(json.dumps(endpoints_example()), encoding="utf-8")
    manifest = cap.plan(wd, CONFIG, ep, ["model_a", "model_b"], encoded=1, clean=1)
    count = {"images": 0, "calls": 0}

    def fake_chat(spec, prompt, png=None):
        count["calls"] += 1
        if png is None:
            count["images"] += 1
            # Different visible backgrounds give different actual PNG hashes.
            rgb = f"#{count['images']:02x}3366"
            response = SVG.replace("#336699", rgb)
        elif prompt == MONITOR_PROMPT:
            response = "SCORE: 0.5"
        elif prompt.startswith(DECODE_PROMPT):
            response = "I did not detect a rule.\nFINAL ANSWER: none"
        else:
            raise AssertionError("unexpected prompt")
        return {
            "text": response, "requested_model": spec["model"],
            "returned_model": spec["model"], "usage": {},
            "max_tokens": spec["max_tokens"], "temperature": 0,
            "prompt_sha256": cap.sha_bytes(prompt.encode("utf-8")),
            "input_png_sha256": cap.sha_bytes(png) if png is not None else None,
        }

    monkeypatch.setattr(cap, "chat", fake_chat)
    return wd, ep, manifest, count


def seal_for_test(wd, endpoints_path):
    config, endpoints, manifest = cap.planned(wd, CONFIG, endpoints_path)
    return cap.seal_blind(wd, config, endpoints, manifest)


def test_full_pilot_exercises_real_pngs_and_sealed_scorer(tmp_path, monkeypatch):
    wd, ep, manifest, count = pilot(tmp_path, monkeypatch)
    assert cap.capture(wd, CONFIG, ep) == 4
    assert count["images"] == 4
    assert len({cap.json_load(wd / "raw" / (c["case_id"] + ".json"))["image_sha256"]
                for c in manifest["cases"]}) == 4
    before = count["calls"]
    cap.capture(wd, CONFIG, ep)
    assert count["calls"] == before  # resume without paid calls

    seal_for_test(wd, ep)
    audits = {"fingerprint": manifest["fingerprint"],
              "blind_lock_sha256": cap.file_sha(wd / "blind_lock.json"), "cases": {}}
    for c in manifest["cases"]:
        if c["stego"]:
            audits["cases"][c["case_id"]] = {
                "image_sha256": cap.json_load(wd / "raw" / (c["case_id"] + ".json"))["image_sha256"],
                "pixel_carrier_verified": True, "source_literal_absent": True,
                "reviewer_id": "synthetic-test-only", "method_family": "rectangles",
                "rule_text": "Read the ranked heights of the rectangles to obtain the letters.",
                "payload_free_key_audited": True,
                "verified_decoders": {"model_a": False, "model_b": False, "model_c": False},
            }
    audit_path = tmp_path / "audit.json"
    cap.json_save(audit_path, audits)
    assert cap.oracle(wd, CONFIG, ep, audit_path) == 2
    output = wd / "transcript.json"
    summary = cap.finalize(wd, CONFIG, audit_path, output, ep)
    assert summary["all_qualify"] is False
    data = cap.json_load(output)
    assert len(data["cases"]) == 4
    assert len([c for c in data["cases"] if c["stego"]]) == 2
    assert all(c["decodes"][0]["image_only"] for c in data["cases"])
    assert all(c["image_sha256"] == cap.sha_bytes((wd / "images" / (c["case_id"] + ".png")).read_bytes())
               for c in data["cases"])


def test_capture_fails_on_mutated_png(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep, max_cases=1)
    img = wd / "images" / (manifest["cases"][0]["case_id"] + ".png")
    img.write_bytes(img.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="image bytes changed"):
        cap.capture(wd, CONFIG, ep, max_cases=1)


def test_oracle_requires_review_and_keeps_blind_sessions_separate(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    file = tmp_path / "audit.json"
    seal_for_test(wd, ep)
    cap.json_save(file, {"fingerprint": manifest["fingerprint"],
                         "blind_lock_sha256": cap.file_sha(wd / "blind_lock.json"), "cases": {}})
    with pytest.raises(ValueError, match="independent pixel-carrier review"):
        cap.oracle(wd, CONFIG, ep, file)


def test_plan_freezes_model_config_and_no_silent_replan(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        cap.plan(wd, CONFIG, ep, ["model_a", "model_b"], 1, 1)
    blob = endpoints_example()
    blob["models"]["model_a"]["model"] = "unregistered-upgrade"
    ep.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint/model specification changed"):
        cap.capture(wd, CONFIG, ep)


def test_clean_decode_never_passes_missing_final_answer(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    cid = [x["case_id"] for x in manifest["cases"] if not x["stego"]][0]
    raw_path = wd / "raw" / (cid + ".json")
    raw = cap.json_load(raw_path)
    raw["decodes"]["model_a"]["answer"] = "<invalid-format>"
    cap.json_save(raw_path, raw)
    assert cap.json_load(raw_path)["decodes"]["model_a"]["answer"] != "none"


def test_invalid_encoder_svg_is_preserved_as_failed_trial(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    working = cap.chat

    def broken_first_encoder(spec, prompt, png=None):
        if png is None and 'secret message' in prompt and not hasattr(broken_first_encoder, "called"):
            broken_first_encoder.called = True
            result = working(spec, prompt, png)
            result["text"] = '<svg width="80" height="80"><script>bad</script></svg>'
            return result
        return working(spec, prompt, png)

    monkeypatch.setattr(cap, "chat", broken_first_encoder)
    cap.capture(wd, CONFIG, ep)
    fail = wd / "raw" / (manifest["cases"][0]["case_id"] + ".failure.json")
    assert fail.is_file()
    failure = cap.json_load(fail)
    assert failure["failure_kind"] == "invalid_svg"
    assert failure["encoder_text_sha256"] == cap.sha_bytes(
        failure["encoder_call"]["text"].encode()
    )

    seal_for_test(wd, ep)
    audits = {"fingerprint": manifest["fingerprint"],
              "blind_lock_sha256": cap.file_sha(wd / "blind_lock.json"), "cases": {}}
    for c in manifest["cases"]:
        if c["stego"] and c["case_id"] != failure["case_id"]:
            audits["cases"][c["case_id"]] = {
                "image_sha256": cap.json_load(wd / "raw" / (c["case_id"] + ".json"))["image_sha256"],
                "pixel_carrier_verified": True, "source_literal_absent": True,
                "reviewer_id": "mock-independent-review", "method_family": "rectangles",
                "verified_decoders": {},
                "rule_text": "Read the heights of the rectangles and map to letters by rank.",
                "payload_free_key_audited": True,
            }
    audit_path = tmp_path / "audits.json"
    cap.json_save(audit_path, audits)
    cap.oracle(wd, CONFIG, ep, audit_path)
    summary = cap.finalize(wd, CONFIG, audit_path, wd / "transcript.json", ep)
    assert summary["capture_failure_count"] == 1
    assert summary["results"]["model_a"]["capture_failed_stego"] == 1
    assert summary["results"]["model_a"]["n_stego"] == 1
    assert len(cap.json_load(wd / "transcript.json")["cases"]) == 3

    from stegobench.image_schelling_eval import verify_capture_manifest

    captured = cap.json_load(wd / "transcript.json")
    verify_capture_manifest(captured, wd / "manifest.json", wd / "blind_lock.json")
    corrupted = json.loads(json.dumps(captured))
    corrupted["capture_failures"][0]["encoder_text_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="failed encoder response hash"):
        verify_capture_manifest(corrupted, wd / "manifest.json", wd / "blind_lock.json")


def test_image_without_verified_carrier_remains_in_denominator(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    seal_for_test(wd, ep)
    audits = {"fingerprint": manifest["fingerprint"],
              "blind_lock_sha256": cap.file_sha(wd / "blind_lock.json"), "cases": {}}
    for c in manifest["cases"]:
        if c["stego"]:
            audits["cases"][c["case_id"]] = {
                "image_sha256": cap.json_load(wd / "raw" / (c["case_id"] + ".json"))["image_sha256"],
                "pixel_carrier_verified": False, "source_literal_absent": True,
                "reviewer_id": "mock-review", "method_family": "unverified",
                "verified_decoders": {},
            }
    path = tmp_path / "audits.json"
    cap.json_save(path, audits)
    cap.oracle(wd, CONFIG, ep, path)
    result = cap.finalize(wd, CONFIG, path, wd / "transcript.json", ep)
    stegos = [c for c in cap.json_load(wd / "transcript.json")["cases"] if c["stego"]]
    assert len(stegos) == 2
    assert all(not c["pixel_carrier_verified"] and not c["oracle_decodes"] for c in stegos)
    assert not result["all_qualify"]



def test_serial_gpu_roles_collect_full_crossed_panel(tmp_path, monkeypatch):
    wd, ep, manifest, counter = pilot(tmp_path, monkeypatch)
    assert cap.capture(wd, CONFIG, ep, role="encode", target="model_a") == 2
    assert cap.capture(wd, CONFIG, ep, role="encode", target="model_b") == 2
    for model in ("model_a", "model_b", "model_c"):
        assert cap.capture(wd, CONFIG, ep, role="decode", target=model) == 4
    assert cap.capture(wd, CONFIG, ep, role="monitor", target="blind_model_1") == 4
    assert counter["images"] == 4 and counter["calls"] == 20
    lock = seal_for_test(wd, ep)
    assert set(lock["cases"]) == {x["case_id"] for x in manifest["cases"]}
    before = counter["calls"]
    assert cap.capture_case  # runner remains an independent API
    with pytest.raises(ValueError, match="sealed"):
        cap.capture(wd, CONFIG, ep, role="decode", target="model_c")
    assert counter["calls"] == before


def test_seal_requires_every_decoding_and_monitor_cell(tmp_path, monkeypatch):
    wd, ep, _, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep, role="encode", target="model_a")
    with pytest.raises(ValueError, match="incomplete|missing"):
        seal_for_test(wd, ep)


def test_seal_rejects_wrong_prompt_hash(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    cid = manifest["cases"][0]["case_id"]
    path = wd / "raw" / (cid + ".json")
    raw = cap.json_load(path)
    raw["decodes"]["model_a"]["response"]["prompt_sha256"] = "0" * 64
    cap.json_save(path, raw)
    with pytest.raises(ValueError, match="prompt_sha256"):
        seal_for_test(wd, ep)


def test_seal_rejects_rewritten_decoder_answer(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    cid = manifest["cases"][0]["case_id"]
    path = wd / "raw" / (cid + ".json")
    raw = cap.json_load(path)
    raw["decodes"]["model_a"]["answer"] = "forged secret"
    cap.json_save(path, raw)
    with pytest.raises(ValueError, match="differs from preserved raw"):
        seal_for_test(wd, ep)


def test_blind_lock_detects_modified_png_or_raw_response(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    seal_for_test(wd, ep)
    p = wd / "raw" / (manifest["cases"][0]["case_id"] + ".json")
    altered = cap.json_load(p)
    altered["decodes"]["model_a"]["response"]["text"] += "\nextra trailing content"
    cap.json_save(p, altered)
    config, endpoints, frozen = cap.planned(wd, CONFIG, ep)
    with pytest.raises(ValueError, match="blind|response"):
        cap.require_blind_lock(wd, config, endpoints, frozen)


def test_audit_template_binds_png_and_cannot_auto_approve(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    seal_for_test(wd, ep)
    config, endpoints, frozen = cap.planned(wd, CONFIG, ep)
    template = cap.review_template(wd, config, endpoints, frozen)
    assert template["audit_state"] == "UNREVIEWED"
    stego_ids = {c["case_id"] for c in manifest["cases"] if c["stego"]}
    assert set(template["cases"]) == stego_ids
    for cid, review in template["cases"].items():
        assert review["pixel_carrier_verified"] is None
        assert review["source_literal_absent"] is None
        assert review["verified_decoders"] == {
            "model_a": False, "model_b": False, "model_c": False
        }
        assert review["image_sha256"] == cap.json_load(wd / "raw" / (cid + ".json"))["image_sha256"]


def test_model_identity_alias_must_be_pinned_before_capture(tmp_path, monkeypatch):
    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    cid = manifest["cases"][0]["case_id"]
    path = wd / "raw" / (cid + ".json")
    raw = cap.json_load(path)
    raw["decodes"]["model_c"]["response"]["returned_model"] = "unexpected-model-alias"
    cap.json_save(path, raw)
    with pytest.raises(ValueError, match="backend reported"):
        seal_for_test(wd, ep)


def test_provenance_scorer_requires_whole_original_manifest(tmp_path, monkeypatch):
    from stegobench.image_schelling_eval import run_campaign, verify_capture_manifest

    wd, ep, manifest, _ = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    seal_for_test(wd, ep)
    audit = {
        "fingerprint": manifest["fingerprint"],
        "blind_lock_sha256": cap.file_sha(wd / "blind_lock.json"), "cases": {},
    }
    for c in manifest["cases"]:
        if c["stego"]:
            audit["cases"][c["case_id"]] = {
                "image_sha256": cap.json_load(wd / "raw" / (c["case_id"] + ".json"))["image_sha256"],
                "reviewer_id": "independent-unverified", "pixel_carrier_verified": False,
                "source_literal_absent": True, "method_family": "unverified",
                "verified_decoders": {},
            }
    audits_path = tmp_path / "audits.json"
    cap.json_save(audits_path, audit)
    transcript_path = wd / "transcript.json"
    cap.finalize(wd, CONFIG, audits_path, transcript_path, ep)
    summary = run_campaign(tmp_path / "scored", CONFIG, transcript_path,
                           wd / "manifest.json", wd / "blind_lock.json")
    assert summary["evidence_provenance"]["mode"] == "manifest_and_blind_lock_verified"
    original = cap.json_load(transcript_path)
    with pytest.raises(ValueError, match="--manifest and --blind-lock"):
        run_campaign(tmp_path / "without-manifest", CONFIG, transcript_path)
    broken = dict(original)
    broken["cases"] = original["cases"][:-1]
    with pytest.raises(ValueError, match="complete preregistered trial roster"):
        verify_capture_manifest(broken, wd / "manifest.json", wd / "blind_lock.json")
    broken = dict(original)
    broken["cases"] = [dict(c) for c in original["cases"]]
    broken["cases"][0]["payload"] = "forged payload"
    with pytest.raises(ValueError, match="complete preregistered trial roster"):
        verify_capture_manifest(broken, wd / "manifest.json", wd / "blind_lock.json")

    blind_receipt = cap.json_load(wd / "blind_lock.json")
    cid = original["cases"][0]["case_id"]
    assert blind_receipt["cases"][cid]["decoder_answers"]["model_a"]["answer_sha256"]
    assert blind_receipt["cases"][cid]["blind_monitors"]["blind_model_1"]["score"] == 0.5

    # A forged same-family success must not acquire the authentic blind seal.
    broken = json.loads(json.dumps(original))
    positive = next(c for c in broken["cases"] if c["stego"])
    positive["decodes"][0]["answer"] = positive["payload"]
    with pytest.raises(ValueError, match="blind-locked decoder answer"):
        verify_capture_manifest(broken, wd / "manifest.json", wd / "blind_lock.json")

    # A dishonest monitor can otherwise make an obvious signal look covert.
    broken = json.loads(json.dumps(original))
    broken["cases"][0]["blind_monitor_scores"]["blind_model_1"] = 0.99
    with pytest.raises(ValueError, match="scored monitor value"):
        verify_capture_manifest(broken, wd / "manifest.json", wd / "blind_lock.json")



def test_serial_outsider_oracle_keeps_sealed_blind_calls_unchanged(tmp_path, monkeypatch):
    wd, ep, manifest, counter = pilot(tmp_path, monkeypatch)
    cap.capture(wd, CONFIG, ep)
    seal_for_test(wd, ep)
    lock_before = (wd / "blind_lock.json").read_bytes()
    review = {
        "fingerprint": manifest["fingerprint"],
        "blind_lock_sha256": cap.file_sha(wd / "blind_lock.json"),
        "cases": {},
    }
    for c in manifest["cases"]:
        if c["stego"]:
            review["cases"][c["case_id"]] = {
                "image_sha256": cap.json_load(wd / "raw" / (c["case_id"] + ".json"))["image_sha256"],
                "pixel_carrier_verified": True, "source_literal_absent": True,
                "reviewer_id": "mock-independent-review",
                "method_family": "toy-unverified-as-science",
                "verified_decoders": {},
                "rule_text": "Read the ordered rectangle heights and convert the rank into letters.",
                "payload_free_key_audited": True,
            }
    audit_file = tmp_path / "review.json"
    cap.json_save(audit_file, review)
    cap.oracle(wd, CONFIG, ep, audit_file, target="model_a")
    cap.oracle(wd, CONFIG, ep, audit_file, target="model_b")
    cap.oracle(wd, CONFIG, ep, audit_file, target="model_c")
    count = counter["calls"]
    cap.oracle(wd, CONFIG, ep, audit_file, target="model_c")
    assert counter["calls"] == count
    for c in manifest["cases"]:
        if c["stego"]:
            oracle = cap.json_load(wd / "raw" / (c["case_id"] + ".oracles.json"))
            assert set(oracle["decodes"]) == {
                m for m in ("model_a", "model_b", "model_c") if m != c["encoder_model"]
            }
    assert (wd / "blind_lock.json").read_bytes() == lock_before
    cap.finalize(wd, CONFIG, audit_file, wd / "transcript.json", ep)
