"""Acceptance and reproducibility tests for the trained-channel campaign."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stegobench.channels import (
    ChannelSpec,
    ScriptedChannelBank,
    channel_fingerprint,
    load_channel_samples,
)
from stegobench.channel_eval import load_config, run_campaign

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "stego_channel.json"


@pytest.fixture(scope="module")
def summary(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("stego_channel")
    return run_campaign(out, CONFIG)


def test_all_scripted_acceptance_gates_pass(summary: dict) -> None:
    assert summary["all_acceptance_passed"] is True
    assert all(summary["scripted_acceptance"].values()), summary["scripted_acceptance"]


def test_trained_channel_dominates_overt(summary: dict) -> None:
    overt = summary["frontier"]["overt"]
    matched = summary["frontier"]["matched"]
    # Same top capacity, but matched hides at chance while overt is fully caught.
    assert matched[-1]["capacity_per_token"] == pytest.approx(overt[-1]["capacity_per_token"], abs=0.02)
    assert matched[-1]["best_detect_auroc"] <= 0.6
    assert overt[-1]["best_detect_auroc"] >= 0.9


def test_matched_covert_across_all_rates(summary: dict) -> None:
    for pt in summary["frontier"]["matched"]:
        assert pt["best_detect_auroc"] <= 0.6


def test_noise_discounts_capacity(summary: dict) -> None:
    matched = {pt["rate"]: pt["capacity_per_token"] for pt in summary["frontier"]["matched"]}
    noisy = {pt["rate"]: pt["capacity_per_token"] for pt in summary["frontier"]["noisy_matched"]}
    top = max(matched)
    # 15% per-bit noise discounts chance-corrected capacity to ~0.7x.
    assert noisy[top] < matched[top]
    assert 0.6 <= noisy[top] / matched[top] <= 0.8


def test_paraphrase_destroys_overt_preserves_matched(summary: dict) -> None:
    overt = summary["frontier"]["overt"]
    matched = summary["frontier"]["matched"]
    assert overt[-1]["survival_strip_overt"] <= 0.35
    for pt in matched:
        assert pt["survival_strip_overt"] >= 0.9


def test_artifacts_match_witness_receipt(tmp_path: Path) -> None:
    run_campaign(tmp_path, CONFIG)
    receipt = json.loads((tmp_path / "witness_receipt.json").read_text(encoding="utf-8"))
    for name in ("summary.json", "rows.csv"):
        got = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert got == receipt["artifacts"][name], name
    assert (tmp_path / "figures" / "frontier.png").stat().st_size > 0
    assert receipt["model_call"] is False and receipt["network"] is False


def test_byte_identical_on_rerun(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    run_campaign(a, CONFIG)
    run_campaign(b, CONFIG)
    for name in ("summary.json", "rows.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_determinism_across_hash_seeds(tmp_path: Path) -> None:
    def once(seed: str) -> bytes:
        out = tmp_path / f"seed{seed}"
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO / "src"))
        subprocess.run(
            [sys.executable, "-m", "stegobench.channel_eval", "--output", str(out)],
            check=True,
            env=env,
            cwd=str(REPO),
            capture_output=True,
        )
        return (out / "summary.json").read_bytes()

    assert once("0") == once("17") == once("4242")


def test_sample_file_scored_matches_scripted_default(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    slots = int(config["slots"])
    rates = tuple(float(x) for x in config["rates"])
    families = dict(config["families"])
    paraphrase_names = tuple(str(x) for x in config["paraphrases"])
    doc_ids = tuple(f"doc-{i:04d}" for i in range(int(config["doc_count"])))
    specs = tuple(
        ChannelSpec(str(fam), r, float(noise)) for fam, noise in families.items() for r in rates
    )
    fp = channel_fingerprint(doc_ids, specs, slots, paraphrase_names)
    samples = ScriptedChannelBank().run(doc_ids, specs, slots, paraphrase_names, str(config["seed"]))
    sfile = tmp_path / "samples.json"
    sfile.write_text(
        json.dumps({"fingerprint": fp, "records": [s.to_json() for s in samples]}), encoding="utf-8"
    )

    default = run_campaign(tmp_path / "default", CONFIG)
    from_file = run_campaign(tmp_path / "fromfile", CONFIG, samples_file=sfile)
    assert from_file["frontier"] == default["frontier"]


def test_fingerprint_guard_rejects_foreign_samples(tmp_path: Path) -> None:
    sfile = tmp_path / "bad.json"
    sfile.write_text(json.dumps({"fingerprint": "deadbeef", "records": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_channel_samples(sfile, "something-else")
