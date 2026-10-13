"""Shared artifact and witness-receipt plumbing for the stego suite runners.

The three campaign runners each own their row and summary *schema* (matching the
RTG convention), but they share how a campaign is read in and written out: the
schema/description preamble, canonical JSON, the CSV dialect, and the
reproducibility receipt that hashes every artifact and asserts ``model_call``
and ``network`` are both false.  Keeping that here means the three witness
receipts are identically shaped and a change to the receipt format happens once.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

__all__ = [
    "Config",
    "load_config",
    "require",
    "sha256_file",
    "repo_relative",
    "write_rows",
    "write_summary",
    "write_receipt",
]

Config = dict[str, object]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> Config:
    """Read a campaign config, stripping the schema/description preamble."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw.pop("schema_version", None)
    raw.pop("description", None)
    return raw


def require(config: Config, key: str) -> object:
    if key not in config:
        raise ValueError(f"config missing required key {key!r}")
    return config[key]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    """Posix path relative to the repo root, or the bare name if outside it.

    Keeps committed receipts free of machine-specific absolute paths.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as UTF-8 CSV with the header taken from the first row."""

    if not rows:
        raise ValueError("cannot write an empty rows table")
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    """Write a summary as canonical, newline-terminated JSON."""

    Path(path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_receipt(
    output_dir: Path,
    config_path: Path,
    experiment_prefix: str,
    artifacts: dict[str, Path],
    *,
    safety_boundary: str,
    extra: dict[str, object] | None = None,
    schema_version: str = "1.0",
) -> dict[str, object]:
    """Write ``witness_receipt.json`` hashing every artifact; return the receipt.

    ``model_call`` and ``network`` are always false: this package never does
    either, and a receipt that claimed otherwise would be a bug worth catching.
    """

    config_bytes = Path(config_path).read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    art_hashes = {name: sha256_file(p) for name, p in sorted(artifacts.items())}
    combined = hashlib.sha256(
        "".join(art_hashes[name] for name in sorted(art_hashes)).encode("utf-8")
    ).hexdigest()
    receipt: dict[str, object] = {
        "schema_version": schema_version,
        "experiment_id": f"{experiment_prefix}-{config_hash[:12]}",
        "configuration_file": repo_relative(config_path),
        "configuration_hash": config_hash,
        "artifacts": art_hashes,
        "combined_sha256": combined,
        "model_call": False,
        "network": False,
        "safety_boundary": safety_boundary,
    }
    if extra:
        receipt.update(extra)
    write_summary(output_dir / "witness_receipt.json", receipt)
    return receipt
