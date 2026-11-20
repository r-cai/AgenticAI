"""Deterministic, safe report naming for multi-dataset A/B/C/D experiments.

No modeling or holdout behavior is changed by this module.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


def safe_stem(value: str, *, max_chars: int = 80) -> str:
    """Return a filesystem-safe identifier from a dataset or config basename."""
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_-.")
    if not value:
        raise ValueError("Cannot derive a safe report identifier from an empty name")
    return value[:max_chars]


def experiment_label(config_path: str | Path) -> str:
    """Identify the actual A/B/C/D configuration by filename, without guessing D.

    Legacy D_no_rag_level2 is a no-RAG control, not evidence-grounded D.
    """
    stem = Path(config_path).stem
    if stem.casefold() == "d_no_rag_level2":
        return "NoRAG_Level2"
    match = re.match(r"^([ABCD])_", stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else safe_stem(stem)


def _atomic_write_json(path: Path, report: dict) -> None:
    """Write a complete report atomically, safe against truncated latest copies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".report-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def save_report_copies(report: dict, *, csv_path: str | Path,
                       config_path: str | Path, run_dir: str | Path,
                       reports_root: str | Path = "experiments/reports",
                       rewrite_original: bool = True) -> dict:
    """Save original + named run copy + unique archive + most-recent alias.

    The per-run original and unique archive are never intentionally replaced;
    only the named latest alias is overwritten on a rerun of the same arm.
    """
    run_dir = Path(run_dir)
    reports_root = Path(reports_root)
    dataset = safe_stem(Path(csv_path).stem)
    experiment = experiment_label(config_path)
    run_id = safe_stem(str(report["run_id"]), max_chars=90)
    basename = f"final_report_{dataset}_{experiment}"
    original = run_dir / "final_report.json"
    named_in_run = run_dir / f"{basename}.json"
    archive = reports_root / f"{basename}_{run_id}.json"
    latest = reports_root / f"{basename}.json"
    if archive.exists():
        raise FileExistsError(f"Report archive already exists: {archive}")
    report.update({
        "dataset_id": dataset,
        "experiment_id": experiment,
        "report_paths": {
            "original": str(original),
            "named_run_copy": str(named_in_run),
            "archived_copy": str(archive),
            "latest_copy": str(latest),
        },
    })
    destinations = (original, named_in_run, archive, latest) if rewrite_original else (named_in_run, archive, latest)
    for path in destinations:
        _atomic_write_json(path, report)
    return report["report_paths"]


def experiment_from_snapshot(snapshot: dict) -> str:
    """Reconstruct an experiment arm from an older run's saved config.

    A/no-RAG, B/context-RAG, C/template-compiler, D/strict LLM-grounded.
    """
    if not isinstance(snapshot, dict):
        raise TypeError("config snapshot must be a dictionary")
    if snapshot.get("llm_evidence_design", {}).get("enabled"):
        return "D"
    if snapshot.get("evidence_composition", {}).get("enabled"):
        return "C"
    if snapshot.get("rag", {}).get("enabled"):
        return "B"
    return "A"
