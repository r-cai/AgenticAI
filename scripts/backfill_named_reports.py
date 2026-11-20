"""Create dataset/experiment-named copies of reports from existing completed runs.

Run from project root: python -m scripts.backfill_named_reports
Existing final_report.json files are never modified by this utility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestration.report_naming import (
    experiment_from_snapshot, save_report_copies,
)


def backfill(runs_root: str | Path = "experiments/runs",
             reports_root: str | Path = "experiments/reports") -> dict:
    runs_root = Path(runs_root)
    reports_root = Path(reports_root)
    summary = {"copied": 0, "existing_archives": 0, "skipped": 0, "errors": []}
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {runs_root.resolve()}")
    # Run IDs sort chronologically. The most recent run for each dataset/arm
    # will determine its latest alias.
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        report_file = run_dir / "final_report.json"
        snapshot_file = run_dir / "config_snapshot.json"
        if not report_file.is_file() or not snapshot_file.is_file():
            summary["skipped"] += 1
            continue
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
            cfg = json.loads(snapshot_file.read_text(encoding="utf-8"))
            dataset = report.get("dataset")
            run_id = report.get("run_id")
            if not dataset or not run_id:
                summary["skipped"] += 1
                continue
            arm = report.get("experiment_id") or experiment_from_snapshot(cfg)
            # Compatibility: save_report_copies infers the arm from the config name.
            # Only copies are written; original JSON remains byte-for-byte identical.
            report = dict(report)
            archive = reports_root / f"final_report_{Path(dataset).stem}_{arm}_{run_id}.json"
            if archive.exists():
                summary["existing_archives"] += 1
                # Latest copy will be populated by the original run's end-of-run code.
                continue
            paths = save_report_copies(report, csv_path=dataset,
                 config_path=f"configs/{arm}_recovered.yaml", run_dir=run_dir,
                 reports_root=reports_root, rewrite_original=False)
            print(f"[backfill] {paths['latest_copy']} (run {run_id})")
            summary["copied"] += 1
        except Exception as exc:
            summary["errors"].append(f"{run_dir.name}: {type(exc).__name__}: {exc}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--reports-dir", default="experiments/reports")
    args = parser.parse_args()
    result = backfill(args.runs_dir, args.reports_dir)
    print(f"Copied {result['copied']} run(s); "
          f"{result['existing_archives']} already archived; "
          f"{result['skipped']} skipped; {len(result['errors'])} error(s)")
    for error in result['errors']:
        print("[warning]", error)


if __name__ == "__main__":
    main()
