"""Run preregistered, matched-budget A/B/C/D experiments on shared splits.

Run from the project directory that contains main.py. Python 3.10+.
No real LLM is bundled; Ollama must be running separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

DATASETS = {
    "cbis": ("data/data-cbis-ddsm.csv", "Classify benign and malignant breast lesions using mammographic metadata, feature selection, ensemble learning, and tabular machine learning", "examples/frozen_cbis_example"),
    "wdbc": ("data/data-3.csv", "Predict malignant versus benign tumors using WDBC measurements", "examples/frozen_wdbc_example"),
}
ARMS = {
    "A": "A_no_rag.yaml",
    "B": "B_rag_context_only.yaml",
    "C": "C_rag_evidence_compiler.yaml",
    "D": "D_rag_llm_grounded.yaml",
}
FROZEN_FILES = ("research.json", "evidence.json")


def digest_files(directory: Path) -> str:
    """Hash canonical parsed JSON, preserving content but ignoring whitespace changes."""
    h = hashlib.sha256()
    for name in FROZEN_FILES:
        p = directory / name
        value = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Expected JSON list in {p}")
        h.update(name.encode())
        h.update(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
    return h.hexdigest()


def configure_arm(original: dict, *, arm: str, seed: int, budget: int,
                  max_iterations: int, frozen: Path, frozen_sha: str,
                  suite_id: str) -> dict:
    cfg = json.loads(json.dumps(original))
    loop = cfg.setdefault("loop", {})
    loop.update({
        "target_evaluated_candidates": budget,
        "candidates_per_iteration": 3,
        "max_iterations": max_iterations,
        "patience": max_iterations + 1,
        "max_failures": max_iterations + 1,
        "max_batch_rounds": 5,
    })
    cfg["seed"] = int(seed)
    cfg.setdefault("agents", {}).setdefault("critic", {})["enabled"] = False
    rag = cfg.setdefault("rag", {})
    rag["replay_from_run"] = str(frozen.resolve()) if arm != "A" else None
    rag.setdefault("knowledge_base", {})["refresh_mode"] = "offline"
    cfg["controlled_study"] = {
        "suite_id": suite_id,
        "frozen_corpus_sha256": frozen_sha,
        "save_baseline_predictions": True,
        "experiment_arm": arm,
    }
    return cfg


def prepare_suite(suite: Path, *, dataset: str, csv: Path, task: str,
                  frozen_source: Path, seeds: list[int], budget: int,
                  max_iterations: int) -> dict:
    suite.mkdir(parents=True, exist_ok=True)
    frozen = suite / "frozen_corpus"
    frozen.mkdir(exist_ok=True)
    for name in FROZEN_FILES:
        dest = frozen / name
        if not dest.exists():
            shutil.copy2(frozen_source / name, dest)
    corpus_sha = digest_files(frozen)
    if not json.loads((frozen / "research.json").read_text(encoding="utf-8")):
        raise ValueError("Frozen research list is empty")
    if not json.loads((frozen / "evidence.json").read_text(encoding="utf-8")):
        raise ValueError("Frozen evidence list is empty")
    csv_sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "suite_id": suite.name,
        "dataset": dataset,
        "csv": str(csv.resolve()),
        "dataset_sha256": csv_sha,
        "task": task,
        "seeds": list(seeds),
        "target_evaluated_candidates": int(budget),
        "max_iterations": int(max_iterations),
        "arms": list(ARMS),
        "frozen_corpus": str(frozen.resolve()),
        "frozen_corpus_sha256": corpus_sha,
        "holdout_policy": "one stratified 80/20 split per seed; identical A/B/C/D row indices",
        "candidate_policy": "identical number of successfully CV-evaluated candidates, not just slots",
        "inference_policy": "paired within each seed; repeated splits overlap, so cross-seed intervals are descriptive",
    }
    path = suite / "suite_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        # Fail-closed for frozen evidence, dataset versions and assignment parameters.
        for k in ("dataset_sha256", "frozen_corpus_sha256", "task", "seeds", "target_evaluated_candidates", "max_iterations", "arms"):
            if previous.get(k) != manifest.get(k):
                raise ValueError(f"Existing suite differs in {k}; choose another --suite-id")
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_suite(*, dataset: str, seeds: list[int], budget: int,
              max_iterations: int, suite_id: str | None = None,
              csv_path: str | None = None, task: str | None = None,
              frozen_path: str | None = None, dry_run: bool = False,
              force: bool = False) -> Path:
    root = Path.cwd()
    if not (root / "main.py").exists():
        raise FileNotFoundError("Run from the project directory containing main.py")
    default_csv, default_task, default_frozen = DATASETS[dataset]
    csv = Path(csv_path or default_csv).resolve()
    if not csv.is_file():
        raise FileNotFoundError(csv)
    description = task or default_task
    src = Path(frozen_path or default_frozen).resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    if budget < 1 or max_iterations < 1 or budget > 3 * max_iterations:
        raise ValueError("Budget must be 1..3*max_iterations")
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be a nonempty list of distinct integers")
    if suite_id is None:
        suite_id = dataset + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", suite_id):
        raise ValueError("suite-id may only contain letters, numbers, underscores and hyphens")
    suite = root / "experiments" / "controlled" / suite_id
    manifest = prepare_suite(suite, dataset=dataset, csv=csv, task=description,
                             frozen_source=src, seeds=seeds, budget=budget,
                             max_iterations=max_iterations)
    results = suite / "reports"
    results.mkdir(parents=True, exist_ok=True)
    configs = suite / "configs"
    print(f"[suite] {suite}")
    print(f"[suite] dataset SHA256={manifest['dataset_sha256']}")
    print(f"[suite] frozen corpus SHA256={manifest['frozen_corpus_sha256']}")
    for seed in seeds:
        for arm, filename in ARMS.items():
            cfg = yaml.safe_load((root / "configs" / filename).read_text(encoding="utf-8"))
            snapshot = configure_arm(cfg, arm=arm, seed=seed, budget=budget,
                                     max_iterations=max_iterations,
                                     frozen=suite / "frozen_corpus",
                                     frozen_sha=manifest["frozen_corpus_sha256"],
                                     suite_id=suite_id)
            config_path = configs / str(seed) / filename
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
            target_report = results / f"seed_{seed}_{arm}.json"
            if target_report.exists() and not force and \
                    (suite / "predictions" / f"seed_{seed}_{arm}_selected.csv").exists():
                previous = json.loads(target_report.read_text(encoding="utf-8"))
                print(f"[skip] seed={seed} arm={arm} existing run={previous.get('run_id')}")
                continue
            command = [sys.executable, str(root / "main.py"),
                       "--csv", str(csv), "--target", "diagnosis", "--task", description,
                       "--config", str(config_path)]
            print(f"[run] seed={seed} arm={arm} {command}", flush=True)
            if dry_run:
                continue
            logfile = suite / "logs" / f"seed_{seed}_{arm}.log"
            logfile.parent.mkdir(exist_ok=True)
            with logfile.open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
            output = logfile.read_text(encoding="utf-8", errors="replace")
            ids = re.findall(r"(?m)^Run (\d{8}_\d{6}_\d{6})\s*$", output)
            if proc.returncode != 0 or not ids:
                print(f"[ERROR] seed={seed} arm={arm} exit={proc.returncode}; see {logfile}")
                continue
            run_dir = root / "experiments" / "runs" / ids[-1]
            original = run_dir / "final_report.json"
            if not original.exists():
                print(f"[ERROR] missing {original}; see {logfile}")
                continue
            report = json.loads(original.read_text(encoding="utf-8"))
            # Each run should obey the contract before its report is archived.
            if report.get("dataset_sha256") != manifest["dataset_sha256"] or \
               report.get("controlled_study", {}).get("frozen_corpus_sha256") != manifest["frozen_corpus_sha256"] or \
               report.get("split_sha256") is None:
                print(f"[ERROR] contract mismatch: seed={seed} arm={arm}; see {logfile}")
                continue
            shutil.copy2(original, target_report)
            predictions_dir = suite / "predictions"
            predictions_dir.mkdir(exist_ok=True)
            for source_name, prefix in (("holdout_predictions.csv", "selected"),
                                        ("baseline_holdout_predictions.csv", "baseline")):
                source_file = run_dir / source_name
                if source_file.is_file():
                    shutil.copy2(source_file, predictions_dir / f"seed_{seed}_{arm}_{prefix}.csv")
            print(f"[saved] {target_report} budget_complete={report['controlled_study']['budget_complete']}", flush=True)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=9)
    parser.add_argument("--suite-id", default=None)
    parser.add_argument("--csv", default=None, help="Alternative CSV path")
    parser.add_argument("--task", default=None)
    parser.add_argument("--frozen", default=None, help="Frozen literature directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun and overwrite the suite's per-arm alias (original run archives remain)")
    args = parser.parse_args()
    path = run_suite(dataset=args.dataset, seeds=args.seeds, budget=args.budget,
                     max_iterations=args.max_iterations, suite_id=args.suite_id,
                     csv_path=args.csv, task=args.task, frozen_path=args.frozen,
                     dry_run=args.dry_run, force=args.force)
    print(f"[done] suite directory: {path}")
    print(f"Analyze with: python -m scripts.analyze_controlled_abcd --suite {path}")

if __name__ == "__main__":
    main()
