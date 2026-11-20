"""Paired, row-aligned analysis of a controlled A/B/C/D suite.

Uncertainty: stratified paired patient bootstrap *within each seed*, conditional
on its selected models/split. Overlapping holdouts across different seeds mean
cross-seed variation is descriptive, NOT an independent-sample confidence interval.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ARMS = ("A", "B", "C", "D")


def read_predictions(path: Path, report: dict) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"true_label": str, "predicted_label": str})
    mandatory = {"row_index", "true_label", "predicted_label", "split_sha256", "dataset_sha256", "seed"}
    if not mandatory.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {mandatory-set(df.columns)}")
    if df.row_index.duplicated().any():
        raise ValueError(f"Repeated holdout row index in {path}")
    if len(df) != int(report["holdout"]["n_test"]):
        raise ValueError(f"Incorrect holdout size in {path}")
    for col in ("split_sha256", "dataset_sha256"):
        if set(df[col].astype(str)) != {str(report[col])}:
            raise ValueError(f"Wrong {col} in {path}")
    if df["seed"].nunique() != 1:
        raise ValueError(f"Multiple seeds in {path}")
    return df.sort_values("row_index").reset_index(drop=True)


def metric_scores(df: pd.DataFrame, indices=None) -> dict:
    if indices is None:
        indices = np.arange(len(df))
    y = df.true_label.to_numpy()[indices]
    yp = df.predicted_label.to_numpy()[indices]
    result = {
        "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, yp)),
    }
    # Positive label is M (malignant) for WDBC/CBIS; handle no-probability models.
    if "probability_M" in df and len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score((y == "M").astype(int),
                                                 df["probability_M"].to_numpy()[indices]))
    return result


def compare_pair(left: pd.DataFrame, right: pd.DataFrame, *, n_bootstrap=2000,
                 seed=42) -> list[dict]:
    """Paired stratified bootstrap; each resample uses identical patient rows."""
    if not np.array_equal(left.row_index.to_numpy(), right.row_index.to_numpy()) or \
       not np.array_equal(left.true_label.to_numpy(), right.true_label.to_numpy()):
        raise ValueError("Cannot compare: holdout row IDs or true labels differ")
    left_scores = metric_scores(left)
    right_scores = metric_scores(right)
    keys = sorted(set(left_scores) & set(right_scores))
    arr = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    class_indices = [np.where(left.true_label.to_numpy() == c)[0]
                     for c in sorted(left.true_label.unique())]
    for _ in range(n_bootstrap):
        idx = np.concatenate([rng.choice(c_idx, size=len(c_idx), replace=True)
                              for c_idx in class_indices])
        a, b = metric_scores(left, idx), metric_scores(right, idx)
        for key in keys:
            arr[key].append(b[key] - a[key])
    return [{
        "metric": metric, "left_score": left_scores[metric],
        "right_score": right_scores[metric],
        "delta_right_minus_left": right_scores[metric] - left_scores[metric],
        "bootstrap_95_lower": float(np.percentile(arr[metric], 2.5)),
        "bootstrap_95_upper": float(np.percentile(arr[metric], 97.5)),
        "n_holdout_rows": len(left), "bootstrap_replicates": n_bootstrap,
    } for metric in keys]


def analyze_suite(suite: Path, *, n_bootstrap: int = 2000) -> dict:
    suite = Path(suite).resolve()
    manifest = json.loads((suite / "suite_manifest.json").read_text(encoding="utf-8"))
    analyses_dir = suite / "analysis"
    analyses_dir.mkdir(exist_ok=True)
    per_run, per_pair, audit = [], [], []
    for seed in manifest["seeds"]:
        valid = {}
        for arm in ARMS:
            report_path = suite / "reports" / f"seed_{seed}_{arm}.json"
            pred_path = suite / "predictions" / f"seed_{seed}_{arm}_selected.csv"
            status = {"seed": seed, "arm": arm, "usable": False, "reason": ""}
            if not report_path.exists():
                status["reason"] = "missing_report"
            else:
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    control = report.get("controlled_study", {})
                    if not control.get("budget_complete", False) or control.get("successfully_evaluated_candidates") != manifest["target_evaluated_candidates"]:
                        raise ValueError("unfilled_or_mismatched_budget")
                    if arm == "D" and report.get("strict_D_completed") is not True:
                        raise ValueError("strict_D_incomplete")
                    if report.get("dataset_sha256") != manifest["dataset_sha256"]:
                        raise ValueError("dataset_digest_mismatch")
                    if control.get("frozen_corpus_sha256") != manifest["frozen_corpus_sha256"]:
                        raise ValueError("frozen_corpus_mismatch")
                    if arm != "A" and report.get("research_source") != "frozen_replay":
                        raise ValueError("not_frozen_replay")
                    if report.get("description") != manifest["task"]:
                        raise ValueError("task_mismatch")
                    predictions = read_predictions(pred_path, report)
                    if not (predictions.seed.astype(int) == int(seed)).all():
                        raise ValueError("prediction_seed_mismatch")
                    valid[arm] = (report, predictions)
                    status["usable"] = True
                    status["reason"] = "ok"
                    scores = metric_scores(predictions)
                    per_run.append({"seed": seed, "arm": arm, "run_id": report["run_id"],
                                    "model": report["best"]["config"].get("composition", {}).get("kind") if report["best"]["config"].get("composition") else report["best"]["config"].get("model", {}).get("name"),
                                    "cv_mean": report["best"]["cv_mean"],
                                    "holdout_macro_f1": scores["macro_f1"],
                                    "holdout_accuracy": scores["accuracy"],
                                    "holdout_roc_auc": scores.get("roc_auc"),
                                    "candidates_evaluated": control["successfully_evaluated_candidates"],
                                    "run_wall_time_seconds": report.get("run_wall_time_seconds_excluding_pdf"),
                                    "split_sha256": report["split_sha256"]})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    status["reason"] = f"invalid:{exc}"
            audit.append(status)
        if len(valid) == 4:
            splits = {r["split_sha256"] for r, _ in valid.values()}
            if len(splits) != 1:
                audit.append({"seed": seed, "arm": "ALL", "usable": False,
                              "reason": "different_data_splits"})
                continue
            # Even with the same snapshot path, verify that B/C/D actually
            # consumed identical evidence payloads, when their digests exist.
            evidence_hashes = {valid[arm][0].get("research_evidence_sha256")
                               for arm in ("B", "C", "D")}
            if len(evidence_hashes) != 1:
                audit.append({"seed": seed, "arm": "ALL", "usable": False,
                              "reason": "different_evidence_payloads"})
                continue
            for a, b in itertools.combinations(ARMS, 2):
                pair_seed = int(seed) + int(hashlib.sha256(f"{a}:{b}".encode()).hexdigest()[:6], 16)
                try:
                    values = compare_pair(valid[a][1], valid[b][1],
                                          n_bootstrap=n_bootstrap, seed=pair_seed)
                    for v in values:
                        per_pair.append({"seed": seed, "left_arm": a,
                                         "right_arm": b, **v})
                except ValueError as exc:
                    audit.append({"seed": seed, "arm": f"{a}-{b}", "usable": False,
                                  "reason": f"pairwise_alignment_error:{exc}"})
        else:
            audit.append({"seed": seed, "arm": "ALL", "usable": False,
                          "reason": "one_or_more_arms_incomplete"})
    # Summaries over matched seeds; overlap among different holdouts means no
    # assumption of independent cross-seed observations is made.
    pair_df = pd.DataFrame(per_pair)
    summaries = []
    if not pair_df.empty:
        for (a, b, metric), group in pair_df.groupby(["left_arm", "right_arm", "metric"]):
            differences = group["delta_right_minus_left"].to_numpy(dtype=float)
            summaries.append({"left_arm": a, "right_arm": b, "metric": metric,
                              "n_matched_seeds": len(differences),
                              "mean_paired_difference": float(differences.mean()),
                              "seed_to_seed_sd_descriptive": float(differences.std(ddof=1)) if len(differences)>1 else None,
                              "min_paired_difference": float(differences.min()),
                              "max_paired_difference": float(differences.max()),
                              "n_seeds_positive": int((differences>0).sum())})
    pd.DataFrame(per_run).to_csv(analyses_dir / "per_run_metrics.csv", index=False)
    pd.DataFrame(per_pair).to_csv(analyses_dir / "paired_per_seed_bootstrap.csv", index=False)
    pd.DataFrame(summaries).to_csv(analyses_dir / "paired_across_seeds_descriptive.csv", index=False)
    pd.DataFrame(audit).to_csv(analyses_dir / "quality_audit.csv", index=False)
    outcome = {
        "suite": str(suite),
        "total_seeds": len(manifest["seeds"]),
        "fully_matched_seeds": len(set(pair_df.seed)) if not pair_df.empty else 0,
        "n_valid_pairs": len(per_pair),
        "interpretation": "Within-seed bootstrap intervals are conditional on selected models and split; overlapping holdouts across seeds imply cross-seed summaries are descriptive, not confirmatory independent-replicate CIs. All holdout uses must stay out of model selection.",
    }
    (analyses_dir / "analysis_summary.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    print(json.dumps(analyze_suite(Path(args.suite), n_bootstrap=args.bootstrap), indent=2))

if __name__ == "__main__":
    main()
