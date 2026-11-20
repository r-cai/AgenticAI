"""Offline regression tests for the controlled research protocol."""
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification

from agents.schemas import CandidateConfig, GeneratorOutput, ModelConfig, PreprocessingConfig
from orchestration.controller import run_experiment
from scripts.analyze_controlled_abcd import analyze_suite, compare_pair
from scripts.run_controlled_abcd import configure_arm, prepare_suite, digest_files


@contextmanager
def cd(p):
    old = Path.cwd()
    try:
        os.chdir(p)
        yield
    finally:
        os.chdir(old)


def candidate(name):
    return CandidateConfig(rationale="controlled comparison proposal", preprocessing=PreprocessingConfig(),
                           model=ModelConfig(name=name))


class ControlledProtocolTest(unittest.TestCase):
    def test_budget_caps_valid_evaluations_and_writes_row_aligned_predictions(self):
        x, y = make_classification(n_samples=90, n_features=8, n_informative=5,
                                   n_redundant=0, random_state=11)
        frame = pd.DataFrame(x, columns=[f"feat_{i}" for i in range(8)])
        frame["diagnosis"] = np.where(y, "M", "B")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "dataset.csv"
            frame.to_csv(csv, index=False)
            frozen = root / "frozen"
            frozen.mkdir()
            (frozen / "research.json").write_text(json.dumps([{"title":"test paper","abstract":"classification"}]), encoding="utf-8")
            (frozen / "evidence.json").write_text("[]", encoding="utf-8")
            cfg = {
                "seed": 42, "test_size": 0.2,
                "cv": {"folds": 2, "repeats": 1, "scoring": "f1_macro", "n_jobs": 1},
                "loop": {"max_iterations": 3, "candidates_per_iteration": 3,
                         "target_evaluated_candidates": 2, "patience": 9,
                         "max_failures": 9, "max_batch_rounds": 1},
                "agents": {"generator": {"model": "mock", "temperature": 0},
                           "critic": {"enabled": False}},
                "rag": {"enabled": False, "knowledge_base": {"enabled": False}},
                "evidence": {"enabled": False},
                "evidence_composition": {"enabled": False},
                "controlled_study": {"suite_id": "mock", "save_baseline_predictions": True},
            }
            generator = lambda **kw: GeneratorOutput(reasoning="test", candidates=[
                candidate("logistic_regression"), candidate("gradient_boosting"), candidate("extra_trees")])
            with cd(root), patch("orchestration.controller.generate", side_effect=generator), \
                         patch("orchestration.reporter.generate_pdf", return_value=root/"mock.pdf"):
                (root/"A_no_rag.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
                a = run_experiment(str(csv), "diagnosis", "test", "A_no_rag.yaml")
                cfg["rag"] = {"enabled": True, "replay_from_run": str(frozen), "knowledge_base": {"enabled": False}}
                cfg["evidence"]["enabled"] = True
                (root/"B_rag_context_only.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
                b = run_experiment(str(csv), "diagnosis", "test", "B_rag_context_only.yaml")
                pa = pd.read_csv(Path(a["predictions_file"]))
                pb = pd.read_csv(Path(b["predictions_file"]))
                self.assertTrue(a["controlled_study"]["budget_complete"])
                self.assertEqual(a["controlled_study"]["successfully_evaluated_candidates"], 2)
                self.assertEqual(len([h for h in a["history"] if h["iteration"] > 0]), 2)
                self.assertEqual(a["stop_reason"], "target_evaluated_candidates_reached")
                self.assertIsNone(a["strict_D_completed"])
                self.assertEqual(a["split_sha256"], b["split_sha256"])
                self.assertEqual(a["dataset_sha256"], b["dataset_sha256"])
                self.assertEqual(pa.row_index.tolist(), pb.row_index.tolist())
                self.assertEqual(pa.true_label.tolist(), pb.true_label.tolist())
                self.assertIn("probability_M", pa.columns)
                self.assertTrue(Path(a["output_dir"], "baseline_holdout_predictions.csv").exists())
                self.assertEqual(len(pa), 18)
                self.assertEqual(len(compare_pair(pa, pb, n_bootstrap=100)), 3)

    def test_frozen_suite_manifest_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frozen = base / "frozen"
            frozen.mkdir()
            for name in ("research.json", "evidence.json"):
                (frozen / name).write_text(json.dumps([{"test":name}]), encoding="utf-8")
            csv = base / "data.csv"
            csv.write_text("a,diagnosis\n1,B\n2,M\n", encoding="utf-8")
            suite = base/"study"
            manifest = prepare_suite(suite, dataset="cbis", csv=csv, task="test", frozen_source=frozen,
                                     seeds=[42, 52], budget=6, max_iterations=9)
            self.assertEqual(manifest["frozen_corpus_sha256"], digest_files(suite/"frozen_corpus"))
            with self.assertRaisesRegex(ValueError, "target_evaluated_candidates"):
                prepare_suite(suite, dataset="cbis", csv=csv, task="test", frozen_source=frozen,
                              seeds=[42, 52], budget=7, max_iterations=9)
            (suite/"frozen_corpus"/"evidence.json").write_text(json.dumps([{"changed":True}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen_corpus_sha256"):
                prepare_suite(suite, dataset="cbis", csv=csv, task="test", frozen_source=frozen,
                              seeds=[42, 52], budget=6, max_iterations=9)

    def test_configure_arm_preserves_A_no_rag_and_BCD_shared_frozen(self):
        cfg = {"rag":{"enabled":True},"loop":{"max_iterations":3},"agents":{}}
        with tempfile.TemporaryDirectory() as tmp:
            a = configure_arm(cfg, arm="A", seed=52, budget=6, max_iterations=9,
                              frozen=Path(tmp), frozen_sha="digest", suite_id="pilot")
            b = configure_arm(cfg, arm="B", seed=52, budget=6, max_iterations=9,
                              frozen=Path(tmp), frozen_sha="digest", suite_id="pilot")
            self.assertIsNone(a["rag"]["replay_from_run"])
            self.assertEqual(b["rag"]["replay_from_run"], str(Path(tmp).resolve()))
            self.assertEqual(b["loop"]["target_evaluated_candidates"], 6)
            self.assertEqual(b["seed"], 52)
            self.assertEqual(b["agents"]["critic"]["enabled"], False)

    def test_analysis_excludes_missing_arms_and_unfilled_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            (suite/"reports").mkdir()
            (suite/"predictions").mkdir()
            (suite/"suite_manifest.json").write_text(json.dumps({
                "seeds":[42], "dataset_sha256":"abc", "frozen_corpus_sha256":"xyz",
                "target_evaluated_candidates":2, "task":"test"}), encoding="utf-8")
            for arm in "ABCD":
                report = {"run_id":arm, "dataset_sha256":"abc", "split_sha256":"spl",
                          "description":"test", "research_source":"frozen_replay" if arm!="A" else "none",
                          "strict_D_completed":True if arm=="D" else None,
                          "controlled_study":{"budget_complete":True,
                              "successfully_evaluated_candidates":2,"frozen_corpus_sha256":"xyz"},
                          "holdout":{"n_test":6},"best":{"cv_mean":.7,
                              "config":{"model":{"name":"random_forest"}}}}
                (suite/"reports"/f"seed_42_{arm}.json").write_text(json.dumps(report), encoding="utf-8")
                predictions = pd.DataFrame({"row_index":list(range(6)),
                    "true_label":["B","M","B","M","B","M"],
                    "predicted_label":["B","M","B","M","B","M"] if arm in "AB" else ["B","B","B","M","M","M"],
                    "probability_M":[.2,.8,.2,.8,.2,.8],
                    "split_sha256":["spl"]*6, "dataset_sha256":["abc"]*6, "seed":[42]*6})
                predictions.to_csv(suite/"predictions"/f"seed_42_{arm}_selected.csv",index=False)
            outcome = analyze_suite(suite,n_bootstrap=100)
            self.assertEqual(outcome["fully_matched_seeds"],1)
            df = pd.read_csv(suite/"analysis"/"paired_per_seed_bootstrap.csv")
            self.assertEqual(len(df), 18)
            # Deliberately violate arm D budget: exclude *all* per-seed pairs.
            d_path = suite/"reports"/"seed_42_D.json"
            r=json.loads(d_path.read_text(encoding="utf-8"));r["controlled_study"]["budget_complete"]=False
            d_path.write_text(json.dumps(r), encoding="utf-8")
            outcome = analyze_suite(suite,n_bootstrap=100)
            self.assertEqual(outcome["fully_matched_seeds"],0)

if __name__ == "__main__":
    unittest.main()
