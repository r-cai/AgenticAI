"""Guided Experiment D regression tests (no network or Ollama required)."""
import json
import unittest
from types import SimpleNamespace

from agents.deep import generate
from rag.guided_design import (
    DesignChoice, compile_choice, generate_grounded, allowed_methods,
)
from rag.llm_grounding import validate_llm_grounded_candidate
from rag.evidence_selection import select_evidence
from ml.trainer import build_pipeline
from ml.scoring import cross_validate_pipeline
import pandas as pd
from sklearn.datasets import load_breast_cancer

EVIDENCE = {
    "evidence_id": "ev_rfe_test", "paper_title": "Methods for numerical data",
    "source_excerpt": "We employ Recursive Feature Elimination (RFE) in classification.",
    "source_grounded": True, "extraction_method": "llm_verified_excerpt", "suggested_operations": ["rfe"],
}


def mock_response(ideas):
    return SimpleNamespace(message=SimpleNamespace(
        content=json.dumps({"choices": ideas})))


class GuidedDesignTests(unittest.TestCase):
    def test_valid_method_is_listed(self):
        menu = allowed_methods([EVIDENCE])
        self.assertEqual(menu[0]["permitted_operations"], ["rfe"])

    def test_caution_is_not_used(self):
        ev = dict(EVIDENCE, paper_title="A Cautionary Tale")
        self.assertEqual(allowed_methods([ev]), [])

    def test_invalid_citation_method_rejected(self):
        choice = DesignChoice(evidence_id="ev_rfe_test", operation="voting",
                              base_models=["logistic_regression", "random_forest"],
                              hypothesis="Test voting for numerical class labels.")
        with self.assertRaisesRegex(ValueError, "does not support"):
            compile_choice(choice, {"n_features": 30}, {"ev_rfe_test": EVIDENCE})

    def test_guided_compiles_and_validates_real_cv(self):
        ds = load_breast_cancer(as_frame=True)
        df = ds.frame.drop(columns=["target"])
        y = ds.target
        choice = {
            "evidence_id": "ev_rfe_test", "operation": "rfe",
            "base_models": ["logistic_regression"], "n_features": 12,
            "hypothesis": "Use RFE to reduce noisy input features.",
        }
        replies = iter([mock_response([choice])])
        out = generate_grounded(
            profile={"n_rows": 569, "n_features": 30, "n_classes": 2},
            research_evidence=[EVIDENCE],
            candidates_per_iteration=1,
            chat_fn=lambda **kwargs: next(replies),
        )
        self.assertEqual(len(out.candidates), 1)
        c = out.candidates[0]
        self.assertEqual(c.composition.kind, "pipeline")
        self.assertIn("rfe", [t.op for t in c.composition.transforms])
        self.assertEqual(c.evidence_ids, ["ev_rfe_test"])
        validate_llm_grounded_candidate(c, {"ev_rfe_test": EVIDENCE})
        pipe = build_pipeline(c, df, {"seed":42})
        metrics = cross_validate_pipeline(pipe, df.iloc[:140], y.iloc[:140],
                                          folds=2, repeats=1, seed=42, scoring="f1_macro")
        self.assertGreater(metrics["mean_cv_score"], 0.6)

    def test_guided_small_decision_retries_invalid_then_accepts(self):
        invalid = {"evidence_id":"ev_rfe_test","operation":"voting",
                   "base_models":["logistic_regression", "random_forest"],
                   "hypothesis":"Voting on the input data."}
        valid = {"evidence_id":"ev_rfe_test","operation":"rfe",
                 "base_models":["random_forest"], "n_features":8,
                 "hypothesis":"RFE then a forest to test feature reduction."}
        responses = iter([mock_response([invalid]), mock_response([valid])])
        out = generate_grounded(profile={"n_features":30}, research_evidence=[EVIDENCE],
                               candidates_per_iteration=1, validation_retries=1,
                               chat_fn=lambda **kwargs: next(responses))
        self.assertEqual(len(out.candidates), 1)
        self.assertEqual(len(out._validation_errors), 1)

    def test_guided_no_papers_does_not_fabricate(self):
        with self.assertRaisesRegex(ValueError, "No verified"):
            generate_grounded(profile={"n_features":30}, research_evidence=[],
                             chat_fn=lambda **kwargs: None)


if __name__ == "__main__": unittest.main()

class GuidedControllerIntegrationTest(unittest.TestCase):
    def test_full_controller_trains_guided_composition(self):
        """No real LLM: simulate its compact decision, not a full pipeline."""
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import yaml
        from sklearn.datasets import make_classification
        from orchestration.controller import run_experiment

        X, y = make_classification(n_samples=90, n_features=8, n_informative=5,
                                   n_redundant=0, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        df["diagnosis"] = y
        choices = {"choices":[{"evidence_id":"ev_rfe_test", "operation":"rfe",
                    "base_models":["logistic_regression"], "n_features":4,
                    "hypothesis":"RFE may reduce redundant features for this classification problem."}]}
        response = SimpleNamespace(message=SimpleNamespace(content=json.dumps(choices)))
        from types import ModuleType
        import sys
        mocked_ollama = ModuleType('ollama')
        mocked_ollama.chat = lambda **kwargs: response
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv = root / "cbis_dummy.csv"
            df.to_csv(csv,index=False)
            cfg = {
                'seed':42,'test_size':0.2,
                'cv':{'folds':2,'repeats':1,'scoring':'f1_macro','n_jobs':1},
                'loop':{'max_iterations':1,'candidates_per_iteration':1,
                        'max_failures':1,'max_batch_rounds':1,'patience':2,'min_delta':0.001},
                'agents':{'generator':{'model':'mock-qwen','temperature':0.0},
                          'critic':{'enabled':False}},
                'rag':{'enabled':True,'top_k':1,'timeout_sec':10},
                'evidence':{'enabled':True,'mode':'llm','model':'mock','max_papers':1},
                'evidence_selection':{'enabled':True,'max_records':2,
                                      'max_per_method':1,'require_structural':True},
                'evidence_composition':{'enabled':False},
                'llm_evidence_design':{'enabled':True,'strategy':'guided',
                                       'validation_retries':0,
                                       'empty_evidence_recovery':{'enabled':False}},
            }
            (root/'D_rag_llm_grounded.yaml').write_text(yaml.safe_dump(cfg), encoding="utf-8")
            previous = os.getcwd()
            try:
                os.chdir(root)
                with patch('orchestration.controller.build_research_brief',return_value=[{
                    'title':'Test RFE work','abstract':EVIDENCE['source_excerpt']}]), \
                     patch('orchestration.controller.extract_evidence',return_value=[EVIDENCE]), \
                     patch('orchestration.reporter.generate_pdf',return_value=root/'mock.pdf'), \
                     patch.dict(sys.modules,{'ollama':mocked_ollama}):
                    report=run_experiment(str(csv),'diagnosis','tabular benchmark','D_rag_llm_grounded.yaml')
                self.assertTrue(report['strict_D_completed'])
                self.assertEqual(report['research_influence']['llm_grounded_candidates_executed'],1)
                self.assertEqual(report['llm_grounded_strategy'],'guided')
                self.assertTrue((root/'experiments/reports/final_report_cbis_dummy_D.json').exists())
            finally:
                os.chdir(previous)
