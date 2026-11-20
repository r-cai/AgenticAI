"""Offline tests: no Ollama or remote literature APIs required."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from agents.schemas import GeneratorOutput, CandidateConfig, ModelConfig, PreprocessingConfig
from rag.evidence_compiler import translate_evidence, validate_candidate_citations
from orchestration.evidence_integration import generate_batch
from ml.trainer import run_cv


def profile():
    return {"target": "diagnosis", "n_features": 31,
            "columns": {"id": {"type": "numeric"},
                        **{f"f{i}": {"type": "numeric"} for i in range(30)}}}


def evidence(eid="ev_rfe", method="rfe", excerpt="Recursive Feature Elimination (RFE) was evaluated.",
             title="Feature selection methods in medical classification", extraction="llm_verified_excerpt"):
    return {"evidence_id": eid, "paper_title": title,
            "source_excerpt": excerpt, "source_grounded": True,
            "extraction_method": extraction, "suggested_operations": [method]}


def dataset():
    bundle = load_breast_cancer()
    X = pd.DataFrame(bundle.data, columns=[f"f{i}" for i in range(30)])
    X.insert(0, "id", np.arange(1, len(X) + 1))
    X["diagnosis"] = 1 - bundle.target  # malignant=1
    return X


class EvidenceCompilerTests(unittest.TestCase):
    def test_rfe_to_real_pipeline_with_provenance(self):
        out = translate_evidence([evidence()], profile())
        self.assertEqual(len(out.candidates), 1)
        cand = out.candidates[0]
        self.assertEqual(cand.composition.kind, "pipeline")
        self.assertIn("rfe", [t.op for t in cand.composition.transforms])
        self.assertEqual(cand.preprocessing.drop_columns, ["id"])
        self.assertEqual(cand.evidence_ids, ["ev_rfe"])
        self.assertEqual(validate_candidate_citations(cand, {"ev_rfe": evidence()}), [])

    def test_voting_generated_without_llm_code(self):
        ev = evidence("ev_vote", "voting", "We use a Majority Voting Classifier to combine models.")
        out = translate_evidence([ev], profile())
        self.assertEqual(out.candidates[0].composition.kind, "voting")
        self.assertEqual(len(out.candidates[0].composition.branches), 2)

    def test_image_paper_is_rejected(self):
        ev = evidence(title="Breast MRI U-Net segmentation", excerpt="U-Net segmentation of MRI images was evaluated.")
        out = translate_evidence([ev], profile())
        self.assertEqual(len(out.candidates), 0)
        self.assertEqual(out.audit[0]["reason"], "image_method_incompatible_with_numeric_tabular_data")

    def test_unimplemented_sampling_is_not_misrepresented(self):
        ev = evidence(excerpt="Random undersampling and Feature Selection performed well with SVM.")
        out = translate_evidence([ev], profile())
        self.assertEqual(len(out.candidates), 0)
        self.assertEqual(out.audit[0]["reason"], "excerpt_requires_unsupported_method")

    def test_invented_rfe_keyword_is_rejected(self):
        ev = evidence(excerpt="We used feature selection in a small data study.")
        out = translate_evidence([ev], profile())
        self.assertEqual(out.candidates[0].composition.transforms[-1].op, "select_k_best")
        self.assertNotIn("rfe", [t.op for t in out.candidates[0].composition.transforms])

    def test_keyword_fallback_not_auto_promoted(self):
        ev = evidence(extraction="keyword_fallback")
        self.assertEqual(translate_evidence([ev], profile()).candidates, [])

    def test_duplicate_rejected(self):
        first = translate_evidence([evidence()], profile()).candidates[0]
        out = translate_evidence([evidence()], profile(), tried={first.fingerprint()})
        self.assertEqual(out.candidates, [])
        self.assertEqual(out.audit[0]["reason"], "duplicate_executable_architecture")

    def test_invalid_citation_rejected(self):
        cand = translate_evidence([evidence()], profile()).candidates[0]
        cand.evidence_ids = ["ev_nonexistent"]
        self.assertTrue(validate_candidate_citations(cand, {"ev_rfe": evidence()}))

    def test_train_real_rfe_composition(self):
        d = dataset()
        cand = translate_evidence([evidence()], profile()).candidates[0]
        result = run_cv(cand, d.iloc[:150], "diagnosis", {"seed": 42, "cv": {"folds": 3, "repeats": 1, "scoring": "f1_macro", "n_jobs": 1}})
        self.assertGreater(result["mean_cv_score"], 0.8)
        self.assertEqual(result["model_name"], "composition:pipeline")

    def test_equal_budget_and_selection(self):
        ev = evidence()
        d = dataset().iloc[:140]
        cfg = {"seed": 42, "loop": {"candidates_per_iteration": 3},
               "rag": {"enabled": True}, "evidence": {"enabled": True},
               "evidence_composition": {"enabled": True, "reserve_slots": 1}}
        calls = []
        def llm(**kwargs):
            calls.append(kwargs["candidates_per_iteration"])
            return GeneratorOutput(reasoning="No literature claim", candidates=[
                CandidateConfig(rationale="baseline improvement", preprocessing=PreprocessingConfig(),
                                model=ModelConfig(name="random_forest", hyperparameters={"max_depth": 4})),
                CandidateConfig(rationale="baseline improvement 2", preprocessing=PreprocessingConfig(),
                                model=ModelConfig(name="logistic_regression", hyperparameters={"C": 0.9}))])
        b = generate_batch(evidence=[ev], evidence_catalog={ev["evidence_id"]: ev},
                           profile=profile(), train_df=d, target="diagnosis", cfg=cfg,
                           tried=set(), generate_fn=llm, generator_kwargs={})
        self.assertEqual(calls, [2])
        self.assertEqual(len(b.candidates), 3)
        self.assertEqual(sum(x.composition is not None for x in b.candidates), 1)
        self.assertEqual(b.origins[b.candidates[0].fingerprint()], "evidence_compiler")

    def test_no_rag_does_not_translate(self):
        ev = evidence()
        cfg = {"loop": {"candidates_per_iteration": 1},
               "rag": {"enabled": False}, "evidence": {"enabled": False},
               "evidence_composition": {"enabled": True, "reserve_slots": 1}}
        def llm(**kwargs):
            return GeneratorOutput(reasoning="control", candidates=[
                CandidateConfig(rationale="control", preprocessing=PreprocessingConfig(),
                                model=ModelConfig(name="logistic_regression"))])
        b = generate_batch(evidence=[ev], evidence_catalog={ev["evidence_id"]: ev},
                           profile=profile(), train_df=dataset().iloc[:50], target="diagnosis",
                           cfg=cfg, tried=set(), generate_fn=llm, generator_kwargs={})
        self.assertEqual(len(b.candidates), 1)
        self.assertEqual(b.candidates[0].evidence_ids, [])
        self.assertIsNone(b.candidates[0].composition)


if __name__ == "__main__":
    unittest.main()
