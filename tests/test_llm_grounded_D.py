"""Offline regression tests: D must execute source-backed model COMPOSITIONS."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
from sklearn.datasets import load_breast_cancer
from agents.schemas import CandidateConfig, GeneratorOutput
from rag.llm_grounding import validate_llm_grounded_candidate
from orchestration.evidence_integration import generate_batch

EID = "ev_test_voting"
EVIDENCE = {"evidence_id": EID, "paper_title": "Testing voting methods",
            "source_excerpt": "We use a Majority Voting Classifier.",
            "source_grounded": True, "extraction_method": "llm_verified_excerpt",
            "suggested_operations": ["voting"]}


def voting_candidate():
    return CandidateConfig.model_validate({
        "rationale": "Test a voting system, choose bases as hypotheses",
        "preprocessing": {"numeric_imputer":"median", "numeric_scaler":"standard", "categorical_encoder":"onehot", "drop_columns":[]},
        "model": None,
        "composition": {"kind":"voting", "voting":"soft", "branches":[
            {"name":"lr", "model":{"name":"logistic_regression", "hyperparameters":{"C":1.0}}},
            {"name":"rf", "model":{"name":"random_forest", "hyperparameters":{"n_estimators":100}}},
        ]},
        "research_basis": ["Testing voting methods"], "evidence_ids":[EID]
    })

class GroundedDTests(unittest.TestCase):
    def test_voting_is_grounded(self):
        mapping=validate_llm_grounded_candidate(voting_candidate(), {EID:EVIDENCE})
        self.assertEqual(mapping['trace'][0]['matched_executable_operations'],['voting'])

    def test_missing_citation_rejected(self):
        c=voting_candidate().model_copy(update={'evidence_ids':[]})
        with self.assertRaisesRegex(ValueError,'requires at least one'):
            validate_llm_grounded_candidate(c,{EID:EVIDENCE})

    def test_incompatible_citation_rejected(self):
        c=voting_candidate()
        ev={**EVIDENCE, 'source_excerpt':'Our model is a Support Vector Machine.'}
        with self.assertRaisesRegex(ValueError, 'source'):
            validate_llm_grounded_candidate(c,{EID:ev})

    def test_cautionary_paper_rejected(self):
        ev={**EVIDENCE,'paper_title':'A Cautionary Tale',
            'source_excerpt':'Majority Voting Classifier may produce misleading results.'}
        with self.assertRaisesRegex(ValueError,'cautionary'):
            validate_llm_grounded_candidate(voting_candidate(),{EID:ev})

    def test_batch_gates_and_tracks_llm_grounded_design(self):
        X=pd.DataFrame({'a':[1,2,3,4,5,6], 'b':[6,5,4,3,2,1]})
        df=X.assign(label=[0,1,0,1,0,1]); c=voting_candidate()
        mock=Mock(return_value=GeneratorOutput(reasoning='grounded',candidates=[c]))
        cfg={'loop':{'candidates_per_iteration':1},'seed':42,
             'rag':{'enabled':True},'evidence':{'enabled':True},
             'evidence_composition':{'enabled':False},
             'llm_evidence_design':{'enabled':True,'validation_retries':2}}
        batch=generate_batch(evidence=[EVIDENCE],evidence_catalog={EID:EVIDENCE},
            profile={'n_rows':6,'n_features':2},train_df=df,target='label',cfg=cfg,
            tried=set(),generate_fn=mock,generator_kwargs={})
        self.assertEqual(len(batch.candidates),1)
        self.assertEqual(batch.origins[c.fingerprint()],'llm_evidence_design')
        self.assertEqual(batch.mapping[c.fingerprint()]['evidence_ids'],[EID])
        self.assertTrue(mock.call_args.kwargs['evidence_grounded'])

if __name__=='__main__': unittest.main()
