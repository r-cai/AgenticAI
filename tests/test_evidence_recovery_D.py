"""Regression coverage for strict D empty-evidence recovery and honest abort."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
from sklearn.datasets import make_classification

from agents.schemas import GeneratorOutput
from rag.evidence_selection import select_evidence
from rag.evidence_recovery import recover_evidence
from tests.test_llm_grounded_D import voting_candidate
from orchestration.controller import run_experiment

VOTING_PAPER = {"title": "Tabular ensemble selection", "abstract":
                "We developed a Majority Voting Classifier for tabular classification.",
                "url": "https://example.org/not-a-real-paper"}
VOTING_EVIDENCE = {"evidence_id": "ev_test_voting", "paper_title": VOTING_PAPER["title"],
    "paper_url": VOTING_PAPER["url"],
    "source_excerpt": "Majority Voting Classifier",
    "source_grounded": True, "extraction_method": "llm_verified_excerpt",
    "suggested_operations": ["voting"]}


def config():
    return {
        'seed': 42, 'test_size': 0.2,
        'cv': {'folds': 3, 'repeats': 1, 'scoring': 'f1_macro', 'n_jobs': 1},
        'loop': {'max_iterations': 1, 'candidates_per_iteration': 1,
                 'patience': 3, 'max_failures': 2, 'max_batch_rounds': 1},
        'agents': {'generator': {'model': 'mock', 'temperature': 0},
                   'critic': {'enabled': False}},
        'rag': {'enabled': True, 'top_k': 2, 'timeout_sec': 5},
        'evidence': {'enabled': True, 'mode': 'llm', 'model': 'mock', 'max_papers': 2},
        'evidence_selection': {'enabled': True, 'max_records': 6,
                               'max_per_method': 2, 'require_structural': True},
        'evidence_composition': {'enabled': False},
        'llm_evidence_design': {'enabled': True, 'validation_retries': 1,
            'empty_evidence_recovery': {'enabled': True, 'max_rounds': 2,
                 'top_k': 6, 'extract_max_papers': 6}},
    }

class EvidenceRecoveryTests(unittest.TestCase):
    def test_mammography_metadata_can_support_voting_but_images_not(self):
        good = dict(VOTING_EVIDENCE, paper_title='Mammography metadata classification',
                    source_excerpt='Mammography descriptors were classified with Majority Voting Classifier.')
        bad = dict(VOTING_EVIDENCE, evidence_id='ev_bad',
                   paper_title='Mammogram images CNN classification')
        result = select_evidence([good, bad])
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]['evidence_id'], 'ev_test_voting')

    def test_recovery_is_bounded_and_does_not_fabricate_papers(self):
        calls = []
        def retrieve(queries, description, k, exclude=None):
            calls.append(queries)
            return [VOTING_PAPER] if len(calls) == 2 else []
        def extract(papers, max_papers):
            return [VOTING_EVIDENCE]
        output = recover_evidence('tabular diagnosis', {}, config(), [], [],
            retrieve_fn=retrieve, extract_fn=extract)
        self.assertEqual(len(output.evidence), 1)
        self.assertEqual(len(output.papers), 1)
        self.assertEqual(len(output.attempts), 1)
        self.assertEqual(output.attempts[0]['eligible_after_round'], 1)
        self.assertLessEqual(len(calls), 4)

    def run_mocked(self, with_recovery):
        X,y=make_classification(n_samples=80,n_features=6,n_informative=4,
                               n_redundant=0,random_state=42)
        df=pd.DataFrame(X,columns=[f'x{i}' for i in range(6)])
        df['diagnosis']=y
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            csv=tmp/'data.csv';df.to_csv(csv,index=False)
            cfg=config();cfg['llm_evidence_design']['empty_evidence_recovery']['enabled']=with_recovery
            (tmp/'cfg.yaml').write_text(yaml.safe_dump(cfg), encoding="utf-8")
            old=os.getcwd()
            try:
                os.chdir(tmp)
                def retrieve(queries, description, k, exclude=None):
                    return [VOTING_PAPER]
                def extract(papers, **kwargs):
                    return [VOTING_EVIDENCE] if papers and papers[0].get('title')==VOTING_PAPER['title'] else []
                # Must use the EVIDENCE id in voting_candidate schema.
                from tests.test_llm_grounded_D import EVIDENCE
                ev = dict(EVIDENCE)
                ev['source_excerpt']='Majority Voting Classifier'
                def extracted(papers, **kwargs):
                    return [ev] if papers and papers[0].get('title')==VOTING_PAPER['title'] else []
                def generated(**kwargs):
                    return GeneratorOutput(reasoning='offline mock', candidates=[voting_candidate()])
                with patch('orchestration.controller.build_research_brief',return_value=[{
                         'title':'MRI image CNN', 'abstract':'Image segmentation by CNN'}]), \
                     patch('orchestration.controller._retrieve_queries',side_effect=retrieve), \
                     patch('orchestration.controller.extract_evidence',side_effect=extracted), \
                     patch('orchestration.controller.generate',side_effect=generated) as gen, \
                     patch('orchestration.reporter.generate_pdf',return_value=tmp/'mock.pdf'):
                    report=run_experiment(str(csv),'diagnosis','tabular diagnosis','cfg.yaml')
                return report,gen.call_count
            finally:
                os.chdir(old)

    def test_strict_D_recovers_and_generates_models(self):
        report,calls=self.run_mocked(True)
        self.assertTrue(report['strict_D_completed'])
        self.assertEqual(report['research_influence']['llm_grounded_candidates_executed'],1)
        self.assertGreater(len(report['evidence_recovery_audit']),0)
        self.assertEqual(calls,1)

    def test_strict_D_honestly_aborts_without_evidence(self):
        report,calls=self.run_mocked(False)
        self.assertFalse(report['strict_D_completed'])
        self.assertEqual(report['stop_reason'],'insufficient_eligible_evidence')
        self.assertEqual(report['iterations_run'],0)
        self.assertEqual(calls,0)

if __name__=='__main__':
    unittest.main()
