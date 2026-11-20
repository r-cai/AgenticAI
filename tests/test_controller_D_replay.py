"""D integration: frozen literature, LLM-grounded candidate and report provenance."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
import pandas as pd
from sklearn.datasets import load_breast_cancer
import yaml

from agents.schemas import GeneratorOutput
from orchestration.controller import run_experiment
from tests.test_llm_grounded_D import EVIDENCE, voting_candidate

@contextmanager
def working_directory(path):
    old=os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)

class DReplayIntegration(unittest.TestCase):
    def test_controller_uses_frozen_corpus_and_reports_grounding(self):
        sample=load_breast_cancer()
        df=pd.DataFrame(sample.data,columns=[f'x{i}' for i in range(sample.data.shape[1])])
        df.insert(0,'id',range(len(df)))
        df['diagnosis']=1-sample.target
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            file=root/'wdbc.csv';df.to_csv(file,index=False)
            frozen=root/'frozen';frozen.mkdir()
            (frozen/'research.json').write_text(json.dumps([{
                'title':'Testing voting methods','abstract':'We use a Majority Voting Classifier.'
            }]), encoding="utf-8")
            (frozen/'evidence.json').write_text(json.dumps([EVIDENCE]), encoding="utf-8")
            cfg={
                'seed':42,'test_size':0.2,'data':{'exclude_columns':['id']},
                'cv':{'folds':3,'repeats':1,'scoring':'f1_macro','n_jobs':1},
                'loop':{'max_iterations':1,'candidates_per_iteration':1,'patience':2,'min_delta':0.002,'max_failures':2},
                'agents':{'generator':{'model':'mock','temperature':0},'critic':{'enabled':False}},
                'rag':{'enabled':True,'top_k':2,'timeout_sec':10,'replay_from_run':str(frozen)},
                'evidence':{'enabled':True,'mode':'llm','model':'mock','max_papers':2},
                'evidence_composition':{'enabled':False},
                'llm_evidence_design':{'enabled':True,'validation_retries':2},
            }
            (root/'config.yaml').write_text(yaml.safe_dump(cfg), encoding="utf-8")
            received=[]
            def mock_generate(**kwargs):
                received.append(kwargs)
                return GeneratorOutput(reasoning='literature-guided',candidates=[voting_candidate()])
            with working_directory(root), \
                patch('orchestration.controller.build_research_brief',side_effect=AssertionError('live retrieval must not run')), \
                patch('orchestration.controller.extract_evidence',side_effect=AssertionError('live extraction must not run')), \
                patch('orchestration.controller.generate',side_effect=mock_generate), \
                patch('orchestration.reporter.generate_pdf',return_value=root/'mock.pdf'):
                report=run_experiment(str(file),'diagnosis','classification','config.yaml')
            self.assertEqual(report['research_source'],'frozen_replay')
            self.assertTrue(report['llm_evidence_design_enabled'])
            self.assertEqual(report['research_influence']['llm_grounded_candidates_executed'],1)
            self.assertTrue(report['history'][1]['evidence_mapping']['trace'])
            self.assertEqual(report['history'][1]['origin'],'llm_evidence_design')
            self.assertEqual(len(received),1)
            self.assertTrue(received[0]['evidence_grounded'])
            self.assertTrue(report['research_evidence_sha256'])

if __name__=='__main__':unittest.main()
