import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
import yaml

from agents.schemas import CandidateConfig, GeneratorOutput, ModelConfig, PreprocessingConfig
from orchestration.controller import run_experiment
from tests.test_evidence_compiler import evidence


@contextmanager
def cwd(path):
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


class ControllerIntegration(unittest.TestCase):
    def test_controller_reports_executed_evidence_composition(self):
        sample = load_breast_cancer()
        df = pd.DataFrame(sample.data, columns=[f"f{i}" for i in range(30)])
        df.insert(0, "id", range(len(df)))
        df["diagnosis"] = 1-sample.target
        ev = evidence()
        calls = []
        def fake_generate(**kwargs):
            calls.append(kwargs['candidates_per_iteration'])
            i = len(calls)
            return GeneratorOutput(reasoning="Simulated LLM", candidates=[
                CandidateConfig(rationale="compare", preprocessing=PreprocessingConfig(),
                                model=ModelConfig(name="logistic_regression", hyperparameters={"C": float(i)}))])
        def fake_extract(*args, **kwargs):
            return [ev]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            csv=root/'wdbc.csv'; df.to_csv(csv,index=False)
            cfg={"seed":42, "test_size":0.2,"data":{"exclude_columns":["id"]},
                 "cv":{"folds":3,"repeats":1,"scoring":"f1_macro","n_jobs":1},
                 "loop":{"max_iterations":2,"candidates_per_iteration":2,"patience":4,"min_delta":0.002,"max_failures":2},
                 "agents":{"generator":{"model":"fake","temperature":0},"critic":{"enabled":False}},
                 "rag":{"enabled":True,"top_k":2,"timeout_sec":10,"planner_model":"fake"},
                 "evidence":{"enabled":True,"model":"fake","max_papers":2,"mode":"llm"},
                 "evidence_composition":{"enabled":True,"reserve_slots":1,"allow_keyword":False}}
            (root/'config.yaml').write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with cwd(root), patch('orchestration.controller.build_research_brief',return_value=[{'title':'paper','abstract':'Recursive Feature Elimination (RFE) was evaluated.'}]), \
                    patch('orchestration.controller.build_followup_brief',return_value=[]), \
                    patch('orchestration.controller.extract_evidence',side_effect=fake_extract), \
                    patch('orchestration.controller.generate',side_effect=fake_generate), \
                    patch('orchestration.reporter.generate_pdf',return_value=root/'mock.pdf'):
                report=run_experiment(str(csv),'diagnosis','Classify malignant versus benign','config.yaml')
            self.assertEqual(report['research_influence']['literature_derived_candidates_executed'],1)
            self.assertGreaterEqual(report['research_influence']['literature_cited_candidates_executed'],1)
            self.assertEqual(report['research_influence']['generated_candidates_evaluated'],3)
            self.assertEqual(report['history'][1]['origin'],'evidence_compiler')
            self.assertEqual(report['history'][1]['composition']['kind'],'pipeline')
            self.assertEqual(sum(h['model'].startswith('composition:') for h in report['history']),1)
            self.assertEqual(calls,[1,2])
            self.assertNotIn('id',json.loads(next(root.glob('experiments/runs/*/profile.json')).read_text(encoding="utf-8"))['columns'])


if __name__=='__main__':
    unittest.main()
