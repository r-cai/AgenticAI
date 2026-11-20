"""Regression tests: optional missing IDs, candidate backfill and evidence quality."""
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from agents.schemas import CandidateConfig, GeneratorOutput, ModelConfig, PreprocessingConfig
from orchestration.controller import run_experiment
from orchestration.evidence_integration import generate_batch
from rag.evidence_selection import select_evidence


@contextmanager
def cwd(path):
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def make_candidate(n):
    return CandidateConfig(rationale=f"test C={n}", preprocessing=PreprocessingConfig(),
                           model=ModelConfig(name="logistic_regression", hyperparameters={"C": float(n)}))


class PipelineFixes(unittest.TestCase):
    def test_batch_backfills_after_duplicates_without_exceeding_budget(self):
        calls = []
        candidates = [[make_candidate(1)], [make_candidate(1), make_candidate(2)], [make_candidate(3)]]
        def fake_generate(**kwargs):
            calls.append(kwargs)
            return GeneratorOutput(reasoning="mock", candidates=candidates[len(calls)-1])
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "diagnosis": [0, 1, 0, 1, 0, 1]})
        cfg={"seed": 42, "loop": {"candidates_per_iteration": 3,"max_batch_rounds": 3},
             "rag": {"enabled": False}, "evidence": {"enabled": False},
             "evidence_composition": {"enabled": False}}
        batch=generate_batch(evidence=[],evidence_catalog={},profile={},train_df=df,
                             target="diagnosis",cfg=cfg,tried=set(),generate_fn=fake_generate,
                             generator_kwargs={"history": []})
        self.assertEqual(len(batch.candidates),3)
        self.assertEqual(len(calls),3)
        self.assertEqual([c.model.hyperparameters['C'] for c in batch.candidates],[1.,2.,3.])
        self.assertEqual(batch.audit[-1]['unfilled'],0)

    def test_evidence_quality_caution_and_image_and_method_diversity(self):
        def evidence(eid, title, excerpt):
            return {'evidence_id':eid,'paper_title':title,'source_excerpt':excerpt,
                    'source_grounded': True,'extraction_method':'llm_verified_excerpt'}
        items=[evidence('a','Paper A','Majority Voting Classifier'),
               evidence('b','Paper A','Majority Voting Classifier'),
               evidence('c','Paper B','Recursive Feature Elimination (RFE)'),
               evidence('d','A Cautionary Tale','Avoid feature selection on small datasets'),
               evidence('e','MRI research','MRI image segmentation'),
               evidence('f','Paper C','Feature selection for small tabular datasets')]
        out=select_evidence(items,max_records=5,max_per_method=2)
        self.assertEqual(set(x['evidence_id'] for x in out.records),{'a','c','f'})
        rejected={x['evidence_id']:x['reason'] for x in out.audit if x['status']=='rejected'}
        self.assertEqual(rejected['b'],'duplicate_paper_and_method')
        self.assertIn('cautionary',rejected['d'])
        self.assertIn('image_modality',rejected['e'])

    def test_optional_identifier_is_ignored_but_target_is_never_excluded(self):
        df=pd.DataFrame(load_breast_cancer().data,columns=[f'x{i}' for i in range(30)])
        df['diagnosis']=1-load_breast_cancer().target
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); file=root/'data.csv';df.to_csv(file,index=False)
            cfg={'seed':42,'test_size':0.2,'data':{'exclude_columns':['id'], 'missing_exclusions':'ignore'},
                 'cv':{'folds':3,'repeats':1,'scoring':'f1_macro','n_jobs':1},
                 'loop':{'max_iterations':1,'candidates_per_iteration':1,'patience':5,'max_batch_rounds':1},
                 'agents':{'generator':{'model':'mock','temperature':0},'critic':{'enabled':False}},
                 'rag':{'enabled':False},'evidence':{'enabled':False},
                 'evidence_composition':{'enabled':False}}
            (root/'config.yaml').write_text(yaml.safe_dump(cfg), encoding="utf-8")
            def fake_generate(**kwargs):
                return GeneratorOutput(reasoning='test',candidates=[make_candidate(1)])
            with cwd(root),patch('orchestration.controller.generate',side_effect=fake_generate),\
                 patch('orchestration.reporter.generate_pdf',return_value=root/'report.pdf'):
                report=run_experiment(str(file),'diagnosis','test','config.yaml')
            self.assertEqual(report['data_exclusion_audit']['absent_optional'],['id'])
            self.assertEqual(report['data_exclusion_audit']['removed'],[])
            self.assertEqual(report['research_influence']['generated_candidates_evaluated'],1)
            cfg['data']['exclude_columns']=['diagnosis']
            (root/'config.yaml').write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with cwd(root):
                with self.assertRaisesRegex(ValueError,'Cannot exclude the prediction target'):
                    run_experiment(str(file),'diagnosis','test','config.yaml')

if __name__=='__main__':unittest.main()
