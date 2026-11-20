"""Offline regression tests for dataset/arm-named final report copies."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
from sklearn.datasets import make_classification

from orchestration.report_naming import safe_stem, experiment_label, save_report_copies
from orchestration.controller import run_experiment


class ReportNamesTests(unittest.TestCase):
    def test_configuration_infers_correct_arms(self):
        for arm, name in [('A','A_no_rag.yaml'),('B','B_rag_replay.yaml'),
                          ('C','C_rag_evidence_compiler.yaml'),('D','D_rag_llm_grounded.yaml')]:
            self.assertEqual(experiment_label('configs/' + name), arm)
        self.assertEqual(experiment_label('configs/D_no_rag_level2.yaml'), 'NoRAG_Level2')

    def test_safe_dataset_identifier(self):
        self.assertEqual(safe_stem('data-cbis-ddsm'), 'data-cbis-ddsm')
        self.assertEqual(safe_stem('some folder / data-v2'), 'some_folder_data-v2')
        with self.assertRaises(ValueError):
            safe_stem('  ') 

    def test_named_copy_and_latest_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            runs=root/'experiments'/'runs'
            reports=root/'experiments'/'reports'
            for arm in 'ABCD':
                report={'run_id':'20261119_100000_'+arm, 'best':{'cv_mean':0.9}}
                folder=runs/report['run_id']
                folder.mkdir(parents=True)
                paths=save_report_copies(report, csv_path='data/data-cbis-ddsm.csv',
                    config_path=f'configs/{arm}_suitable.yaml', run_dir=folder, reports_root=reports)
                self.assertTrue(all(Path(path).is_file() for path in paths.values()))
                self.assertTrue(paths['latest_copy'].endswith(f'final_report_data-cbis-ddsm_{arm}.json'))
                self.assertEqual(json.loads(Path(paths['latest_copy']).read_text(encoding="utf-8"))['experiment_id'],arm)
            self.assertEqual(len(list(reports.glob('*.json'))),8)

    def test_rerun_preserves_archive_but_replaces_latest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive_paths=[]
            for n in [1,2]:
                run_dir=root/'runs'/f'run{n}'
                run_dir.mkdir(parents=True)
                report={'run_id':f'run{n}', 'best':{'cv_mean':n}}
                paths=save_report_copies(report,csv_path='data/data-3.csv',
                    config_path='configs/B_rag_context_only.yaml',run_dir=run_dir,
                    reports_root=root/'reports')
                archive_paths.append(Path(paths['archived_copy']))
            self.assertTrue(all(path.exists() for path in archive_paths))
            self.assertEqual(json.loads(archive_paths[0].read_text(encoding="utf-8"))['best']['cv_mean'],1)
            self.assertEqual(json.loads(Path(paths['latest_copy']).read_text(encoding="utf-8"))['best']['cv_mean'],2)
            with self.assertRaises(FileExistsError):
                save_report_copies({'run_id':'run2'},csv_path='data/data-3.csv',
                    config_path='configs/B_rag_context_only.yaml',run_dir=run_dir,
                    reports_root=root/'reports')

    def test_actual_controller_run_creates_named_report(self):
        # No live Ollama or RAG; test only output behavior using real training/CV.
        X,y=make_classification(n_samples=60,n_features=5,n_informative=3,
                                n_redundant=0,random_state=42)
        df=pd.DataFrame(X,columns=[f'x{i}' for i in range(5)])
        df['diagnosis']=['M' if val else 'B' for val in y]
        cfg={'seed':42,'test_size':0.2,'cv':{'folds':3,'repeats':1,'n_jobs':1,'scoring':'f1_macro'},
             'loop':{'max_iterations':0,'candidates_per_iteration':1,'patience':2,
                     'max_failures':2,'max_batch_rounds':1},
             'rag':{'enabled':False,'top_k':1,'timeout_sec':1},
             'evidence':{'enabled':False},
             'evidence_composition':{'enabled':False},
             'agents':{'generator':{'model':'unused','temperature':0},
                       'critic':{'enabled':False}},
             'data':{'exclude_columns':['id'],'missing_exclusions':'ignore'}}
        with tempfile.TemporaryDirectory() as temp:
            original=Path.cwd()
            try:
                os.chdir(temp)
                (Path(temp)/'data').mkdir()
                (Path(temp)/'configs').mkdir()
                df.to_csv('data/data-cbis-ddsm.csv',index=False)
                Path('configs/A_no_rag.yaml').write_text(yaml.safe_dump(cfg), encoding="utf-8")
                with patch('orchestration.reporter.generate_pdf',return_value=Path('mock.pdf')):
                    report=run_experiment('data/data-cbis-ddsm.csv','diagnosis',
                              'tabular classification','configs/A_no_rag.yaml')
                named=Path('experiments/reports/final_report_data-cbis-ddsm_A.json')
                self.assertTrue(named.exists())
                saved=json.loads(named.read_text(encoding="utf-8"))
                self.assertEqual(saved['dataset_id'],'data-cbis-ddsm')
                self.assertEqual(saved['experiment_id'],'A')
                self.assertIn('pdf_report',saved)
                self.assertEqual(saved['run_id'],report['run_id'])
            finally:
                os.chdir(original)

if __name__=='__main__':
    unittest.main()


class BackfillTests(unittest.TestCase):
    def test_backfill_preserves_original_report(self):
        from scripts.backfill_named_reports import backfill
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run_dir=root/'runs'/'20261119_100000'
            run_dir.mkdir(parents=True)
            report_file=run_dir/'final_report.json'
            original='{"run_id":"20261119_100000","dataset":"data/data-3.csv"}\n'
            report_file.write_text(original, encoding='utf8')
            (run_dir/'config_snapshot.json').write_text(json.dumps({
                'rag':{'enabled':True},
                'evidence_composition':{'enabled':False},
                'llm_evidence_design':{'enabled':False}
            }), encoding="utf-8")
            result=backfill(root/'runs',root/'reports')
            self.assertEqual(result['copied'],1)
            self.assertEqual(report_file.read_text(encoding="utf-8"),original)
            alias=root/'reports'/'final_report_data-3_B.json'
            self.assertTrue(alias.exists())
            self.assertEqual(json.loads(alias.read_text(encoding="utf-8"))['experiment_id'],'B')
