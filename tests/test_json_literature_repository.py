"""Portable paper-per-JSON storage and verifiable recovery regression tests."""
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from rag.knowledge_base import LiteratureStore, database_path
from rag.retriever import _retrieve_queries
from scripts.restore_literature import import_sources, import_legacy_databases


class OneJsonPerPaperTests(unittest.TestCase):
    def test_seed_auto_creates_85_readable_paper_files_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LiteratureStore(Path(tmp) / 'literature.sqlite3')
            records = sorted(store.documents_dir.glob('*.json'))
            self.assertEqual(len(records), 85)
            sample = json.loads(records[0].read_text(encoding='utf-8'))
            self.assertIn('title', sample)
            self.assertIn('abstract', sample)
            self.assertIn('provenance', sample)
            self.assertIn('verification', sample)
            self.assertIn('evidence', sample)
            self.assertIn('schema_version', sample)
            self.assertEqual(store.stats()['paper_json_files'], 85)
            verified = sum(sum(ev['verified'] for ev in json.loads(p.read_text(encoding="utf-8"))['evidence'])
                           for p in records)
            self.assertEqual(verified, 21)

    def test_new_paper_writes_one_indented_file_and_updates_without_duplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LiteratureStore(Path(tmp) / 'literature.sqlite3', bootstrap_seed=False)
            item = {'title':'An Example of RFE and tabular classification',
                    'abstract':'We examine recursive feature elimination on clinical tables.',
                    'url':'https://arxiv.org/abs/2604.12345v1', 'source':'arxiv'}
            store.add_papers([item], origin='api:arxiv:first_query')
            store.add_papers([dict(item, url='https://arxiv.org/abs/2604.12345v3')], origin='api:arxiv:second_query')
            files = list(store.documents_dir.glob('*.json'))
            self.assertEqual(len(files), 1)
            raw = files[0].read_text(encoding='utf-8')
            self.assertIn('\n  "abstract"', raw)
            doc = json.loads(raw)
            self.assertEqual(doc['verification']['status'], 'metadata_checked')
            self.assertEqual(len(doc['provenance']), 2)
            self.assertFalse(doc['verification']['independently_verified_with_publisher'])

    def test_bad_source_is_quarantined_and_not_exposed_in_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LiteratureStore(Path(tmp) / 'literature.sqlite3', bootstrap_seed=False)
            paper = {'title':'Fake Arxiv source',
                     'abstract':'This is a long abstract allegedly collected from arxiv.',
                     'url':'https://not-arxiv.example.org/made-up', 'source':'arxiv'}
            store.add_papers([paper], origin='untrusted')
            file = next(store.documents_dir.glob('*.json'))
            data = json.loads(file.read_text(encoding='utf-8'))
            self.assertEqual(data['verification']['status'], 'needs_review')
            self.assertFalse(data['verification']['checks']['source_identifier_consistent'])
            self.assertEqual(store.papers(), [])

    def test_json_only_disaster_recovery_rechecks_all_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiteratureStore(root / 'literature.sqlite3', bootstrap_seed=False)
            paper = {'title':'Evidence based selection study', 'url':'https://arxiv.org/abs/2611.45678',
                     'source':'arxiv', 'abstract':'Recursive feature elimination (RFE) improved our benchmark.'}
            claim = {'evidence_id':'ev_test_original','paper_title':paper['title'],
                     'paper_url':paper['url'], 'source_excerpt':paper['abstract'],
                     'suggested_operations':['rfe'],'source_grounded':True}
            store.add_papers([paper], origin='original_api')
            store.add_evidence([claim], origin='old_report')
            self.assertEqual(store.stats()['evidence_excerpt_matched'], 1)
            file = next(store.documents_dir.glob('*.json'))
            tampered = json.loads(file.read_text(encoding='utf-8'))
            tampered['evidence'][0]['record']['source_excerpt'] = 'Invented methodology not in the paper.'
            tampered['evidence'][0]['verified'] = True  # malicious/human mistake, must ignore
            file.write_text(json.dumps(tampered, ensure_ascii=False, indent=2), encoding='utf-8')
            (root/'literature.sqlite3').unlink()
            recovered = LiteratureStore(root/'literature.sqlite3', bootstrap_seed=False)
            self.assertEqual(recovered.stats()['papers_total'], 1)
            self.assertEqual(recovered.stats()['evidence_excerpt_matched'], 0)
            self.assertEqual(recovered.stats()['evidence_unverified'], 1)
            restored = json.loads(file.read_text(encoding='utf-8'))
            self.assertFalse(restored['evidence'][0]['verified'])
            self.assertIn('old_report', restored['evidence'][0]['provenance'])

    def test_import_portable_json_from_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = {'schema_version':1, 'paper_key':'p_demo',
                   'title':'Reusing legacy literature', 'abstract':'This methodology is documented in an abstract.',
                   'source':'arxiv','url':'https://arxiv.org/abs/2603.45879',
                   'provenance':['run_A','run_B'],
                   'evidence':[{'record':{'evidence_id':'ev_demo','paper_title':'Reusing legacy literature',
                                          'paper_url':'https://arxiv.org/abs/2603.45879',
                                          'source_excerpt':'This methodology is documented in an abstract.',
                                          'source_grounded':True},'provenance':['run_D']}],
                   'verification':{'status':'metadata_checked'}}
            archive = root / 'old.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('old_project/RAG_Knowledge_Base/papers/demo.json',json.dumps(doc))
            store = LiteratureStore(root/'kb'/'literature.sqlite3', bootstrap_seed=False)
            result = import_sources([archive], store)
            self.assertEqual(result['new_papers'], 1)
            self.assertEqual(result['new_evidence'], 1)
            self.assertEqual(store.stats()['papers_total'], 1)
            record = json.loads(next(store.documents_dir.glob('*.json')).read_text(encoding="utf-8"))
            self.assertIn('run_A',record['provenance'])
            self.assertIn('run_B',record['provenance'])
            self.assertIn('run_D',record['evidence'][0]['provenance'])

    def test_legacy_sqlite_import_retains_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=LiteratureStore(root/'old'/'literature.sqlite3',bootstrap_seed=False)
            doc={'title':'Old persistent archive','abstract':'A publication with detailed laboratory evaluation.',
                 'url':'https://arxiv.org/abs/2605.98765', 'source':'arxiv'}
            old.add_papers([doc],origin='old_A')
            new=LiteratureStore(root/'new'/'literature.sqlite3',bootstrap_seed=False)
            summary=import_legacy_databases([old.path],new)
            self.assertEqual(summary['errors'],[])
            self.assertEqual(new.stats()['papers_total'],1)
            record=json.loads(next(new.documents_dir.glob('*.json')).read_text(encoding="utf-8"))
            self.assertIn('old_A',record['provenance'])

    def test_autodetect_shared_agenticai_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'AgenticAI'
            child=root/'PipelineVersions'/'D_fix'
            child.mkdir(parents=True)
            original=Path.cwd()
            try:
                os.chdir(child)
                with patch.dict(os.environ, {'AGENTIC_RAG_KB_DIR':''}):
                    self.assertEqual(database_path(),root/'RAG_Knowledge_Base'/'literature.sqlite3')
            finally:
                os.chdir(original)

    def test_live_results_written_to_shared_paper_json_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings={'enabled':True,'directory':tmp,'refresh_mode':'on_unseen_query'}
            paper={'title':'A newly discovered clinical feature-selection technique',
                   'abstract':'Our work compared recursive feature elimination and logistic regression in clinical tabular data.',
                   'id':'https://arxiv.org/abs/2610.00123','url':'https://arxiv.org/abs/2610.00123',
                   'source':'arxiv'}
            with patch('rag.retriever._search_arxiv',return_value=[paper]), \
                 patch('rag.retriever._search_semantic_scholar',return_value=[]), \
                 patch('rag.retriever._search_pubmed',return_value=[]):
                found=_retrieve_queries(['clinical feature selection'], 'medical tabular classification',k=1,
                                        knowledge_base=settings)
            self.assertTrue(found)
            docs=list((Path(tmp)/'papers').glob('*.json'))
            self.assertGreaterEqual(len(docs),85)
            self.assertTrue(any(json.loads(p.read_text(encoding="utf-8"))['title']==paper['title'] for p in docs))

if __name__=='__main__':
    unittest.main()
