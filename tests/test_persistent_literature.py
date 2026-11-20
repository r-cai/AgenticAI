"""Regression tests for recovery from historical archives and persistent retrieval."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from rag.knowledge_base import LiteratureStore
from rag.retriever import _retrieve_queries
from rag.evidence_selection import select_evidence
from scripts.restore_literature import import_sources


class PersistentRepositoryTests(unittest.TestCase):
    def test_bundled_restoration_has_genuine_data_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            store = LiteratureStore(Path(td) / 'literature.sqlite3')
            stats = store.stats()
            self.assertEqual(stats['papers_with_abstracts'], 78)
            self.assertEqual(stats['evidence_total'], 32)
            self.assertEqual(stats['evidence_excerpt_matched'], 21)
            self.assertEqual(stats['evidence_unverified'], 11)
            self.assertGreater(stats['provenance_links'], 12)
            selected = store.search('breast cancer feature selection', k=10)
            self.assertTrue(selected)
            self.assertTrue(all(p.get('abstract') for p in selected))

    def test_duplicate_archive_import_and_claim_upgrades_only_with_source(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            store = LiteratureStore(td / 'db.sqlite3', bootstrap_seed=False)
            p = {'id': 'http://arxiv.org/abs/2503.01234v1',
                 'title': 'Fictional Trial of Recursive Feature Elimination',
                 'abstract': 'We compared Recursive Feature Elimination (RFE) on tabular datasets.',
                 'source': 'arxiv', 'url': 'https://arxiv.org/abs/2503.01234v1'}
            ev = {'evidence_id': 'ev_archive_demo', 'paper_title': p['title'],
                  'paper_url': p['url'], 'source_excerpt': 'We compared Recursive Feature Elimination (RFE) on tabular datasets.',
                  'source_grounded': True, 'suggested_operations': ['rfe'], 'method': 'rfe',
                  'extraction_method': 'llm_verified_excerpt'}
            src = td / 'older_project.zip'
            with zipfile.ZipFile(src, 'w') as z:
                z.writestr('AgenticResearchPipeline/rag/cache/arxiv_legacy.json', json.dumps([p]))
                z.writestr('AgenticResearchPipeline/experiments/runs/x/evidence.json', json.dumps([ev]))
            first = import_sources([src], store)
            self.assertEqual(first['new_papers'], 1)
            self.assertEqual(first['new_evidence'], 1)
            self.assertEqual(store.stats()['evidence_excerpt_matched'], 1)
            again = import_sources([src], store)
            self.assertEqual(again['new_papers'], 0)
            self.assertEqual(again['new_evidence'], 0)
            self.assertEqual(store.stats()['papers_total'], 1)
            second_source = dict(p, id='https://arxiv.org/abs/2503.01234v5',
                                 url='https://arxiv.org/abs/2503.01234v5')
            store.add_papers([second_source], origin='re-import')
            self.assertEqual(store.stats()['papers_total'], 1)
            selected = store.verified_evidence_for([second_source])
            self.assertEqual(len(select_evidence(selected).records), 1)

    def test_unverified_report_kept_out_until_abstract_imported(self):
        with tempfile.TemporaryDirectory() as td:
            store = LiteratureStore(Path(td) / 'db.sqlite3', bootstrap_seed=False)
            excerpt = 'Voting Classifier improves clinical prediction.'
            claim = {'evidence_id': 'ev_legacy', 'paper_title': 'Legacy Article',
                     'paper_url': 'https://example.org/article', 'source_excerpt': excerpt,
                     'source_grounded': True, 'suggested_operations': ['voting']}
            store.add_evidence([claim], origin='old_report')
            self.assertEqual(store.stats()['evidence_unverified'], 1)
            self.assertEqual(store.verified_evidence_for([{'title': 'Legacy Article'}]), [])
            store.add_papers([{'title': 'Legacy Article', 'url': 'https://example.org/article',
                              'abstract': 'The study was observational and did not evaluate classifiers.'}], origin='paper_import')
            self.assertEqual(store.stats()['evidence_excerpt_matched'], 0)
            store.add_papers([{'title': 'Legacy Article', 'url': 'https://example.org/article',
                              'abstract': 'The study was observational. Voting Classifier improves clinical prediction.'}], origin='full_abstract')
            self.assertEqual(store.stats()['evidence_excerpt_matched'], 1)
            self.assertEqual(len(store.verified_evidence_for([{'title': 'Legacy Article'}])), 1)

    def test_persistent_local_retrieval_does_not_require_online_services(self):
        with tempfile.TemporaryDirectory() as td:
            settings = {'enabled': True, 'directory': td, 'refresh_mode': 'offline'}
            with patch('rag.retriever._search_arxiv', side_effect=AssertionError('network forbidden')), \
                 patch('rag.retriever._search_pubmed', side_effect=AssertionError('network forbidden')), \
                 patch('rag.retriever._search_semantic_scholar', side_effect=AssertionError('network forbidden')):
                found = _retrieve_queries(['recursive feature elimination classification'],
                       'breast cancer tabular dataset', 10, knowledge_base=settings)
            self.assertGreater(len(found), 0)
            store = LiteratureStore(Path(td) / 'literature.sqlite3')
            evidence = store.verified_evidence_for(found)
            self.assertTrue(select_evidence(evidence).records)

    def test_new_queries_search_then_cache_discovery_across_runs(self):
        with tempfile.TemporaryDirectory() as td:
            settings = {'enabled': True, 'directory': td, 'refresh_mode': 'on_unseen_query'}
            mocked = {'title': 'New study of compact classification',
                      'abstract': 'New structured feature selection and cross validation.',
                      'url': 'https://arxiv.org/abs/2509.12346', 'id': 'https://arxiv.org/abs/2509.12346',
                      'source': 'arxiv'}
            query = 'new scientific classification experiment'
            with patch('rag.retriever._search_arxiv', return_value=[mocked]) as arxiv, \
                 patch('rag.retriever._search_pubmed', return_value=[]), \
                 patch('rag.retriever._search_semantic_scholar', return_value=[]):
                _retrieve_queries([query], 'medical classification new research', 4,
                                  knowledge_base=settings)
                self.assertEqual(arxiv.call_count, 1)
                _retrieve_queries([query], 'medical classification new research', 4,
                                  knowledge_base=settings)
                self.assertEqual(arxiv.call_count, 1)  # recent query and sufficient local matches
                store = LiteratureStore(Path(td) / 'literature.sqlite3')
                self.assertTrue(any(p['title'] == mocked['title'] for p in store.papers()))
                _retrieve_queries([query], 'medical classification new research', 1,
                                  knowledge_base=settings)
                self.assertEqual(arxiv.call_count, 1)  # enough papers + recently attempted query


if __name__ == '__main__':
    unittest.main()
