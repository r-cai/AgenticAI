"""Frozen snapshots are deterministic inputs for paired B/C/D comparisons."""
import json
import tempfile
import unittest
from pathlib import Path
from rag.knowledge_base import LiteratureStore
from scripts.freeze_literature import freeze


class FreezeLiteratureTests(unittest.TestCase):
    def test_freeze_produces_replay_compatible_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = LiteratureStore(root / 'knowledge.sqlite3')
            out = root / 'frozen'
            manifest = freeze(store, 'recursive feature elimination voting breast cancer classification', out, top_k=25)
            papers = json.loads((out / 'research.json').read_text(encoding='utf-8'))
            evidence = json.loads((out / 'evidence.json').read_text(encoding='utf-8'))
            self.assertEqual(len(papers), manifest['paper_count'])
            self.assertGreaterEqual(manifest['eligible_method_records'], 1)
            self.assertGreater(len(evidence), 0)
            self.assertTrue(all(e['source_grounded'] for e in evidence))
            self.assertTrue(all(e['source_excerpt'] for e in evidence))
            for ev in evidence:
                sources = [p for p in papers if p.get('title') == ev['paper_title']]
                self.assertTrue(sources)
                self.assertTrue(any(ev['source_excerpt'].casefold() in p['abstract'].casefold() for p in sources))
            for required in ['research.json','evidence.json','eligible_evidence.json','evidence_selection_audit.json','manifest.json']:
                self.assertTrue((out / required).is_file())

    def test_freeze_never_silently_overwrites_prior_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            store=LiteratureStore(root/'db.sqlite3')
            out=root/'frozen'
            freeze(store,'medical feature selection',out,top_k=12)
            with self.assertRaises(FileExistsError):
                freeze(store,'other topic',out,top_k=12)
            updated=freeze(store,'other topic',out,top_k=12,overwrite=True)
            self.assertEqual(updated['query'],'other topic')


if __name__ == '__main__':
    unittest.main()
