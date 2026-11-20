"""Create a reproducible task-specific research/evidence snapshot from the KB.

No external API calls and no LLM calls. Outputs are accepted directly by
rag.replay_from_run / main.py --rag-replay for matched B/C/D experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rag.knowledge_base import LiteratureStore, database_path, utcnow
from rag.evidence_selection import select_evidence


def stable_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def freeze(store: LiteratureStore, query: str, output: Path, *, top_k=25,
           min_score=0.04, overwrite=False) -> dict:
    if not query.strip() or top_k < 1:
        raise ValueError('A nonempty query and positive top_k are required')
    output = Path(output)
    files = [output / f for f in ('research.json', 'evidence.json', 'eligible_evidence.json',
                                  'evidence_selection_audit.json', 'manifest.json')]
    if not overwrite and any(x.exists() for x in files):
        raise FileExistsError(f'{output} contains a previous snapshot; use --overwrite or a new folder')
    papers = store.search(query, k=top_k, min_score=min_score)
    evidence = store.verified_evidence_for(papers)
    selection = select_evidence(evidence)
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in [('research.json', papers), ('evidence.json', evidence),
                           ('eligible_evidence.json', selection.records),
                           ('evidence_selection_audit.json', selection.audit)]:
        (output / filename).write_text(stable_dump(data), encoding='utf-8')
    manifest = {'created_utc': utcnow(), 'query': query, 'top_k': top_k,
                'min_score': min_score, 'paper_count': len(papers),
                'abstract_matched_evidence_count': len(evidence),
                'eligible_method_records': len(selection.records),
                'paper_titles': [p['title'] for p in papers],
                'research_sha256': hashlib.sha256(stable_dump(papers).encode()).hexdigest(),
                'evidence_sha256': hashlib.sha256(stable_dump(evidence).encode()).hexdigest(),
                'source_repository': str(store.path.resolve())}
    (output / 'manifest.json').write_text(stable_dump(manifest), encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True, help='Task and methodological search focus')
    parser.add_argument('--output', required=True, help='Snapshot folder')
    parser.add_argument('--top-k', type=int, default=25)
    parser.add_argument('--min-score', type=float, default=0.04)
    parser.add_argument('--db', help='Override persistent repository location')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    snapshot = freeze(LiteratureStore(args.db or database_path()), args.query,
                      Path(args.output), top_k=args.top_k, min_score=args.min_score,
                      overwrite=args.overwrite)
    print(json.dumps({k: v for k,v in snapshot.items() if k!='paper_titles'}, indent=2))
    if not snapshot['eligible_method_records']:
        print('[warning] no eligible composition methods; strict D will remain incomplete with this frozen corpus')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
