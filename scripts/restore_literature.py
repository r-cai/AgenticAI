"""Recover paper abstracts and evidence from every previous pipeline version.

Examples (PowerShell, from current project root):
  python -m scripts.restore_literature --summary
  python -m scripts.restore_literature --scan "C:\\Users\\charl\\anaconda3\\AgenticAI"
  python -m scripts.restore_literature --source "C:\\backups\\AgenticResearchPipeline.zip"
  python -m scripts.restore_literature --source "C:\\oldproject\\experiments\\runs"

All imports are incremental; duplicates are merged by DOI/arXiv/URL/title aliases.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
import zipfile
from pathlib import Path

from rag.knowledge_base import LiteratureStore, database_path

MAX_JSON_BYTES = 24_000_000
SKIP_DIRS = {'__pycache__', '.git', '.venv', 'venv', 'node_modules', 'site-packages', 'anaconda3', '.mypy_cache'}


def kind_of_json(relative_name: str):
    norm = relative_name.replace('\\', '/').lower()
    name = norm.rsplit('/', 1)[-1]
    if '/rag/cache/' in '/' + norm or norm.startswith('rag/cache/'):
        if name.endswith('.json'):
            return 'papers'
    # Every self-contained paper from the new portable JSON KB, and the
    # bundled restored paper collection, is a valid legacy import source.
    if name.endswith('.json') and any(x in '/' + norm for x in
                                          ('/rag_knowledge_base/papers/', '/seed_papers/', '/papers/')):
        return 'paper_document'
    if name in {'research.json', 'research_next.json'} or name.startswith('research_round_') and name.endswith('.json'):
        return 'papers'
    if name in {'evidence.json', 'evidence_next.json', 'eligible_evidence.json'}:
        return 'evidence'
    if name.startswith('final_report') and name.endswith('.json'):
        return 'report'
    return None


def parse_content(name: str, raw: bytes, origin: str, papers, claims, errors):
    kind = kind_of_json(name)
    if not kind or len(raw) > MAX_JSON_BYTES:
        return
    try:
        document = json.loads(raw.decode('utf-8-sig'))
    except (UnicodeError, ValueError) as exc:
        errors.append(f'{origin}: {type(exc).__name__}')
        return
    if kind == 'paper_document':
        if isinstance(document, dict) and document.get('title'):
            sources = document.get('provenance')
            sources = sources if isinstance(sources, list) and sources else [origin]
            for source in sources:
                papers.append((str(source), [document]))
            for ev in document.get('evidence', []):
                if isinstance(ev, dict) and isinstance(ev.get('record'), dict):
                    origins = ev.get('provenance')
                    origins = origins if isinstance(origins, list) and origins else [origin]
                    for evidence_origin in origins:
                        claims.append((str(evidence_origin), [ev['record']]))
        else:
            errors.append(f'{origin}: malformed paper document')
    elif kind == 'report':
        raw_evidence = document.get('evidence_records') if isinstance(document, dict) else []
        if isinstance(raw_evidence, list):
            claims.append((origin, raw_evidence))
    elif isinstance(document, list):
        if kind == 'papers':
            papers.append((origin, document))
        else:
            claims.append((origin, document))


def _scan_dir(root: Path, papers, claims, errors):
    count = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        for name in files:
            file_path = Path(base) / name
            if file_path.suffix.lower() == '.zip':
                count += _scan_zip(file_path, papers, claims, errors)
                continue
            if kind_of_json(str(file_path.relative_to(root))) is None:
                continue
            count += 1
            try:
                if file_path.stat().st_size > MAX_JSON_BYTES:
                    continue
                parse_content(str(file_path.relative_to(root)), file_path.read_bytes(), str(file_path.resolve()), papers, claims, errors)
            except (OSError, ValueError) as exc:
                errors.append(f'{file_path}: {type(exc).__name__}')
    return count


def _scan_zip(path: Path, papers, claims, errors):
    count = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or kind_of_json(member.filename) is None:
                    continue
                if member.file_size > MAX_JSON_BYTES:
                    continue
                count += 1
                try:
                    parse_content(member.filename, archive.read(member),
                                  f'{path.resolve()}!{member.filename}', papers, claims, errors)
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                    errors.append(f'{path}!{member.filename}: {type(exc).__name__}')
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        errors.append(f'{path}: {type(exc).__name__}')
    return count


def import_sources(sources: list[str | Path], store: LiteratureStore) -> dict:
    papers, claims, errors = [], [], []
    scanned = 0
    for entry in sources:
        path = Path(entry).expanduser()
        if path.is_dir():
            scanned += _scan_dir(path, papers, claims, errors)
        elif path.is_file() and path.suffix.lower() == '.zip':
            scanned += _scan_zip(path, papers, claims, errors)
        elif path.is_file() and kind_of_json(str(path)):
            scanned += 1
            parse_content(str(path), path.read_bytes(), str(path.resolve()), papers, claims, errors)
        else:
            errors.append(f'{path}: missing or unsupported input')

    # First ingest ALL papers, then ALL evidence. Otherwise historical
    # report-only claims may incorrectly stay unverified until a later run.
    new_papers = updated = new_evidence = verified_matches = 0
    for origin, batch in papers:
        results = store.add_papers(batch, origin=origin)
        new_papers += results['new']
        updated += results['updated']
    for origin, batch in claims:
        results = store.add_evidence(batch, origin=origin)
        new_evidence += results['new']
        verified_matches += results['verified']
    return {'json_files_scanned': scanned, 'paper_batches': len(papers),
            'paper_records_observed': sum(len(batch) for _, batch in papers),
            'evidence_records_observed': sum(len(batch) for _, batch in claims),
            'new_papers': new_papers, 'duplicate_or_updated_papers': updated,
            'new_evidence': new_evidence, 'evidence_excerpt_matches_observed': verified_matches,
            'errors': errors, 'repository': store.stats()}


def import_legacy_databases(paths: list[str | Path], store: LiteratureStore) -> dict:
    """Recover papers, original origin records and evidence from older SQLite KBs.

    Only imports the known schema through SELECT. Never mutates the old file.
    """
    totals = {'imported_paper_records': 0, 'imported_evidence_records': 0, 'errors': []}
    for filename in paths:
        dbpath = Path(filename).expanduser()
        if not dbpath.is_file():
            totals['errors'].append(f'{dbpath}: database not found')
            continue
        try:
            # Open read-only through SQLite URI, avoiding accidental DB creation.
            with closing(sqlite3.connect(dbpath.resolve().as_uri() + '?mode=ro', uri=True)) as old:
                old.row_factory = sqlite3.Row
                paper_rows = list(old.execute('SELECT * FROM papers'))
                provenance = list(old.execute('SELECT * FROM provenance'))
                ev_rows = list(old.execute('SELECT * FROM evidence'))
                ev_origins = list(old.execute('SELECT * FROM evidence_provenance'))
            source_paper_origins = {}
            for r in provenance:
                source_paper_origins.setdefault(r['paper_key'], []).append(r['origin'])
            source_ev_origins = {}
            for r in ev_origins:
                source_ev_origins.setdefault(r['evidence_id'], []).append(r['origin'])
            for row in paper_rows:
                record = dict(row)
                record['id'] = record.get('paper_id')
                for origin in source_paper_origins.get(row['paper_key'], [f'legacy_db:{dbpath}']):
                    store.add_papers([record], origin=origin)
                totals['imported_paper_records'] += 1
            for row in ev_rows:
                record = json.loads(row['payload_json'])
                for origin in source_ev_origins.get(row['evidence_id'], [f'legacy_db:{dbpath}']):
                    store.add_evidence([record], origin=origin)
                totals['imported_evidence_records'] += 1
        except (OSError, sqlite3.DatabaseError, KeyError, ValueError) as exc:
            totals['errors'].append(f'{dbpath}: {type(exc).__name__}: {exc}')
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='append', default=[], help='Recursively inspect a folder containing all old projects and ZIPs')
    parser.add_argument('--source', action='append', default=[], help='Import a single ZIP, JSON, or run directory')
    parser.add_argument('--db', help='Optional SQLite database filename or parent directory')
    parser.add_argument('--legacy-db', action='append', default=[], help='Import an older persistent SQLite database (read-only)')
    parser.add_argument('--rebuild-index', action='store_true', help='Reindex every paper JSON; do not trust stored verified flags')
    parser.add_argument('--summary', action='store_true', help='Display knowledge repository statistics')
    parser.add_argument('--output', help='Optional JSON audit output path')
    args = parser.parse_args(argv)
    store = LiteratureStore(args.db or database_path())
    source_paths = args.scan + args.source
    report = import_sources(source_paths, store) if source_paths else {'repository': store.stats(), 'errors': []}
    if args.legacy_db:
        legacy = import_legacy_databases(args.legacy_db, store)
        report['legacy_databases'] = legacy
        report['errors'] += legacy['errors']
    if args.rebuild_index:
        rebuilt = store.rebuild_from_json()
        report['json_rebuild'] = rebuilt
        report['errors'] += rebuilt['errors']
    report['json_export'] = store.export_documents()
    report['repository'] = store.stats()
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get('errors'):
        print('Warning: not every input could be read; review errors above.')
    return 0 if not report.get('errors') else 1


if __name__ == '__main__':
    raise SystemExit(main())
