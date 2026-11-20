"""Persistent, source-aware scholarly repository: one portable JSON per paper.

Human-readable JSON documents in RAG_Knowledge_Base/papers/ are the portable
records. SQLite is an optional transactional search/deduplication index and can
be rebuilt from those JSON documents. Each document includes original abstract,
source metadata, all origins, evidence and separate verification status.

Abstract excerpt matching checks quotation provenance, not scientific validity
or independent publication authenticity. Source-only or unmatched claims must
never be promoted as evidence for strict Experiment D.
"""
from __future__ import annotations

import hashlib
import logging
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SEED = Path(__file__).with_name('seed_literature.sqlite3')
ENV_DIR = 'AGENTIC_RAG_KB_DIR'
DOC_FOLDER = 'papers'
LOG = logging.getLogger(__name__)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def database_path(config: dict | None = None) -> Path:
    config = config or {}
    # Environment variable is an explicit administrator/user override.
    directory = os.environ.get(ENV_DIR) or config.get('directory')
    if directory:
        base = Path(directory).expanduser()
    else:
        # For this user's normal Windows working directory this resolves to
        # C:/Users/charl/anaconda3/AgenticAI/RAG_Knowledge_Base, regardless of
        # how deeply nested the newly extracted pipeline package is.
        agentic_root = next((p for p in [Path.cwd(), *Path.cwd().parents]
                            if p.name.casefold() == 'agenticai'), None)
        base = agentic_root / 'RAG_Knowledge_Base' if agentic_root else Path.home() / '.agentic_research_kb'
    return base if base.suffix.casefold() == '.sqlite3' else base / 'literature.sqlite3'


def normalize_text(s) -> str:
    return ' '.join(str(s or '').split())


def canonical_title(s) -> str:
    return re.sub(r'[^a-z0-9]+', '', normalize_text(s).casefold())


def canonical_doi(s) -> str:
    v = normalize_text(s).casefold().strip().removeprefix('https://doi.org/').removeprefix('http://doi.org/').removeprefix('doi:')
    return v.strip('/ ') if v else ''


def canonical_url(s) -> str:
    if not s:
        return ''
    url = normalize_text(s).strip()
    parsed = urlsplit(url if '://' in url else 'https://' + url)
    host = parsed.hostname or ''
    path = parsed.path.rstrip('/')
    if 'arxiv.org' in host and '/abs/' in path:
        paper_id = path.split('/abs/')[-1]
        paper_id = re.sub(r'v\d+$', '', paper_id)
        return 'arxiv:' + paper_id.casefold()
    if 'pubmed.ncbi.nlm.nih.gov' in host:
        number = re.search(r'/([0-9]+)/?', parsed.path)
        if number:
            return 'pubmed:' + number.group(1)
    return urlunsplit(('https', host.casefold(), path, '', ''))


def paper_aliases(paper: dict) -> list[str]:
    values = []
    doi = canonical_doi(paper.get('doi'))
    if doi:
        values.append('doi:' + doi)
    for value in (paper.get('url'), paper.get('id')):
        if not value:
            continue
        v = normalize_text(value)
        if v.startswith('pubmed:'):
            alias = v.casefold()
        elif v.startswith('arxiv:'):
            alias = re.sub(r'v\d+$', '', v.casefold())
        else:
            alias = canonical_url(v)
        if alias:
            values.append('url:' + alias if not alias.startswith(('pubmed:', 'arxiv:')) else alias)
    title = canonical_title(paper.get('title'))
    if title:
        values.append('title:' + title)
    return list(dict.fromkeys(values))


def paper_from_evidence(item: dict) -> dict:
    return {'title': item.get('paper_title'), 'abstract': '', 'doi': item.get('paper_doi'),
            'url': item.get('paper_url'), 'year': item.get('paper_year'), 'source': 'legacy_report'}


class LiteratureStore:
    def __init__(self, path: str | Path | None = None, *, bootstrap_seed: bool = True):
        self.path = Path(path or database_path()).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_dir = self.path.parent / DOC_FOLDER
        # Record presence before the index is bootstrapped: existing human-readable
        # paper files can recover a lost SQLite index even after ZIP upgrades.
        new_index = not self.path.exists()
        existing_json = sorted(self.documents_dir.glob('*.json')) if self.documents_dir.is_dir() else []
        if bootstrap_seed and new_index and SEED.is_file() and SEED.resolve() != self.path.resolve():
            shutil.copy2(SEED, self.path)
        self._initialize()
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        if new_index and existing_json:
            self.rebuild_from_json()
        else:
            self.export_documents()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('PRAGMA busy_timeout=30000')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _initialize(self):
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS papers (
                    paper_key TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL DEFAULT '',
                    year TEXT, doi TEXT, url TEXT, paper_id TEXT, source TEXT,
                    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    alias TEXT PRIMARY KEY, paper_key TEXT NOT NULL REFERENCES papers(paper_key)
                );
                CREATE TABLE IF NOT EXISTS provenance (
                    paper_key TEXT NOT NULL REFERENCES papers(paper_key), origin TEXT NOT NULL,
                    PRIMARY KEY(paper_key, origin)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY, paper_key TEXT NOT NULL REFERENCES papers(paper_key),
                    payload_json TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL, first_seen TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence_provenance (
                    evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id), origin TEXT NOT NULL,
                    PRIMARY KEY(evidence_id, origin)
                );
                CREATE TABLE IF NOT EXISTS query_attempts (
                    query TEXT PRIMARY KEY, attempted_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evidence_paper_index ON evidence(paper_key, verified);
            ''')

    @staticmethod
    def _filename(key: str, title: str) -> str:
        # A stable key avoids data loss when titles or URLs are updated.
        slug = re.sub(r'[^a-z0-9]+', '-', normalize_text(title).casefold()).strip('-')[:68]
        return f'{slug or "untitled"}--{key}.json'

    @staticmethod
    def _metadata_checks(paper: dict) -> dict:
        """Two conservative automated checks, not an independent external audit.

        1. Bibliographic completeness and identifier syntax/source consistency.
        2. Extracted evidence must match the stored source abstract verbatim.
        """
        abstract = normalize_text(paper.get('abstract'))
        url = normalize_text(paper.get('url'))
        source = normalize_text(paper.get('source')).casefold()
        doi = canonical_doi(paper.get('doi'))
        parsed = urlsplit(url)
        host = (parsed.hostname or '').casefold()
        valid_url = parsed.scheme in ('https', 'http') and bool(host)
        identifier = str(paper.get('id') or paper.get('paper_id') or '').casefold()
        source_match = not (source == 'arxiv' and host not in ('arxiv.org', 'export.arxiv.org') and not identifier.startswith('arxiv:'))
        source_match = source_match and not (source == 'pubmed' and host != 'pubmed.ncbi.nlm.nih.gov' and not identifier.startswith('pubmed:'))
        check = {
            'title_present': bool(normalize_text(paper.get('title'))),
            'abstract_present': len(abstract) >= 20,
            'identifier_present': bool(valid_url or doi or str(paper.get('paper_id') or '').strip()),
            'source_identifier_consistent': bool(source_match),
        }
        status = ('citation_only' if not abstract else
                  'metadata_checked' if all(check.values()) else 'needs_review')
        return {'status': status, 'checks': check,
                'scope': 'local metadata structure and excerpt matching only',
                'independently_verified_with_publisher': False,
                'note': 'Retrieved API metadata is not an independent publication/authenticity check.'}

    def _serialize_paper(self, db, key: str) -> dict | None:
        row = db.execute('SELECT * FROM papers WHERE paper_key=?', (key,)).fetchone()
        if row is None:
            return None
        paper = dict(row)
        paper['id'] = paper.pop('paper_id')
        paper['schema_version'] = 1
        paper['provenance'] = [r[0] for r in db.execute(
            'SELECT origin FROM provenance WHERE paper_key=? ORDER BY origin', (key,))]
        paper['aliases'] = [r[0] for r in db.execute(
            'SELECT alias FROM aliases WHERE paper_key=? ORDER BY alias', (key,))]
        evidence = []
        for ev in db.execute('SELECT * FROM evidence WHERE paper_key=? ORDER BY evidence_id', (key,)):
            evidence.append({
                'evidence_id': ev['evidence_id'],
                'record': json.loads(ev['payload_json']),
                'verified': bool(ev['verified']),
                'verification_reason': ev['reason'],
                'first_seen': ev['first_seen'],
                'provenance': [r[0] for r in db.execute(
                    'SELECT origin FROM evidence_provenance WHERE evidence_id=? ORDER BY origin',
                    (ev['evidence_id'],))],
            })
        paper['evidence'] = evidence
        paper['verification'] = self._metadata_checks(paper)
        paper['verification']['excerpt_matched_count'] = sum(int(ev['verified']) for ev in evidence)
        paper['verification']['unverified_evidence_count'] = sum(not ev['verified'] for ev in evidence)
        return paper

    def _sync_documents(self, keys: set[str] | None = None) -> int:
        """Publish complete, indented paper JSON via atomic file replacement."""
        import tempfile
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            if keys is None:
                keys = {row[0] for row in db.execute('SELECT paper_key FROM papers')}
            payloads = [(key, self._serialize_paper(db, key)) for key in sorted(keys)]
        written = 0
        for key, doc in payloads:
            if not doc:
                continue
            destination = self.documents_dir / self._filename(key, doc['title'])
            raw = json.dumps(doc, ensure_ascii=False, indent=2, default=str) + '\n'
            # Avoid needless disk writes and timestamps on repeated searches.
            if destination.is_file() and destination.read_text(encoding='utf-8') == raw:
                continue
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.documents_dir,
                                                 prefix='.paper-', suffix='.tmp', delete=False) as handle:
                    temp_name = handle.name
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, destination)
                written += 1
            finally:
                if temp_name and Path(temp_name).exists():
                    Path(temp_name).unlink()
        return written

    def export_documents(self) -> dict:
        written = self._sync_documents()
        return {'paper_json_dir': str(self.documents_dir), 'updated_files': written,
                'paper_files': len(list(self.documents_dir.glob('*.json')))}

    def rebuild_from_json(self) -> dict:
        """Reindex imported JSON files; DO NOT trust stored 'verified' flags.

        Matching excerpts are always rechecked against the imported abstract.
        Read all files before writing any, to preserve user files and provenance.
        """
        files = sorted(self.documents_dir.glob('*.json'))
        docs = []
        errors = []
        for file in files:
            try:
                obj = json.loads(file.read_text(encoding='utf-8'))
                if not isinstance(obj, dict) or not obj.get('title'):
                    errors.append(str(file) + ': missing title or not a paper record')
                    continue
                docs.append((file, obj))
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f'{file}: {exc}')
        n_papers = n_claims = 0
        for file, doc in docs:
            original_origins = doc.get('provenance')
            sources = original_origins if isinstance(original_origins, list) and original_origins else [f'json_import:{file.name}']
            for source in sources:
                self.add_papers([doc], origin=str(source))
            n_papers += 1
        for file, doc in docs:
            for ev in doc.get('evidence', []):
                if not isinstance(ev, dict) or not isinstance(ev.get('record'), dict):
                    continue
                origins = ev.get('provenance')
                origins = origins if isinstance(origins, list) and origins else [f'json_import:{file.name}']
                for origin in origins:
                    self.add_evidence([ev['record']], origin=str(origin))
                n_claims += 1
        self.export_documents()
        return {'imported_papers': n_papers, 'imported_evidence': n_claims,
                'errors': errors, 'json_folder': str(self.documents_dir)}

    @staticmethod
    def _get_key(db, obj: dict) -> str | None:
        aliases = paper_aliases(obj)
        for alias in aliases:
            hit = db.execute('SELECT paper_key FROM aliases WHERE alias=?', (alias,)).fetchone()
            if hit:
                return hit['paper_key']
        return None

    def add_papers(self, papers: list[dict], *, origin: str) -> dict:
        counts = {'new': 0, 'updated': 0, 'skipped': 0}
        changed_keys: set[str] = set()
        with self._connect() as db:
            for paper in papers or []:
                if not isinstance(paper, dict):
                    counts['skipped'] += 1
                    continue
                title = normalize_text(paper.get('title'))
                if not title:
                    counts['skipped'] += 1
                    continue
                abstract = normalize_text(paper.get('abstract'))
                key = self._get_key(db, paper)
                now = utcnow()
                if key is None:
                    # Use stable title digest; aliases allow DOI/URL variants later.
                    key = 'p_' + hashlib.sha256(canonical_title(title).encode()).hexdigest()[:24]
                    db.execute('INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (key, title, abstract, str(paper.get('year') or ''),
                                canonical_doi(paper.get('doi')), str(paper.get('url') or ''),
                                str(paper.get('id') or ''), str(paper.get('source') or ''), now, now))
                    counts['new'] += 1
                else:
                    old = db.execute('SELECT * FROM papers WHERE paper_key=?', (key,)).fetchone()
                    # Longer abstract may be a more complete metadata export. Never overwrite
                    # a non-empty abstract with a shorter or missing one.
                    db.execute('''UPDATE papers SET abstract=?, doi=?, url=?, year=?, source=?, paper_id=?, last_seen=?
                        WHERE paper_key=?''',
                        (abstract if len(abstract) > len(old['abstract']) else old['abstract'],
                         canonical_doi(paper.get('doi')) or old['doi'],
                         str(paper.get('url') or '') or old['url'],
                         str(paper.get('year') or '') or old['year'],
                         str(paper.get('source') or '') or old['source'],
                         str(paper.get('id') or '') or old['paper_id'], now, key))
                    counts['updated'] += 1
                for alias in paper_aliases(paper):
                    # Existing conflicting identity is left untouched, never falsely merged.
                    db.execute('INSERT OR IGNORE INTO aliases(alias,paper_key) VALUES (?,?)', (alias, key))
                db.execute('INSERT OR IGNORE INTO provenance(paper_key,origin) VALUES (?,?)', (key, origin))
                self._verify_evidence_for_paper(db, key)
                changed_keys.add(key)
        if changed_keys:
            self._sync_documents(changed_keys)
        return counts

    @staticmethod
    def _verify_evidence_for_paper(db, key: str):
        row = db.execute('SELECT abstract FROM papers WHERE paper_key=?', (key,)).fetchone()
        abstract = normalize_text(row['abstract']).casefold() if row else ''
        for ev in db.execute('SELECT evidence_id,payload_json FROM evidence WHERE paper_key=?', (key,)):
            data = json.loads(ev['payload_json'])
            excerpt = normalize_text(data.get('source_excerpt')).casefold()
            verified = bool(abstract and excerpt and excerpt in abstract and data.get('source_grounded', True))
            reason = 'excerpt_matches_stored_abstract' if verified else 'no_matching_source_abstract'
            db.execute('UPDATE evidence SET verified=?, reason=? WHERE evidence_id=?',
                       (int(verified), reason, ev['evidence_id']))

    def add_evidence(self, records: list[dict], *, origin: str) -> dict:
        counts = {'new': 0, 'verified': 0, 'unverified': 0, 'skipped': 0}
        changed_keys: set[str] = set()
        with self._connect() as db:
            for item in records or []:
                if not isinstance(item, dict) or not item.get('paper_title') or not item.get('source_excerpt'):
                    counts['skipped'] += 1
                    continue
                paper = paper_from_evidence(item)
                key = self._get_key(db, paper)
                if key is None:
                    # Preserve unmatched report citations as metadata-only papers.
                    title = normalize_text(paper['title'])
                    key = 'p_' + hashlib.sha256(canonical_title(title).encode()).hexdigest()[:24]
                    now = utcnow()
                    db.execute('INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (key, title, '', str(paper.get('year') or ''), canonical_doi(paper.get('doi')),
                                str(paper.get('url') or ''), '', 'legacy_report', now, now))
                    for alias in paper_aliases(paper):
                        db.execute('INSERT OR IGNORE INTO aliases(alias,paper_key) VALUES (?,?)', (alias, key))
                ev_id = str(item.get('evidence_id') or '')
                if not ev_id:
                    digest = hashlib.sha256((canonical_title(item.get('paper_title')) + normalize_text(item.get('source_excerpt'))).encode()).hexdigest()[:14]
                    ev_id = 'ev_' + digest
                row = db.execute('SELECT abstract FROM papers WHERE paper_key=?', (key,)).fetchone()
                abstract = normalize_text(row['abstract']).casefold() if row else ''
                excerpt = normalize_text(item.get('source_excerpt')).casefold()
                verified = bool(abstract and excerpt and excerpt in abstract and item.get('source_grounded', True))
                reason = 'excerpt_matches_stored_abstract' if verified else 'no_matching_source_abstract'
                payload = dict(item)
                payload['evidence_id'] = ev_id
                old = db.execute('SELECT payload_json FROM evidence WHERE evidence_id=?', (ev_id,)).fetchone()
                if old is None:
                    db.execute('INSERT INTO evidence VALUES (?,?,?,?,?,?)',
                               (ev_id, key, json.dumps(payload, ensure_ascii=False, default=str), int(verified), reason, utcnow()))
                    counts['new'] += 1
                elif json.loads(old['payload_json']).get('source_excerpt') != item.get('source_excerpt'):
                    # An ID collision must never overwrite a different claim.
                    counts['skipped'] += 1
                    continue
                else:
                    db.execute('UPDATE evidence SET verified=?, reason=? WHERE evidence_id=?', (int(verified), reason, ev_id))
                db.execute('INSERT OR IGNORE INTO evidence_provenance VALUES (?,?)', (ev_id, origin))
                db.execute('INSERT OR IGNORE INTO provenance VALUES (?,?)', (key, origin))
                changed_keys.add(key)
                counts['verified' if verified else 'unverified'] += 1
        if changed_keys:
            self._sync_documents(changed_keys)
        return counts

    def papers(self) -> list[dict]:
        with self._connect() as db:
            result = [{'id': row['paper_id'], 'title': row['title'], 'abstract': row['abstract'],
                       'year': row['year'], 'doi': row['doi'], 'url': row['url'],
                       'source': row['source']} for row in db.execute('SELECT * FROM papers WHERE length(abstract)>0')]
        # A malformed or source-inconsistent newly scraped result remains on disk
        # with status 'needs_review', but is NOT supplied to the LLM as research.
        return [p for p in result if self._metadata_checks(p)['status'] == 'metadata_checked']

    def search(self, query: str, *, k: int = 5, min_score: float = 0.04) -> list[dict]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        papers = self.papers()
        if not papers or not query.strip():
            return []
        docs = [(p['title'] + ' ') * 3 + p['abstract'] for p in papers]
        try:
            matrix = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(docs + [query])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        except ValueError:
            return []
        indexes = scores.argsort()[::-1]
        return [papers[int(i)] for i in indexes if scores[i] >= min_score][:k]

    def verified_evidence_for(self, papers: list[dict]) -> list[dict]:
        out = {}
        with self._connect() as db:
            for paper in papers or []:
                key = self._get_key(db, paper)
                if not key:
                    continue
                bibliographic = db.execute('SELECT * FROM papers WHERE paper_key=?', (key,)).fetchone()
                if not bibliographic or self._metadata_checks(dict(bibliographic))['status'] != 'metadata_checked':
                    continue
                self._verify_evidence_for_paper(db, key)
                for ev in db.execute('SELECT * FROM evidence WHERE paper_key=? AND verified=1', (key,)):
                    data = json.loads(ev['payload_json'])
                    data['source_grounded'] = True
                    out[data['evidence_id']] = data
        return list(out.values())

    def has_recent_attempt(self, query: str, *, cooldown_hours: float = 24) -> bool:
        from datetime import timedelta
        with self._connect() as db:
            hit = db.execute('SELECT attempted_at FROM query_attempts WHERE query=?', (query.strip().casefold(),)).fetchone()
        if not hit:
            return False
        try:
            return datetime.fromisoformat(hit['attempted_at']) + timedelta(hours=cooldown_hours) > datetime.now(timezone.utc)
        except ValueError:
            return False

    def record_attempt(self, query: str):
        with self._connect() as db:
            db.execute('INSERT OR REPLACE INTO query_attempts VALUES (?,?)',
                       (query.strip().casefold(), utcnow()))

    def stats(self) -> dict:
        with self._connect() as db:
            def n(q): return db.execute(q).fetchone()[0]
            return {'papers_total': n('SELECT COUNT(*) FROM papers'),
                    'papers_with_abstracts': n("SELECT COUNT(*) FROM papers WHERE length(abstract)>0"),
                    'evidence_total': n('SELECT COUNT(*) FROM evidence'),
                    'evidence_excerpt_matched': n('SELECT COUNT(*) FROM evidence WHERE verified=1'),
                    'evidence_unverified': n('SELECT COUNT(*) FROM evidence WHERE verified=0'),
                    'provenance_links': n('SELECT COUNT(*) FROM provenance'),
                    'database': str(self.path.resolve()),
                    'paper_json_directory': str(self.documents_dir.resolve()),
                    'paper_json_files': len(list(self.documents_dir.glob('*.json')))}
