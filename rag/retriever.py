# rag/retriever.py
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import requests
except ImportError:  # Offline unit tests or no-RAG runs can still import the controller.
    requests = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag.planner import plan_queries
from rag.knowledge_base import LiteratureStore, database_path

ARXIV_URL = "https://export.arxiv.org/api/query"
SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
TOP_K = 5
RESULTS_PER_QUERY = 8
MAX_ABSTRACT_CHARS = 2000
MIN_INTERVAL_SEC = 3.1
HTTP_TIMEOUT_SEC = 4
USER_AGENT = "agentic-automl-demo/0.1 (academic abstract retrieval)"
HTTP_ERROR = requests.RequestException if requests is not None else OSError
ATOM = {"a": "http://www.w3.org/2005/Atom"}
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "using",
    "based", "predict", "prediction", "predicting", "classify", "dataset",
    "data", "model", "are", "was", "will", "can", "our", "their",
    "which", "what", "how", "each", "whether", "given", "about",
    "task", "problem", "want", "need",
}
_last_arxiv_request = 0.0
_arxiv_unavailable = False

def _keywords(text: str, n: int) -> list[str]:
    seen, words = set(), []
    for word in re.findall(r"[a-z]{3,}", (text or "").lower()):
        if word not in STOPWORDS and word not in seen:
            words.append(word)
            seen.add(word)
    return words[:n]

def _simplify_query(query: str, n: int = 4) -> str:
    """Reduce long scholarly queries to their most useful keywords."""
    return " ".join(_keywords(query, n))


def _query_variants(query: str) -> list[str]:
    """Try original query, then a broader keyword query."""
    variants = [
        query.strip(),
        _simplify_query(query, 4),
        _simplify_query(query, 3),
    ]
    return list(dict.fromkeys(q for q in variants if q))


def _log_result(source: str, query: str, count: int) -> None:
    print(f"[rag] {source}: {count} papers for {query!r}")

def _profile_tags(profile: dict) -> list[str]:
    tags = []
    dist = profile.get("class_distribution") or {}
    try:
        counts = [float(v) for v in dist.values()]
        if counts and sum(counts) > 0 and min(counts) / sum(counts) < 0.15:
            tags.append("imbalanced")
    except (ValueError, TypeError, AttributeError):
        pass
    try:
        if int(profile.get("n_rows", 10**9)) < 1000:
            tags.append("small")
    except (ValueError, TypeError):
        pass
    cols = profile.get("columns") or {}
    if isinstance(cols, dict):
        try:
            if any(float(c.get("missing_pct", 0)) > 5 for c in cols.values()
                   if isinstance(c, dict)):
                tags.append("missing")
        except (ValueError, TypeError):
            pass
    return tags


def _initial_queries(description: str, profile: dict) -> list[str]:
    """Prefer methodological similarity over disease/application keywords."""
    tags = _profile_tags(profile)
    size = "small sample " if "small" in tags else ""
    queries = [
        f"{size}tabular classification regularization",
        f"{size}tabular classification logistic regression random forest",
        "tabular classification cross validation feature selection",
    ]
    if "imbalanced" in tags:
        queries[2] = "imbalanced tabular classification macro f1"
    if "missing" in tags:
        queries[2] = "tabular classification missing values imputation"
    return list(dict.fromkeys(queries))

def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _cache_path(source: str, query: str) -> Path:
    digest = hashlib.sha1(f"{source}:{query}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{source}_{digest}.json"

def _read_cache(path: Path):
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                return value
        except (OSError, ValueError):
            pass
    return None

def _save_cache(path: Path, papers: list[dict]) -> None:
    if papers:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(papers, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            print(f"[rag] cache write failed: {exc}")

def _parse_arxiv(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", ATOM):
        url = _clean(entry.findtext("a:id", "", ATOM))
        title = _clean(entry.findtext("a:title", "", ATOM))
        abstract = _clean(entry.findtext("a:summary", "", ATOM))
        if not (url and title and abstract) or "/api/errors" in url:
            continue
        papers.append({"id": url, "title": title, "abstract": abstract,
                       "year": (entry.findtext("a:published", "", ATOM) or "")[:4],
                       "url": url, "source": "arxiv", "doi": None})
    return papers

def _search_arxiv(query: str) -> list[dict]:
    """One attempt per query; stop arXiv for this retrieval round on 429/timeout."""
    global _last_arxiv_request, _arxiv_unavailable
    path = _cache_path("arxiv", query)
    cached = _read_cache(path)
    if cached:
        _log_result("arxiv cache", query, len(cached))
        return cached
    if requests is None or _arxiv_unavailable:
        return []
    terms = _keywords(query, 5)
    if not terms:
        return []
    delay = MIN_INTERVAL_SEC - (time.monotonic() - _last_arxiv_request)
    if delay > 0:
        time.sleep(delay)
    _last_arxiv_request = time.monotonic()
    try:
        response = requests.get(
            ARXIV_URL,
            params={"search_query": " AND ".join(f"all:{w}" for w in terms),
                    "start": 0, "max_results": RESULTS_PER_QUERY,
                    "sortBy": "relevance"},
            headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT_SEC,
        )
        if response.status_code == 429:
            _arxiv_unavailable = True
            print("[rag] arxiv rate limited; skipping remaining live arxiv requests")
            return []
        response.raise_for_status()
        papers = _parse_arxiv(response.text)
        _log_result("arxiv", query, len(papers))
        _save_cache(path, papers)
        return papers
    except (HTTP_ERROR, ET.ParseError) as exc:
        _arxiv_unavailable = True
        print(f"[rag] arxiv unavailable ({type(exc).__name__}); using cached/other sources")
        return []


def _search_semantic_scholar(
    query: str,
    limit: int = 10,
) -> list[dict]:

    if requests is None:
        return []

    for variant in _query_variants(query):
        path = _cache_path("semantic", variant)
        cached = _read_cache(path)

        if cached is not None:
            if cached:
                return cached
            continue

        try:
            response = requests.get(
                SEMANTIC_URL,
                params={
                    "query": variant,
                    "limit": limit,
                    "fields": (
                        "title,abstract,year,url,"
                        "citationCount,paperId,externalIds"
                    ),
                },
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT_SEC,
            )

            response.raise_for_status()

            items = response.json().get("data", [])

            papers = [
                {
                    "id": p.get("paperId") or p.get("url"),
                    "title": p.get("title"),
                    "abstract": _clean(p.get("abstract")),
                    "year": p.get("year"),
                    "url": p.get("url"),
                    "citations": p.get("citationCount", 0),
                    "source": "semantic_scholar",
                    "doi": (p.get("externalIds") or {}).get("DOI"),
                }
                for p in items
                if p.get("title") and p.get("abstract")
            ]

            _log_result("semantic_scholar", variant, len(papers))

            if papers:
                _save_cache(path, papers)
                return papers

        except (HTTP_ERROR, ValueError, TypeError) as exc:
            print(f"[rag] Semantic Scholar failed: {exc}")

            # Avoid immediately retrying after rate limiting.
            if (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code == 429
            ):
                break

    return []

def _search_pubmed(query: str, limit: int = 6) -> list[dict]:
    """Optional biomedical abstract source; network/API errors are non-fatal."""
    path = _cache_path("pubmed", query)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    if requests is None:
        return []
    try:
        ids = []
        for variant in _query_variants(query):
            search = requests.get(
                PUBMED_SEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": variant,
                    "retmax": limit,
                    "retmode": "json",
                    "sort": "relevance",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT_SEC,
            )

            search.raise_for_status()

            ids = search.json().get(
                "esearchresult", {}
            ).get("idlist", [])

            _log_result("pubmed", variant, len(ids))

            if ids:
                break
        if not ids:
            return []
        fetch = requests.get(PUBMED_FETCH_URL,
                             params={"db": "pubmed", "id": ",".join(ids),
                                     "retmode": "xml"},
                             headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT_SEC)
        fetch.raise_for_status()
        root = ET.fromstring(fetch.text)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            pmid = _clean("".join(article.find(".//MedlineCitation/PMID").itertext())
                          if article.find(".//MedlineCitation/PMID") is not None else "")
            title_el = article.find(".//ArticleTitle")
            abstract_elements = article.findall(".//Abstract/AbstractText")
            title = _clean(" ".join(title_el.itertext()) if title_el is not None else "")
            abstract = _clean(" ".join(" ".join(el.itertext()) for el in abstract_elements))
            doi = next((a.text for a in article.findall(".//ArticleId")
                        if a.attrib.get("IdType") == "doi"), None)
            year = article.findtext(".//PubDate/Year") or article.findtext(".//DateCompleted/Year")
            if pmid and title and abstract:
                papers.append({"id": f"pubmed:{pmid}", "title": title,
                               "abstract": abstract, "year": year,
                               "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                               "doi": doi, "source": "pubmed"})
    except (HTTP_ERROR, ET.ParseError, TypeError, ValueError) as exc:
        print(f"[rag] PubMed search {query!r} failed: {exc}")
        return []
    _save_cache(path, papers)
    return papers


def _rank(papers: list[dict], query: str) -> list[dict]:
    if not papers:
        return []
    docs = [f"{p['title']} {p['abstract']}" for p in papers]
    try:
        matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(
            docs + [query]
        )
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        return [papers[i] for i in scores.argsort()[::-1]]
    except ValueError:
        return papers


def _trim(text: str) -> str:
    if len(text) <= MAX_ABSTRACT_CHARS:
        return text
    return text[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + "..."


def _planned_queries(description, profile, feedback=None, history=None,
                     previous_queries=None, planner_model="qwen3:4b",
                     plan_path=None):
    try:
        plan = plan_queries(description, profile, feedback=feedback, history=history,
                            previous_queries=previous_queries,
                            model=planner_model)
        queries = plan.queries
        metadata = plan.model_dump()
    except Exception as exc:
        print(f"[rag] planner failed ({type(exc).__name__}: {exc}); using task-based fallback")
        queries = [" ".join(_keywords(description, 6)) or "machine learning"]
        metadata = {"research_focus": description, "queries": queries,
                    "reasoning": "Fallback after planner failure", "fallback": True}
    # Preserve at least one exact scientific task query alongside transferable
    # methodological searches. This matters for named benchmark datasets.
    task_text = (description or "").casefold()
    if "wisconsin" in task_text or "breast cancer" in task_text:
        exact = "Wisconsin diagnostic breast cancer feature selection classification"
        queries = [exact] + [q for q in queries if q.casefold() != exact.casefold()]
        queries = queries[:4]
        metadata["queries"] = queries
    if plan_path is not None:
        try:
            Path(plan_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[rag] could not save plan: {exc}")
    print(f"[rag] research queries: {queries}")
    return queries


def _kb_open(settings: dict | None = None) -> LiteratureStore | None:
    """Open persistent shared KB; fail open for runs without disk access."""
    settings = settings or {}
    if settings.get("enabled", True) is False:
        return None
    try:
        return LiteratureStore(database_path(settings))
    except (OSError, ValueError) as exc:
        print(f"[rag/kb] unavailable: {type(exc).__name__}: {exc}")
        return None


def _retrieve_queries(queries, description, k, exclude=None, knowledge_base=None):
    """Search persistent knowledge first; discover new papers on unseen queries.

    Default 'on_unseen_query' attempts live research for new planner queries
    even if the existing knowledge already has k relevant articles. After an
    attempt, cooldown suppresses redundant 429/timeouts for that same query.
    'offline' uses only the persistent database and existing exact query cache.
    This path never decides which evidence supports an operation; that remains
    the responsibility of evidence selection and the grounding validator.
    """
    global _arxiv_unavailable
    _arxiv_unavailable = False
    settings = knowledge_base or {}
    kb = _kb_open(settings)
    mode = settings.get("refresh_mode", "on_unseen_query")
    if mode not in ("offline", "on_miss", "on_unseen_query", "always"):
        raise ValueError("rag.knowledge_base.refresh_mode must be offline, on_miss, on_unseen_query or always")
    papers = {}
    seen_origin = {}

    def add_papers(results, *, origin):
        accepted = []
        for paper in results or []:
            if not paper.get('title') or not paper.get('abstract'):
                continue
            key = paper.get("doi") or paper.get("id") or paper.get("url") or paper['title'].casefold()
            normalized_key = str(key).casefold()
            if normalized_key not in papers:
                papers[normalized_key] = paper
                seen_origin[normalized_key] = 'knowledge_base' if origin == 'local_reuse' else origin
            accepted.append(paper)
        if kb is not None and accepted and origin != 'local_reuse':
            kb.add_papers(accepted, origin=origin)

    ranking_query = " ".join(queries) + " " + (description or "")
    if kb:
        local = kb.search(ranking_query, k=max(k * 3, 15),
                          min_score=float(settings.get('min_local_score', 0.06)))
        add_papers(local, origin='local_reuse')
        print(f"[rag/kb] {len(local)} relevant stored papers; database={kb.path}")

    # Older exact-query caches remain a secondary offline source and are
    # immediately imported into the durable database on first use.
    for query in queries:
        cached = _read_cache(_cache_path("arxiv", query))
        if cached:
            _log_result("arxiv cache", query, len(cached))
            add_papers(cached, origin=f'query_cache:arxiv:{query}')

    unseen = [q for q in queries if not kb or not kb.has_recent_attempt(
        q, cooldown_hours=float(settings.get('cooldown_hours', 24)))]
    need_live = mode != 'offline' and (
        mode == 'always' or
        (mode == 'on_miss' and len(papers) < k) or
        (mode == 'on_unseen_query' and (len(papers) < k or bool(unseen)))
    )

    if need_live:
        # A new task gets a fresh search even when old literature is reusable.
        # Query budget is limited to prevent repeated timeouts and 429 bursts.
        live_queries = (queries[:1] if len(papers) >= k else queries[:2])
        if mode == 'on_unseen_query' and len(papers) >= k:
            live_queries = unseen[:1]
        for query in live_queries:
            try:
                add_papers(_search_arxiv(query), origin=f'live_or_cached:arxiv:{query}')
            finally:
                if kb:
                    kb.record_attempt(query)
            if _arxiv_unavailable:
                break
        print(f"[rag] arxiv+knowledge collection: {len(papers)} papers")
        task = (description or '').lower()
        biomedical = any(x in task for x in ('cancer', 'medical', 'diagnos', 'disease', 'clinical', 'lesion', 'mammograph'))
        if biomedical and live_queries and (len(papers) < k or mode in ('always', 'on_unseen_query')):
            add_papers(_search_pubmed(live_queries[0]), origin=f'live_or_cached:pubmed:{live_queries[0]}')
        if len(papers) < k:
            for query in live_queries:
                add_papers(_search_semantic_scholar(query), origin=f'live_or_cached:semantic:{query}')
                if len(papers) >= k:
                    break
    else:
        print(f"[rag] using stored literature ({mode}); no live search required")

    excluded = {str(title).casefold() for title in (exclude or [])}
    fresh = [p for p in papers.values() if str(p['title']).casefold() not in excluded]
    ranked = _rank(fresh, ranking_query)
    selected = [{
        "title": p["title"], "abstract": _trim(p["abstract"]),
        "year": p.get("year"), "url": p.get("url"),
        "doi": p.get("doi"), "source": p.get("source"),
        "retrieval_origin": seen_origin.get(str(p.get('doi') or p.get('id') or p.get('url') or p['title'].casefold()).casefold(), 'unknown'),
    } for p in ranked[:k]]
    print(f"[rag] retrieval complete: {len(papers)} collected, {len(fresh)} fresh, {len(selected)} selected")
    return selected


def build_research_brief(description: str, profile: dict, k: int = TOP_K,
                         planner_model="qwen3:4b", plan_path=None,
                         knowledge_base=None) -> list[dict]:
    """Plan and retrieve papers in one background task."""
    queries = _planned_queries(description, profile or {}, planner_model=planner_model,
                               plan_path=plan_path)
    return _retrieve_queries(queries, description, k, knowledge_base=knowledge_base)


# --------------------------------------------------------------------------- #
# Follow-up retrieval: optional critic feedback + measured CV history
# --------------------------------------------------------------------------- #
def build_followup_brief(feedback: dict | None, description: str, profile: dict,
                         k: int = TOP_K, exclude=None, history=None,
                         previous_queries=None,
                         planner_model="qwen3:4b", plan_path=None,
                         knowledge_base=None) -> list[dict]:
    """Plan follow-up searches from CV history, with optional critic feedback."""
    queries = _planned_queries(description, profile or {}, feedback=feedback,
                               history=history,
                               previous_queries=previous_queries,
                               planner_model=planner_model, plan_path=plan_path)
    return _retrieve_queries(queries, description, k, exclude=exclude, knowledge_base=knowledge_base)

# determinisitc brief builder
'''
ISSUE_PHRASES = {
    "overfitting": "overfitting regularization",
    "underfitting": "underfitting model capacity",
    "class_imbalance": "imbalanced",
    "high_variance": "variance cross-validation",
    "weak_preprocessing": "feature preprocessing",
    "poor_hyperparameters": "hyperparameter optimization",
    "data_quality": "noisy missing data",
}
MODEL_TERMS = {
    "logistic_regression": "logistic regression",
    "random_forest": "random forest",
    "extra_trees": "extra trees",
    "gradient_boosting": "gradient boosting",
    "hist_gradient_boosting": "gradient boosting",
    "svc": "support vector machine",
    "knn": "nearest neighbor",
    "decision_tree": "decision tree",
}


def _feedback_text(feedback: dict) -> str:
    recs = [str(r.get("suggestion", "")) for r in feedback.get("recommendations", [])
            if isinstance(r, dict)]
    return " ".join([str(feedback.get("summary", "")), *recs])


def _feedback_queries(feedback: dict) -> list[str]:
    """At most two arXiv queries: the main issue type, and a model named in the advice."""
    queries = []
    for issue in feedback.get("issues", []):
        phrase = ISSUE_PHRASES.get(issue.get("type")) if isinstance(issue, dict) else None
        if phrase:
            queries.append(f"{phrase} classification")
            break
    text = _feedback_text(feedback).lower()
    for name, term in MODEL_TERMS.items():
        if name in text or name.replace("_", " ") in text:
            queries.append(f"{term} tabular classification")
            break
    if not queries:  # nothing recognizable: use the critic's own wording
        words = _keywords(feedback.get("summary", ""), 3)
        if words:
            queries.append(" ".join(words + ["classification"]))
    return list(dict.fromkeys(" ".join(dict.fromkeys(q.split())) for q in queries))


def build_followup_brief(feedback: dict | None, description: str, profile: dict,
                         k: int = TOP_K, exclude=None) -> list[dict]:
    """Papers relevant to the critic's latest feedback, skipping titles already shown.
    Returns [] when nothing new is found, so the caller keeps its previous research."""
    feedback = feedback or {}
    exclude = set(exclude or ())
    queries = _feedback_queries(feedback)
    if not queries:
        return []
    papers = {}
    for query in queries:
        for paper in _search_arxiv(query):
            papers.setdefault(paper["id"], paper)
    if len(papers) < k:
        for paper in _search_semantic_scholar(queries[0]):
            papers.setdefault(paper["id"], paper)
    fresh = [p for p in papers.values() if p["title"] not in exclude]
    print(f"[rag] follow-up {queries}: {len(papers)} found, {len(fresh)} not shown before")
    ranked = _rank(fresh, f"{description or ''} {_feedback_text(feedback)}")
    return [
        {"title": p["title"], "abstract": _trim(p["abstract"]),
         "year": p.get("year"), "url": p.get("url")}
        for p in ranked[:k]
    ]
'''

# Optional compatibility with code that expects retrieve(query, k).
def retrieve(query: str, k: int = 5) -> list[dict]:
    papers = _search_semantic_scholar(query, limit=max(k, 10))
    return [
        {"title": p["title"], "abstract": _trim(p["abstract"]),
         "year": p.get("year"), "url": p.get("url")}
        for p in _rank(papers, query)[:k]
    ]
