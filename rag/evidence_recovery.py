"""Bounded, auditable evidence recovery for strict LLM-grounded Experiment D.

No model or evidence is fabricated. Recovery requests actual publications, extracts
source-grounded excerpts and applies the SAME strict selection rules as the
initial pass. A failed rescue leaves D explicitly in an incomplete state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rag.evidence_selection import select_evidence

# Search for transferable tabular methods, rather than vague cancer-image work.
# Search input is task-agnostic; user task enters retrieval ranking.
SEARCH_ROUNDS = (
    (
        "recursive feature elimination classification",
        "voting ensemble classification",
    ),
    (
        "feature selection classification",
        "stacking ensemble classification",
    ),
)

@dataclass
class RecoveryResult:
    papers: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)


def recover_evidence(description: str, profile: dict, cfg: dict,
                     existing_papers: list[dict], existing_evidence: list[dict],
                     *, retrieve_fn: Callable, extract_fn: Callable) -> RecoveryResult:
    """Return incremental publications/claims and per-round diagnostics.

    This function NEVER relaxes require_structural/source-grounding/caution checks.
    `retrieve_fn` and `extract_fn` are injected for offline reproducibility.
    """
    settings = cfg.get("llm_evidence_design", {}).get("empty_evidence_recovery", {})
    result = RecoveryResult()
    if not settings.get("enabled", True):
        return result
    max_rounds = min(max(int(settings.get("max_rounds", 2)), 0), len(SEARCH_ROUNDS))
    top_k = max(1, min(20, int(settings.get("top_k", 10))))
    max_papers = max(1, min(20, int(settings.get("extract_max_papers", top_k))))
    target_groups = max(1, min(4, int(settings.get("target_method_groups", 1))))
    seen_titles = {str(p.get("title", "")).strip().casefold() for p in existing_papers}
    seen_ids = {str(e.get("evidence_id", "")) for e in existing_evidence}
    cumulative = list(existing_evidence)
    selector_cfg = cfg.get("evidence_selection", {})
    for n, queries in enumerate(SEARCH_ROUNDS[:max_rounds], 1):
        audit = {"round": n, "queries": list(queries), "new_papers": 0,
                 "new_evidence": 0, "eligible_after_round": 0,
                 "rejections": {}}
        fresh = []
        for query in queries:
            # Query separately so the first unrelated search cannot saturate top_k.
            try:
                papers = retrieve_fn([query], description, top_k, exclude=list(seen_titles))
            except Exception as exc:
                audit.setdefault("retrieval_errors", []).append(f"{type(exc).__name__}: {exc}")
                continue
            for p in papers or []:
                title = str(p.get("title", "")).strip().casefold()
                if title and title not in seen_titles:
                    fresh.append(p)
                    seen_titles.add(title)
        audit["new_papers"] = len(fresh)
        result.papers.extend(fresh)
        if fresh:
            try:
                extraction = extract_fn(fresh, max_papers=max_papers)
            except Exception as exc:
                audit["extraction_error"] = f"{type(exc).__name__}: {exc}"
                extraction = []
            for ev in extraction or []:
                eid = str(ev.get("evidence_id", ""))
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    result.evidence.append(ev)
                    cumulative.append(ev)
                    audit["new_evidence"] += 1
        selected = select_evidence(cumulative,
            max_records=int(selector_cfg.get("max_records", 9)),
            max_per_method=int(selector_cfg.get("max_per_method", 2)),
            require_structural=True)
        audit["eligible_after_round"] = len(selected.records)
        audit["distinct_eligible_methods"] = len({
            a.get("method_group") for a in selected.audit
            if a.get("status") == "selected" and a.get("method_group")
        })
        for entry in selected.audit:
            if entry.get("status") == "rejected":
                reason = entry.get("reason") or "unknown"
                audit["rejections"][reason] = audit["rejections"].get(reason, 0) + 1
        result.attempts.append(audit)
        if audit["distinct_eligible_methods"] >= target_groups:
            break
    return result
