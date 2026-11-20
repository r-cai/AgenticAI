"""Experiment D: verify an LLM-designed pipeline against quoted literature.

This verifies only operational grounding (the paper mentions an operation that
really occurs in the pipeline). It does NOT verify the paper's scientific
validity, transferability, or claimed performance.
"""
from __future__ import annotations

import re
from typing import Any

from agents.schemas import CandidateConfig
from rag.evidence_compiler import (
    ENSEMBLES, FEATURE_OPS, IMAGE_MODALITY, UNSUPPORTED_DEPENDENCY,
    _grounded_ops, validate_candidate_citations,
)

# Conservative warnings: don't treat explicit negative findings as endorsement.
CAUTION = re.compile(
    r"\b(?:refrain\s+from|avoid\s+feature\s+selection|"
    r"misleading\s+(?:results?|output)|cautionary\s+tale|"
    r"feature\s+selection\s+(?:may\s+be\s+)?unreliable|"
    r"prone\s+to\s+high\s+variance)\b", re.I
)


def _actual_operations(candidate: CandidateConfig) -> set[str]:
    """Return the exact registered components the trainer will execute."""
    comp = candidate.composition
    if comp is None:
        return set()
    ops = {comp.kind}
    ops.update(t.op for t in comp.transforms)
    for b in comp.branches:
        ops.update(t.op for t in b.transforms)
        if b.model is not None:
            ops.add(b.model.name)
    if comp.model is not None:
        ops.add(comp.model.name)
    if comp.final_estimator is not None:
        ops.add(comp.final_estimator.name)
    return ops


def validate_llm_grounded_candidate(candidate: CandidateConfig,
                                    evidence_catalog: dict[str, dict]) -> dict[str, Any]:
    """Fail-closed validation for Level-2 D candidates; return provenance mapping.

    Requires at least one paper-supported STRUCTURAL operation (selector,
    feature construction or ensemble), not merely citing an SVM while running
    a different classifier. This does not claim the entire architecture came
    from a paper: the remaining details are explicitly LLM design choices.
    """
    if candidate.composition is None:
        raise ValueError("D requires a Level-2 composition, not a standalone model")
    if not candidate.evidence_ids:
        raise ValueError("D requires at least one retrieved evidence_id")
    citation_errors = validate_candidate_citations(candidate, evidence_catalog)
    if citation_errors:
        raise ValueError(";".join(citation_errors))

    actual = _actual_operations(candidate)
    trace = []
    structural_match = False
    for eid in candidate.evidence_ids:
        ev = evidence_catalog[eid]
        excerpt = str(ev.get("source_excerpt") or "")
        if CAUTION.search(excerpt) or CAUTION.search(str(ev.get("paper_title") or "")):
            raise ValueError(f"cautionary evidence cannot justify a positive method recommendation:{eid}")
        supported, warnings = _grounded_ops(ev)
        matched = sorted(set(supported) & actual)
        if not matched:
            raise ValueError(f"source does not match executable operations:{eid}")
        if set(matched) & (FEATURE_OPS | ENSEMBLES):
            structural_match = True
        trace.append({
            "evidence_id": eid,
            "paper_title": str(ev.get("paper_title", "")),
            "source_excerpt": excerpt,
            "supported_operations": supported,
            "matched_executable_operations": matched,
            "extraction_warnings": warnings,
            "claim_scope": "operations_only_not_hyperparameters_or_whole_architecture",
        })
    if not structural_match:
        raise ValueError("D requires a source-supported feature or ensemble operation")
    return {
        "method": "llm_evidence_grounded",
        "evidence_ids": list(candidate.evidence_ids),
        "trace": trace,
        "implementation_choices": (
            "Uncited architecture details and hyperparameters are LLM-proposed hypotheses, "
            "not proven settings from the cited papers"
        ),
    }
