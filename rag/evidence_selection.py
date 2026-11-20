"""Conservative, deterministic selection of literature evidence for model design.

This is NOT an assessment of whether a scientific claim is true. The aim is
operational relevance, source-excerpt grounding and methodological diversity.
The raw evidence is always retained separately for an audit.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rag.evidence_compiler import (
    ENSEMBLES, FEATURE_OPS, IMAGE_MODALITY, MODEL_OPS, _grounded_ops,
)
from rag.llm_grounding import CAUTION

PRIORITY = ("voting", "rfe", "stacking", "pca", "select_k_best",
            "variance_threshold", "polynomial_features")


@dataclass
class EvidenceSelection:
    records: list[dict] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)


def select_evidence(records: list[dict], *, max_records: int = 9,
                    max_per_method: int = 2, require_structural: bool = True) -> EvidenceSelection:
    """Balanced round-robin by supported method; reject cautions and modality mismatch.

    Scoring never uses labels, holdout data or CV results. For now it targets
    numeric/tabular model composition and rejects image-specific methods.
    """
    if max_records < 1 or max_per_method < 1:
        raise ValueError("Evidence-selection limits must be positive")
    out = EvidenceSelection()
    groups: dict[str, list[dict]] = defaultdict(list)
    valid_ops = set(PRIORITY) | MODEL_OPS
    for item in records or []:
        eid = item.get("evidence_id")
        title = str(item.get("paper_title") or "")
        excerpt = str(item.get("source_excerpt") or "")
        entry: dict[str, Any] = {"evidence_id": eid, "paper_title": title,
                                 "status": "rejected", "reason": None}
        if not eid or not title or not excerpt or not item.get("source_grounded"):
            entry["reason"] = "missing_or_ungrounded_excerpt"
        elif CAUTION.search(title + " " + excerpt):
            entry["reason"] = "cautionary_not_positive_method_support"
        elif IMAGE_MODALITY.search(title + " " + excerpt):
            entry["reason"] = "image_modality_not_applicable_to_tabular_modeling"
        else:
            ops, warnings = _grounded_ops(item)
            ops = [o for o in ops if o in valid_ops]
            entry["supported_operations"] = ops
            entry["extraction_warnings"] = warnings
            struct_ops = [o for o in PRIORITY if o in ops]
            if require_structural and not struct_ops:
                entry["reason"] = "no_supported_composition_operation"
            elif not ops:
                entry["reason"] = "no_verifiable_registered_operation"
            else:
                # A structural operation is more useful than a generic model mention.
                group = struct_ops[0] if struct_ops else ops[0]
                entry["method_group"] = group
                groups[group].append((item, entry))
                continue
        out.audit.append(entry)

    group_order = [name for name in PRIORITY if name in groups]
    group_order += sorted(name for name in groups if name not in group_order)
    seen_paper_method = set()
    used_counts: dict[str, int] = defaultdict(int)
    while len(out.records) < max_records:
        advanced = False
        for name in group_order:
            while groups[name] and used_counts[name] < max_per_method:
                item, entry = groups[name].pop(0)
                key = (str(item.get("paper_url") or item.get("paper_title")), name)
                if key in seen_paper_method:
                    entry["reason"] = "duplicate_paper_and_method"
                    out.audit.append(entry)
                    continue
                seen_paper_method.add(key)
                entry["status"] = "selected"
                out.records.append(item)
                out.audit.append(entry)
                used_counts[name] += 1
                advanced = True
                break
            if len(out.records) >= max_records:
                break
        if not advanced:
            break
    for name, pending in groups.items():
        for _, entry in pending:
            entry["reason"] = "method_quota_or_global_limit"
            out.audit.append(entry)
    return out
