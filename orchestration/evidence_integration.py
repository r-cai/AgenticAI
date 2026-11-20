"""Build a fair, bounded candidate batch from evidence + the existing LLM.

Candidate count is NEVER increased by enabling literature compilation. A
reserved evidence slot REPLACES an LLM proposal, rather than adding one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agents.schemas import CandidateConfig
from ml.trainer import build_pipeline
from rag.evidence_compiler import translate_evidence, validate_candidate_citations
from rag.llm_grounding import validate_llm_grounded_candidate


@dataclass
class ProposalBatch:
    candidates: list[CandidateConfig] = field(default_factory=list)
    origins: dict[str, str] = field(default_factory=dict)
    mapping: dict[str, Any] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)
    reasoning: str = ""
    generator_error: str | None = None

    def dump(self):
        return {"reasoning": self.reasoning,
                "candidates": [c.model_dump() for c in self.candidates],
                "candidate_origins": self.origins,
                "candidate_evidence_mapping": self.mapping,
                "generator_error": self.generator_error}


def generate_batch(*, evidence: list[dict], evidence_catalog: dict, profile: dict,
                   train_df, target: str, cfg: dict, tried: set[str],
                   generate_fn: Callable, generator_kwargs: dict) -> ProposalBatch:
    """Compile evidence first; fill the remaining unchanged budget with the LLM.

    A failed/missing LLM does not prevent valid evidence candidates from running.
    Invalid proposed pipelines are logged and rejected before CV, not rewritten.
    """
    budget = int(cfg["loop"].get("candidates_per_iteration", 3))
    if budget < 1 or budget > 3:
        raise ValueError("candidates_per_iteration must be 1..3")
    result = ProposalBatch()
    seen = set(tried)
    grounded_cfg = cfg.get("llm_evidence_design", {})
    grounded_mode = bool(grounded_cfg.get("enabled", False))
    if grounded_mode:
        if not cfg.get("rag", {}).get("enabled") or not cfg.get("evidence", {}).get("enabled"):
            raise ValueError("D requires both RAG and evidence extraction")
        if cfg.get("evidence_composition", {}).get("enabled", False):
            raise ValueError("D must disable the template-based evidence compiler (C)")
        if not evidence_catalog:
            result.generator_error = "No extracted evidence available for strict D proposals"
            result.audit.append({"origin": "llm_evidence_design", "status": "rejected",
                                 "reason": "no_evidence_available"})
            return result
    proposal_cfg = cfg.get("evidence_composition", {})
    enable = (bool(cfg.get("rag", {}).get("enabled")) and
              bool(cfg.get("evidence", {}).get("enabled")) and
              bool(proposal_cfg.get("enabled", False)))
    slots = max(0, min(budget, int(proposal_cfg.get("reserve_slots", 1)))) if enable else 0

    if slots:
        trans = translate_evidence(evidence, profile, max_candidates=slots,
                                   tried=tried, allow_keyword=bool(proposal_cfg.get("allow_keyword", False)))
        result.audit.extend(trans.audit)
        for cand in trans.candidates:
            if len(result.candidates) >= slots:
                break
            fp = cand.fingerprint()
            try:
                build_pipeline(cand, train_df.drop(columns=[target]), cfg)
            except Exception as exc:
                result.audit.append({"evidence_id": cand.evidence_ids[0], "status": "rejected",
                                     "reason": f"pipeline_preflight_failed:{type(exc).__name__}:{exc}"})
                continue
            if fp in seen:
                continue
            result.candidates.append(cand)
            result.origins[fp] = "evidence_compiler"
            result.mapping[fp] = trans.mapping.get(fp, {})
            seen.add(fp)

    # Recover unused candidate slots after rejected or duplicate LLM proposals.
    # We NEVER exceed the original CV budget. Extra generation calls are counted
    # separately; a failure to fill is reported rather than hidden.
    max_rounds = max(1, int(cfg.get("loop", {}).get("max_batch_rounds", 1)))
    kind_limit = int(cfg.get("evidence_selection", {}).get("max_candidates_per_kind", 0))
    deferred: list[tuple] = []
    batch_kinds = [c.composition.kind if c.composition else "standalone" for c in result.candidates]
    generator_calls = 0
    for round_index in range(max_rounds):
        remaining = budget - len(result.candidates)
        if remaining <= 0:
            break
        generator_calls += 1
        try:
            call_kwargs = {**generator_kwargs, "candidates_per_iteration": remaining}
            # Optional constraint context helps Qwen avoid previous rejected or
            # duplicated proposals; mocks using **kwargs still work unchanged.
            call_kwargs["previous_batch_kinds"] = list(batch_kinds)
            call_kwargs["excluded_candidate_fingerprints"] = sorted(seen)
            if grounded_mode:
                call_kwargs.update({"evidence_grounded": True,
                                    "grounded_strategy": str(grounded_cfg.get("strategy", "guided")),
                                    "validation_retries": int(grounded_cfg.get("validation_retries", 2))})
            out = generate_fn(**call_kwargs)
            result.reasoning = (result.reasoning + " " + str(out.reasoning or "")).strip()
            for warning in getattr(out, "_validation_errors", []):
                result.audit.append({"origin": "llm", "status": "rejected",
                                     "reason": "generator_schema_validation",
                                     "batch_round": round_index + 1, **warning})
            for candidate in out.candidates:
                if len(result.candidates) >= budget:
                    break
                fp = candidate.fingerprint()
                if fp in seen:
                    result.audit.append({"origin": "llm", "status": "rejected",
                                         "reason": "duplicate_executable_architecture", "fingerprint": fp})
                    continue
                errors = validate_candidate_citations(candidate, evidence_catalog)
                if errors:
                    result.audit.append({"origin": "llm", "status": "rejected",
                                         "reason": ";".join(errors), "fingerprint": fp})
                    continue
                try:
                    build_pipeline(candidate, train_df.drop(columns=[target]), cfg)
                except Exception as exc:
                    result.audit.append({"origin": "llm", "status": "rejected",
                                         "reason": f"pipeline_preflight_failed:{type(exc).__name__}:{exc}",
                                         "fingerprint": fp})
                    continue
                mapping = {}
                if grounded_mode:
                    try:
                        mapping = validate_llm_grounded_candidate(candidate, evidence_catalog)
                    except ValueError as exc:
                        result.audit.append({"origin": "llm_evidence_design", "status": "rejected",
                                             "reason": str(exc), "fingerprint": fp})
                        continue
                kind = candidate.composition.kind if candidate.composition else "standalone"
                if kind_limit > 0 and batch_kinds.count(kind) >= kind_limit:
                    # Preserve as a valid backup but ask for different designs.
                    deferred.append((candidate, fp, mapping, kind))
                    seen.add(fp)
                    result.audit.append({"origin": "llm", "status": "deferred",
                                         "reason": "architecture_kind_diversity", "kind": kind,
                                         "fingerprint": fp})
                    continue
                result.candidates.append(candidate)
                result.origins[fp] = "llm_evidence_design" if grounded_mode else "llm"
                if grounded_mode:
                    result.mapping[fp] = mapping
                    result.audit.append({"origin": "llm_evidence_design", "status": "accepted",
                                         "candidate_fingerprint": fp,
                                         "evidence_ids": candidate.evidence_ids})
                seen.add(fp)
                batch_kinds.append(kind)
        except Exception as exc:
            result.generator_error = f"{type(exc).__name__}: {exc}"
            for warning in getattr(exc, "rejections", []):
                result.audit.append({"origin": "llm", "status": "rejected",
                                     "reason": "generator_schema_validation", **warning})
    # Fill using fully validated deferred candidates rather than waste compute
    # slots. Their architecture-kind overlap remains visible in the audit.
    for candidate, fp, mapping, kind in deferred:
        if len(result.candidates) >= budget:
            break
        result.candidates.append(candidate)
        result.origins[fp] = "llm_evidence_design" if grounded_mode else "llm"
        if grounded_mode:
            result.mapping[fp] = mapping
        result.audit.append({"origin": "llm", "status": "accepted",
                             "reason": "diversity_fallback", "fingerprint": fp, "kind": kind})
    result.audit.append({"origin": "batch_budget", "status": "summary",
                         "requested": budget, "proposed_valid": len(result.candidates),
                         "unfilled": budget - len(result.candidates),
                         "llm_calls": generator_calls})
    return result
