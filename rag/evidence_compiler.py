"""Literature evidence -> bounded, executable Level-2 candidate compositions.

This is deliberately *not* an unrestricted code synthesis engine. Every
proposal is compiled from registered sklearn operators, with an auditable
mapping between a VERBATIM paper excerpt and the operator it motivates.

The paper's claim is NOT verified by this module: only provenance, modality
compatibility and operational support are checked. Results still require CV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agents.schemas import CandidateConfig, CompositionConfig, ModelConfig, PreprocessingConfig

# Only source-excerpt-based method mapping is allowed. Vague classifier words
# cannot license a more specific selector, sampler or novel algorithm.
METHOD_PATTERNS = {
    "rfe": r"\brecursive\s+feature\s+elimination\b|\bRFE\b",
    "select_k_best": r"\bfeature\s+selection\b|\bselected\s+features\b|\bfeature\s+subset\b",
    "pca": r"\bprincipal\s+component\s+analysis\b|\bPCA\b",
    "variance_threshold": r"\bvariance\s+threshold\b|\blow.variance\s+filter",
    "polynomial_features": r"\bpolynomial\s+features?\b|\binteraction\s+features?\b",
    "voting": r"\bmajority\s+voting\b|\bsoft\s+voting\b|\bvoting\s+classifier\b",
    "stacking": r"\bstacking\b|\bstacked\s+(?:generalization|ensemble|classifier)",
    "svc": r"\bsupport\s+vector\s+machine\b|\bSVM\b|\bSVC\b",
    "logistic_regression": r"\blogistic\s+regression\b|\bLR\s+classifier\b",
    "random_forest": r"\brandom\s+forest\b",
    "gradient_boosting": r"\bgradient\s+boosting\b",
    "extra_trees": r"\bextra\s+trees\b",
    "knn": r"\bk.nearest\s+neighbou?rs?\b|\bKNN\b",
    "standard_scaler": r"\bstandard(?:ization|isation|ized|ised|\s+scal(?:ing|er))\b",
    "minmax_scaler": r"\bmin.max\s+scal(?:ing|er)\b",
    "robust_scaler": r"\brobust\s+scal(?:ing|er)\b",
}

# MRI, WSI and ultrasound models are not transferable as *algorithms* to
# Wisconsin's 30 numeric FNA measurements; do not treat all 'cancer AI' as same.
IMAGE_MODALITY = re.compile(
    # Mammography/radiology as a *topic* must not invalidate published work
    # on tabular BI-RADS descriptors. Reject explicitly pixel-based methods.
    r"\b(?:magnetic\s+resonance|MRI|ultrasound|whole.slide|WSI|"
    r"histopatholog\w*|U.Net|nnU.Net|ResNet\d*|ConvNeXt|"
    r"image\s+encoders?|image\s+segmentation|image\s+classification|"
    r"convolutional\s+neural\s+networks?|CNN|pixel.level|"
    r"mammogram\s+images?|mammographic\s+images?|segmentation)\b", re.I)
UNSUPPORTED_DEPENDENCY = re.compile(r"\b(?:random\s+undersampling|SMOTE|synthetic\s+data\s+generation|particle\s+swarm)\b", re.I)
WRONG_TASK = re.compile(r"\b(?:uterine|leiomyoma|phyllodes|thyroid)\b", re.I)

MODEL_OPS = {"svc", "logistic_regression", "random_forest", "gradient_boosting", "extra_trees", "knn"}
FEATURE_OPS = {"rfe", "select_k_best", "pca", "variance_threshold", "polynomial_features"}
ENSEMBLES = {"voting", "stacking"}
SCALERS = {"standard_scaler", "minmax_scaler", "robust_scaler"}


@dataclass
class TranslationResult:
    candidates: list[CandidateConfig] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    mapping: dict[str, dict[str, Any]] = field(default_factory=dict)


def _numeric_profile(profile: dict) -> bool:
    cols = list((profile.get("columns") or {}).values())
    if not cols:
        return False  # Cannot establish compatibility; fail closed.
    return sum(c.get("type") == "numeric" for c in cols if isinstance(c, dict)) >= max(1, int(len(cols) * 0.8))


def _feature_count(profile: dict) -> int:
    return max(1, int(profile.get("n_features") or len(profile.get("columns") or {}) or 1))


def _id_columns(profile: dict) -> list[str]:
    return [str(c) for c in (profile.get("columns") or {})
            if re.fullmatch(r"id|.*_id|patient[_ ]?id|index|record[_ ]?no", str(c), re.I)]


def _grounded_ops(evidence: dict) -> tuple[list[str], list[str]]:
    excerpt = str(evidence.get("source_excerpt") or "")
    declared = set(evidence.get("suggested_operations") or [])
    supported = []
    warnings = []
    for name, pattern in METHOD_PATTERNS.items():
        if re.search(pattern, excerpt, re.I):
            supported.append(name)
    # A generic feature-selection claim does not establish the use of RFE.
    if "rfe" in supported and "select_k_best" in supported:
        supported.remove("select_k_best")
    # Claim keywords are hints, not provenance. Disagreeing hints are reported.
    if declared and not (declared & set(supported)):
        warnings.append("suggested_operations_unsupported_by_excerpt")
    return supported, warnings


def _make_candidate(evidence: dict, operations: list[str], profile: dict):
    """Construct a Level-2 architecture, distinguishing documented method from design choices."""
    if not _numeric_profile(profile):
        raise ValueError("numeric_tabular_profile_required_for_this_compiler")
    n = _feature_count(profile) - len(_id_columns(profile))
    n = max(2, n)
    prep = PreprocessingConfig(numeric_imputer="median", numeric_scaler="none",
                               drop_columns=_id_columns(profile))
    specific = set(operations)
    design_choices = ["median imputation is an engineering choice, not attributed to source"]
    def mdl(name):
        if name == "logistic_regression":
            return ModelConfig(name=name, hyperparameters={"max_iter": 1500, "class_weight": "balanced"})
        if name == "random_forest":
            return ModelConfig(name=name, hyperparameters={"n_estimators": 200, "min_samples_leaf": 2, "class_weight": "balanced"})
        if name == "svc":
            return ModelConfig(name=name, hyperparameters={"C": 1.0, "kernel": "rbf", "class_weight": "balanced"})
        return ModelConfig(name=name, hyperparameters={})

    if "voting" in specific or "stacking" in specific:
        op = "voting" if "voting" in specific else "stacking"
        from agents.schemas import BranchConfig, TransformConfig
        # A source may recommend the voting *principle*, not these exact base models.
        b1 = BranchConfig(name="linear", transforms=[TransformConfig(op="standard_scaler")], model=mdl("logistic_regression"))
        b2 = BranchConfig(name="trees", transforms=[], model=mdl("random_forest"))
        spec = CompositionConfig(kind=op, branches=[b1, b2],
                                 voting="hard" if op == "voting" else "soft")
        design_choices.append("base learners are experiment design choices, not claimed paper settings")
        return CandidateConfig(rationale=f"Test literature-mentioned {op} with two diverse registered learners.",
                               preprocessing=prep, composition=spec,
                               research_basis=[evidence["paper_title"]], evidence_ids=[evidence["evidence_id"]]), op, design_choices

    from agents.schemas import TransformConfig
    transforms = []
    if "rfe" in specific:
        transforms.append(TransformConfig(op="standard_scaler"))
        transforms.append(TransformConfig(op="rfe", params={"n_features_to_select": min(12, n), "step": 1}))
        op = "rfe"
        design_choices.append("RFE uses logistic regression as selector; subset size is exploratory")
    elif "select_k_best" in specific:
        transforms.append(TransformConfig(op="select_k_best", params={"k": min(15, n)}))
        op = "select_k_best"
        design_choices.append("SelectKBest is an implementation of generic feature selection, not necessarily paper's selector")
    elif "pca" in specific:
        transforms += [TransformConfig(op="standard_scaler"),
                       TransformConfig(op="pca", params={"n_components": min(10, n)})]
        op = "pca"
        design_choices.append("PCA component count is a design choice")
    elif "variance_threshold" in specific:
        transforms.append(TransformConfig(op="variance_threshold"))
        op = "variance_threshold"
    elif "polynomial_features" in specific:
        transforms.append(TransformConfig(op="polynomial_features", params={"degree": 2}))
        op = "polynomial_features"
    else:
        # A paper mentioning just a standard model cannot justify a new
        # composition. Let the ordinary LLM path handle model-only experiments.
        raise ValueError("no_supported_feature_or_ensemble_method")

    if op in {"rfe", "select_k_best", "variance_threshold", "polynomial_features"}:
        # A small number of classifier choices, from explicit evidence if present.
        classifier = next((x for x in ("logistic_regression", "svc", "random_forest") if x in specific), "logistic_regression")
        if classifier in {"logistic_regression", "svc"} and not any(t.op == "standard_scaler" for t in transforms):
            transforms.insert(0, TransformConfig(op="standard_scaler"))
        design_choices.append(f"classifier {classifier} is chosen for test; paper may not specify it")
    else:
        classifier = next((x for x in ("logistic_regression", "svc", "random_forest") if x in specific), "logistic_regression")
        design_choices.append(f"classifier {classifier} is chosen for test; paper may not specify it")
    spec = CompositionConfig(kind="pipeline", transforms=transforms, model=mdl(classifier))
    cand = CandidateConfig(rationale=f"Test source-grounded {op} as a registered Level-2 pipeline; examine CV benefit.",
                           preprocessing=prep, composition=spec,
                           research_basis=[evidence["paper_title"]], evidence_ids=[evidence["evidence_id"]])
    return cand, op, design_choices


def translate_evidence(evidence_records: list[dict], profile: dict, *, max_candidates: int = 1,
                       tried: set[str] | None = None, allow_keyword: bool = False) -> TranslationResult:
    """Translate grounded evidence to executable candidate objects and an audit log.

    Uses only the training profile. No outcomes, holdout data, or test metrics
    participate in deciding mappings. `tried` prevents duplicated executions.
    """
    result = TranslationResult()
    tried = set(tried or ())
    local_seen = set()
    if max_candidates <= 0:
        return result
    for ev in evidence_records or []:
        eid = ev.get("evidence_id")
        base = {"evidence_id": eid, "paper_title": ev.get("paper_title"),
                "status": "rejected", "reason": None, "matched_operations": []}
        if not eid or not ev.get("paper_title") or not ev.get("source_excerpt"):
            base["reason"] = "incomplete_evidence_record"
        elif not ev.get("source_grounded"):
            base["reason"] = "not_source_grounded"
        elif ev.get("extraction_method") != "llm_verified_excerpt" and not allow_keyword:
            base["reason"] = "unverified_extraction_method"
        elif IMAGE_MODALITY.search(str(ev.get("paper_title")) + " " + str(ev.get("source_excerpt"))) and _numeric_profile(profile):
            base["reason"] = "image_method_incompatible_with_numeric_tabular_data"
        elif UNSUPPORTED_DEPENDENCY.search(str(ev.get("source_excerpt"))):
            base["reason"] = "excerpt_requires_unsupported_method"
        elif WRONG_TASK.search(str(ev.get("paper_title"))) and "breast" in str(profile.get("target", "")).lower():
            base["reason"] = "different_disease_or_prediction_target"
        else:
            ops, hints = _grounded_ops(ev)
            base["matched_operations"] = ops
            base["extraction_warnings"] = hints
            try:
                cand, demonstrated_op, design_choices = _make_candidate(ev, ops, profile)
                fingerprint = cand.fingerprint()
                if fingerprint in tried or fingerprint in local_seen:
                    raise ValueError("duplicate_executable_architecture")
                base.update({"status": "accepted", "reason": None,
                             "implemented_operation": demonstrated_op,
                             "implementation_choices": design_choices,
                             "candidate_fingerprint": fingerprint})
                local_seen.add(fingerprint)
                result.candidates.append(cand)
                result.mapping[fingerprint] = {"evidence_id": eid,
                                               "implemented_operation": demonstrated_op,
                                               "implementation_choices": design_choices}
            except (ValueError, TypeError) as exc:
                base["reason"] = str(exc)
        result.audit.append(base)
        if len(result.candidates) >= max_candidates:
            break
    return result


def validate_candidate_citations(candidate: CandidateConfig, evidence_catalog: dict[str, dict]) -> list[str]:
    """Reject fake citations, unimplemented evidence claims, and missing linkage."""
    if not candidate.evidence_ids:
        return []
    if candidate.composition is None:
        return ["evidence-linked candidate must be an actual Level-2 composition"]
    cc = candidate.composition
    actual = {t.op for t in cc.transforms}
    for b in cc.branches:
        actual |= {t.op for t in b.transforms}
        if b.model is not None:
            actual.add(b.model.name)
    if cc.model is not None:
        actual.add(cc.model.name)
    actual.add(cc.kind)
    errors = []
    for eid in candidate.evidence_ids:
        ev = evidence_catalog.get(eid)
        if not ev:
            errors.append(f"unrecognized_evidence_id:{eid}")
            continue
        if not ev.get("source_grounded"):
            errors.append(f"ungrounded_evidence:{eid}")
            continue
        if IMAGE_MODALITY.search(str(ev.get("paper_title")) + " " + str(ev.get("source_excerpt"))):
            errors.append(f"image_method_requires_separate_modality:{eid}")
            continue
        if UNSUPPORTED_DEPENDENCY.search(str(ev.get("source_excerpt"))):
            errors.append(f"paper_requires_unsupported_method:{eid}")
            continue
        if ev.get("extraction_method") != "llm_verified_excerpt":
            errors.append(f"unverified_extraction_method:{eid}")
            continue
        supported, _ = _grounded_ops(ev)
        if not (set(supported) & actual):
            errors.append(f"source_does_not_support_used_operations:{eid}")
    return errors
