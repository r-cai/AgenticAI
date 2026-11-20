"""Constrained *LLM-designed* evidence-backed compositions for Experiment D.

The LLM chooses a verified literature operation, constituent estimators, and
bounded hyperparameters. Deterministic Python instantiates the Pydantic
CompositionConfig; nothing from the LLM is executed as code. Unlike C's fixed
one-template-per-paper mechanism, D can propose and iterate varied models.

Fail closed: no eligible evidence, malformed choices, or failed provenance never
become silently repaired, falsely attributed experiments.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import CandidateConfig, GeneratorOutput
from rag.evidence_compiler import ENSEMBLES, FEATURE_OPS, _grounded_ops
from rag.llm_grounding import CAUTION, validate_llm_grounded_candidate

SUPPORTED_MODELS = (
    "logistic_regression", "random_forest", "svc", "gradient_boosting", "extra_trees"
)
SUPPORTED_STRUCTURES = ENSEMBLES | FEATURE_OPS


class DesignChoice(BaseModel):
    """Small structured action, deliberately much simpler than a nested pipeline."""

    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    operation: Literal["rfe", "select_k_best", "pca", "variance_threshold",
                       "polynomial_features", "voting", "stacking"]
    base_models: list[Literal["logistic_regression", "random_forest", "svc",
                              "gradient_boosting", "extra_trees"]] = Field(min_length=1, max_length=3)
    voting: Literal["soft", "hard"] = "soft"
    n_features: int = Field(default=10, ge=2, le=100)
    lr_c: float = Field(default=1.0, gt=0, le=100)
    rf_trees: int = Field(default=150, ge=50, le=500)
    rf_depth: int = Field(default=8, ge=2, le=30)
    hypothesis: str = Field(min_length=8, max_length=300)


class DesignOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choices: list[DesignChoice] = Field(min_length=1, max_length=3)


class GuidedGenerationError(ValueError):
    def __init__(self, message: str, rejections: list[dict]):
        super().__init__(message)
        self.rejections = list(rejections)


def allowed_methods(evidence: list[dict]) -> list[dict]:
    """Only source-grounded, affirmative, registered structural operations."""
    result = []
    for rec in evidence or []:
        if not rec.get("source_grounded"):
            continue
        title = str(rec.get("paper_title") or "")
        excerpt = str(rec.get("source_excerpt") or "")
        if CAUTION.search(title + " " + excerpt):
            continue
        ops, _ = _grounded_ops(rec)
        eligible = sorted(set(ops) & SUPPORTED_STRUCTURES)
        if not eligible:
            continue
        result.append({
            "evidence_id": str(rec["evidence_id"]),
            "paper_title": title[:135],
            "verbatim_excerpt": excerpt[:260],
            "permitted_operations": eligible,
        })
    return result


def _model(name: str, idea: DesignChoice, *, soft_vote: bool = False) -> dict:
    if name == "logistic_regression":
        return {"name": name, "hyperparameters": {"C": idea.lr_c,
                                                     "max_iter": 1500}}
    if name in {"random_forest", "extra_trees"}:
        return {"name": name, "hyperparameters": {"n_estimators": idea.rf_trees,
                                                     "max_depth": idea.rf_depth,
                                                     "min_samples_leaf": 2}}
    if name == "svc":
        return {"name": name, "hyperparameters": {"kernel": "rbf", "C": idea.lr_c,
                                                     "probability": soft_vote}}
    if name == "gradient_boosting":
        return {"name": name, "hyperparameters": {"n_estimators": idea.rf_trees,
                                                     "max_depth": min(5, idea.rf_depth),
                                                     "learning_rate": 0.1}}
    raise ValueError(f"Unsupported estimator: {name}")


def compile_choice(idea: DesignChoice, profile: dict, catalog: dict) -> CandidateConfig:
    """Compile the LLM's *decision*, without synthesizing or executing code."""
    if idea.evidence_id not in catalog:
        raise ValueError("unknown evidence_id")
    supported, _ = _grounded_ops(catalog[idea.evidence_id])
    if idea.operation not in set(supported) & SUPPORTED_STRUCTURES:
        raise ValueError("cited excerpt does not support requested structural method")
    if CAUTION.search(str(catalog[idea.evidence_id].get("paper_title", "")) + " " +
                      str(catalog[idea.evidence_id].get("source_excerpt", ""))):
        raise ValueError("cautionary evidence is not positive method support")
    n = int(profile.get("n_features", 0))
    if n < 3:
        raise ValueError("at least three input features required")
    if len(set(idea.base_models)) != len(idea.base_models):
        raise ValueError("duplicate base models")
    if idea.operation in ENSEMBLES and not 2 <= len(idea.base_models) <= 3:
        raise ValueError("voting/stacking require 2-3 distinct estimators")
    if idea.operation in FEATURE_OPS and len(idea.base_models) != 1:
        raise ValueError("feature-selection pipeline requires exactly one estimator")
    if idea.operation == "polynomial_features" and n > 20:
        raise ValueError("polynomial expansion not allowed for >20 features")
    if idea.operation in {"pca", "rfe", "select_k_best"} and idea.n_features >= n:
        raise ValueError("n_features must be less than the input feature count")

    prep = {"numeric_imputer": "median", "numeric_scaler": "none",
            "categorical_encoder": "onehot", "drop_columns": []}
    if idea.operation in ENSEMBLES:
        branches = []
        for index, name in enumerate(idea.base_models):
            # Scale only the branch that benefits from it.
            transforms = ([{"op": "standard_scaler", "params": {}}]
                          if name in {"logistic_regression", "svc"} else [])
            branches.append({"name": f"branch_{index+1}", "transforms": transforms,
                             "model": _model(name, idea, soft_vote=idea.operation == "voting" and idea.voting == "soft")})
        comp = {"kind": idea.operation, "branches": branches,
                "voting": idea.voting,
                "final_estimator": ({"name": "logistic_regression", "hyperparameters":
                                     {"C": idea.lr_c, "max_iter": 1500}}
                                    if idea.operation == "stacking" else None)}
    else:
        op = idea.operation
        transforms = []
        if op in {"rfe", "select_k_best", "pca"}:
            transforms.append({"op": "standard_scaler", "params": {}})
        if op == "rfe":
            transforms.append({"op": "rfe", "params": {
                "estimator": "logistic_regression", "n_features_to_select": idea.n_features}})
        elif op == "select_k_best":
            transforms.append({"op": "select_k_best", "params": {"k": idea.n_features}})
        elif op == "pca":
            transforms.append({"op": "pca", "params": {"n_components": idea.n_features}})
        elif op == "variance_threshold":
            transforms.append({"op": "variance_threshold", "params": {"threshold": 0.0}})
        elif op == "polynomial_features":
            transforms.append({"op": "polynomial_features", "params": {"degree": 2,
                                                                          "interaction_only": True}})
        comp = {"kind": "pipeline", "transforms": transforms,
                "model": _model(idea.base_models[0], idea)}

    candidate = CandidateConfig.model_validate({
        "rationale": idea.hypothesis,
        "preprocessing": prep,
        "composition": comp,
        "model": None,
        "evidence_ids": [idea.evidence_id],
        "research_basis": [idea.evidence_id],
    })
    validate_llm_grounded_candidate(candidate, catalog)
    return candidate


def generate_grounded(*, profile: dict, research_evidence: list[dict],
                      history=None, model="qwen3.5:4b", temperature=0.2,
                      candidates_per_iteration=3, validation_retries=2,
                      previous_batch_kinds=None, chat_fn=None, **kwargs) -> GeneratorOutput:
    """Qwen chooses evidence operation + model structure using a compact JSON DSL.

    Every accepted candidate goes through the same immutable Pydantic and
    source-operation validation as original D. No silent replacement or
    dummy evidence is ever introduced.
    """
    menu = allowed_methods(research_evidence)
    if not menu:
        raise GuidedGenerationError("No verified structural literature evidence", [])
    if chat_fn is None:
        from ollama import chat as chat_fn
    catalog = {r["evidence_id"]: r for r in research_evidence}
    accepted, seen, errors = [], set(), []
    for attempt in range(max(0, int(validation_retries)) + 1):
        missing = candidates_per_iteration - len(accepted)
        if missing <= 0:
            break
        concise_history = [{"model": h.get("model"), "cv_mean": h.get("cv_mean")}
                           for h in (history or [])[-6:] if isinstance(h, dict)]
        context = {
            "eligible_evidence": menu,
            "dataset": {k: profile.get(k) for k in ("n_rows", "n_features", "n_classes", "imbalance_ratio")},
            "recent_cv_history": concise_history,
            "previous_batch_kinds": previous_batch_kinds or [],
            "requested_choices": missing,
            "allowed_models": list(SUPPORTED_MODELS),
            "previous_rejections": [e.get("error") for e in errors[-4:]],
        }
        system = (
            "You are a scientific ML DESIGN PLANNER, not a Python programmer. "
            "Return a JSON object containing 'choices' only. Each choice must use "
            "the EXACT evidence_id and one of its permitted_operations. "
            "For voting/stacking provide 2 or 3 DIFFERENT base_models; for feature "
            "selection/PCA provide exactly 1 model. n_features must be smaller than "
            "dataset.n_features. All parameter fields are optional defaults. "
            "Propose diverse, untried, scientifically plausible choices. Do not "
            "invent a publication, an operation, or benchmark outcome. "
            "A paper's method support does not imply its hyperparameters are sourced. "
            "Preferred example if a paper allows rfe: "
            "{'evidence_id':'EXACT_ID','operation':'rfe','base_models':['logistic_regression'],"
            "'n_features':12,'hypothesis':'Test RFE then logistic regression for tabular classification.'}. "
            "Preferred example if a paper allows voting: "
            "{'evidence_id':'EXACT_ID','operation':'voting',"
            "'base_models':['logistic_regression','random_forest'],'voting':'soft',"
            "'hypothesis':'Test complementary linear and nonlinear decision boundaries.'}. "
            "Do not generate a complete pipeline schema or executable code."
        )
        try:
            response = chat_fn(
                model=model,
                messages=[{"role":"system", "content":system},
                          {"role":"user", "content":json.dumps(context, default=str)}],
                format=DesignOutput.model_json_schema(),
                think=False,
                options={"temperature":temperature, "num_ctx":4096, "num_predict":1600},
            )
            parsed = DesignOutput.model_validate_json(response.message.content)
            print(f"[guided-D] attempt={attempt+1} requested={missing} choices={len(parsed.choices)}")
        except Exception as exc:
            errors.append({"attempt":attempt+1,"candidate_index":None,
                           "error": f"response_error:{type(exc).__name__}:{exc}"})
            print(f"[guided-D] response rejected: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        for idx, idea in enumerate(parsed.choices):
            if len(accepted) >= candidates_per_iteration:
                break
            try:
                cand = compile_choice(idea, profile, catalog)
                fp = cand.fingerprint()
                if fp in seen:
                    raise ValueError("duplicate_choice_in_response")
                seen.add(fp)
                accepted.append(cand)
                print(f"[guided-D] accepted {idea.operation} ({idea.evidence_id}) ")
            except Exception as exc:
                errors.append({"attempt":attempt+1,"candidate_index":idx,
                               "error":f"{type(exc).__name__}:{exc}"})
                print(f"[guided-D] rejected choice {idx}: {type(exc).__name__}: {str(exc)[:150]}")
    if not accepted:
        raise GuidedGenerationError("Grounded design generated no valid proposals", errors)
    output = GeneratorOutput(reasoning="LLM-selected and validated evidence-guided composition choices",
                             candidates=accepted)
    output._validation_errors = errors
    return output
