from typing import Any, Literal
import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ml.registry import ENCODERS, MODEL_NAMES, NUMERIC_IMPUTERS, SCALERS

#Generator Schemas
class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numeric_imputer: Literal[NUMERIC_IMPUTERS] = "median"
    numeric_scaler: Literal[SCALERS] = "none"
    categorical_encoder: Literal[ENCODERS] = "onehot"
    drop_columns: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal[MODEL_NAMES]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TransformConfig(BaseModel):
    """One registered feature transformation; never arbitrary executable code."""

    model_config = ConfigDict(extra="forbid")
    op: Literal[
        "standard_scaler", "minmax_scaler", "robust_scaler",
        "select_k_best", "rfe", "pca", "polynomial_features",
        "variance_threshold",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class BranchConfig(BaseModel):
    """Parallel feature branch, or a model branch for an ensemble."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]*$")
    transforms: list[TransformConfig] = Field(default_factory=list, max_length=4)
    model: ModelConfig | None = None


class CompositionConfig(BaseModel):
    """A deliberately bounded, declarative architecture search language."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["pipeline", "feature_union", "stacking", "voting"]
    transforms: list[TransformConfig] = Field(default_factory=list, max_length=4)
    branches: list[BranchConfig] = Field(default_factory=list, max_length=3)
    model: ModelConfig | None = None
    final_estimator: ModelConfig | None = None
    voting: Literal["soft", "hard"] = "soft"
    passthrough: bool = False

    @model_validator(mode="after")
    def check_structure(self):
        names = [b.name for b in self.branches]
        if len(names) != len(set(names)):
            raise ValueError("Branch names must be unique")
        if self.kind == "pipeline":
            if self.model is None or self.branches or self.final_estimator:
                raise ValueError("pipeline needs model, no branches/final_estimator")
        elif self.kind == "feature_union":
            if self.model is None or len(self.branches) < 2 or self.final_estimator:
                raise ValueError("feature_union requires a model and 2-3 branches")
            if any(b.model is not None for b in self.branches):
                raise ValueError("feature_union branches must contain transforms only")
        else:
            if self.model is not None or len(self.branches) < 2:
                raise ValueError("ensembles require 2-3 model branches")
            if any(b.model is None for b in self.branches):
                raise ValueError("ensemble branches each require a model")
            if self.kind == "voting" and self.final_estimator is not None:
                raise ValueError("voting must not specify a final_estimator")
        return self


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str
    preprocessing: PreprocessingConfig
    model: ModelConfig | None = None
    composition: CompositionConfig | None = None
    research_basis: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def exactly_one_implementation(self):
        if (self.model is None) == (self.composition is None):
            raise ValueError("Specify exactly one of model or composition")
        return self

    @property
    def display_name(self) -> str:
        return self.model.name if self.model is not None else f"composition:{self.composition.kind}"

    def fingerprint(self) -> str:
        # Exclude rationale and citations: only executable configuration defines duplication.
        config = {
            "preprocessing": self.preprocessing.model_dump(),
            "model": self.model.model_dump() if self.model else None,
            "composition": self.composition.model_dump() if self.composition else None,
        }
        canonical = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Compatibility with orchestration/controller.py
CandidateConfig = Candidate


class GeneratorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    candidates: list[Candidate] = Field(min_length=1, max_length=3)
    # Runtime-only diagnostics, never part of the LLM JSON schema.
    _validation_errors: list[dict[str, Any]] = PrivateAttr(default_factory=list)

class CandidateResult(BaseModel):
    candidate_id: str
    iteration: int
    config: Candidate
    model_name: str | None = None
    mean_cv_score: float | None = None
    std_cv_score: float | None = None
    fold_scores: list[float] = Field(default_factory=list)
    scoring: str | None = None
    cv_fit_time_mean_seconds: float | None = None
    cv_wall_time_seconds: float | None = None
    n_evaluations: int | None = None
    train_cv_gap: float | None = None
    convergence_warning_count: int | None = None
    convergence_warning: str | None = None 
    origin: str | None = None
    evidence_mapping: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


#Critic Schemas
class Issue(BaseModel):
    type: Literal[
        "overfitting",
        "underfitting",
        "class_imbalance",
        "high_variance",
        "weak_preprocessing",
        "poor_hyperparameters",
        "data_quality",
        "other",
    ]
    detail: str


class Recommendation(BaseModel):
    area: Literal[
        "hyperparameters",
        "model_choice",
        "preprocessing",
        "feature_selection",
        "evaluation",
        "other",
    ]
    suggestion: str


class CriticOutput(BaseModel):
    summary: str
    issues: list[Issue] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    likely_converged: bool


GENERATOR_EXAMPLE = {
    "reasoning": (
        "Baseline is weak on minority class. "
        "Try a balanced forest and a scaled logistic regression."
    ),
    "candidates": [
        {
            "rationale": (
                "Trees handle mixed types and class_weight "
                "helps with imbalance."
            ),
            "preprocessing": {
                "numeric_imputer": "median",
                "numeric_scaler": "none",
                "categorical_encoder": "onehot",
                "drop_columns": ["passenger_id"],
            },
            "research_basis": [],
            "model": {
                "name": "random_forest",
                "hyperparameters": {
                    "n_estimators": 300,
                    "max_depth": 10,
                    "class_weight": "balanced",
                },
            },
        }
    ],
}


CRITIC_EXAMPLE = {
    "summary": (
        "Random forest overfits: train 0.99 vs CV 0.81. "
        "Logistic regression is stable but weaker."
    ),
    "issues": [
        {
            "type": "overfitting",
            "detail": "train_cv_gap of 0.18 for iter01_c1.",
        },
    ],
    "recommendations": [
        {
            "area": "hyperparameters",
            "suggestion": (
                "Limit max_depth to 5-8 and raise min_samples_leaf."
            ),
        },
        {
            "area": "model_choice",
            "suggestion": (
                "Try hist_gradient_boosting with early stopping."
            ),
        },
    ],
    "likely_converged": False,
}
