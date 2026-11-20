"""Deterministic Level-2 model composition from a bounded declarative schema.

The LLM specifies *which* registered operators to use. This module never
imports, evaluates or executes source text supplied by an LLM or a paper.
"""

import math
import numbers

from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   PolynomialFeatures, RobustScaler, StandardScaler)

from ml.registry import MODEL_REGISTRY


MAX_OPERATORS = 12
MAX_EXPANDED_FEATURES = 600


def _plain(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _validate_keys(values, allowed, label):
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {label} parameters: {sorted(unknown)}")


def _scalar_values(params, label):
    if not isinstance(params, dict):
        raise ValueError(f"{label} params must be a JSON object")
    for key, value in params.items():
        if isinstance(value, (list, dict, tuple, set)):
            raise ValueError(f"{label}.{key} must be a scalar, not {type(value).__name__}")
        if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
            raise ValueError(f"{label}.{key} must be finite")


def _model(spec, seed):
    spec = _plain(spec)
    name = spec["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported registered estimator: {name}")
    params = dict(spec.get("hyperparameters") or {})
    _scalar_values(params, name)
    cls = MODEL_REGISTRY[name]
    defaults = cls().get_params(deep=False)
    _validate_keys(params, defaults, name)
    if "random_state" in defaults:
        # A fixed experiment seed, not an LLM-selected source of variation.
        params["random_state"] = seed
    for positive in ("C", "n_estimators", "max_iter", "n_neighbors"):
        if positive in params and (not isinstance(params[positive], numbers.Real)
                                   or isinstance(params[positive], bool)
                                   or params[positive] <= 0):
            raise ValueError(f"{name}.{positive} must be positive")
    if params.get("n_estimators", 0) > 1000:
        raise ValueError("n_estimators exceeds the 1000-tree resource limit")
    if params.get("n_jobs", 1) not in (None, 1):
        raise ValueError("Model n_jobs must be 1 to bound CPU consumption")
    return cls(**params)


def _transform(spec, n_features, seed):
    spec = _plain(spec)
    op = spec["op"]
    p = dict(spec.get("params") or {})
    _scalar_values(p, op)
    allowed = {
        "standard_scaler": {"with_mean", "with_std"},
        "minmax_scaler": set(),
        "robust_scaler": set(),
        "select_k_best": {"k"},
        "rfe": {"n_features_to_select", "step", "estimator"},
        "pca": {"n_components"},
        "polynomial_features": {"degree", "interaction_only"},
        "variance_threshold": {"threshold"},
    }
    if op not in allowed:
        raise ValueError(f"Unsupported transformation: {op}")
    _validate_keys(p, allowed[op], op)
    if op in {"standard_scaler", "minmax_scaler", "robust_scaler"}:
        cls = {"standard_scaler": StandardScaler, "minmax_scaler": MinMaxScaler,
               "robust_scaler": RobustScaler}[op]
        return cls(**p)
    if op == "select_k_best":
        k = p.get("k", min(10, n_features))
        if type(k) is not int or not 1 <= k <= n_features:
            raise ValueError("select_k_best.k must be between 1 and input feature count")
        return SelectKBest(f_classif, k=k)
    if op == "rfe":
        k = p.get("n_features_to_select", max(1, n_features // 2))
        if type(k) is not int or not 1 <= k <= n_features:
            raise ValueError("rfe.n_features_to_select out of range")
        step = p.get("step", 1)
        if not isinstance(step, numbers.Real) or isinstance(step, bool) or step <= 0:
            raise ValueError("rfe.step must be positive")
        estimator_name = p.get("estimator", "logistic_regression")
        if estimator_name not in {"logistic_regression", "random_forest"}:
            raise ValueError("rfe.estimator must be logistic_regression or random_forest")
        return RFE(_model({"name": estimator_name, "hyperparameters": {}}, seed),
                   n_features_to_select=k, step=step)
    if op == "pca":
        k = p.get("n_components", min(10, n_features))
        if type(k) is not int or not 1 <= k <= n_features:
            raise ValueError("pca.n_components out of range")
        return PCA(n_components=k, random_state=seed)
    if op == "polynomial_features":
        degree = p.get("degree", 2)
        if type(degree) is not int or degree not in (2, 3):
            raise ValueError("polynomial degree must be 2 or 3")
        count = math.comb(n_features + degree, degree)
        if count > MAX_EXPANDED_FEATURES:
            raise ValueError(f"Polynomial expansion exceeds {MAX_EXPANDED_FEATURES} features")
        return PolynomialFeatures(degree=degree, include_bias=False,
                                  interaction_only=p.get("interaction_only", False))
    threshold = p.get("threshold", 0.0)
    if not isinstance(threshold, numbers.Real) or threshold < 0:
        raise ValueError("variance_threshold.threshold must be >= 0")
    return VarianceThreshold(threshold=threshold)


def _transforms(specs, n_features, seed, prefix):
    steps = []
    current = n_features
    for i, item in enumerate(specs):
        spec = _plain(item)
        op = spec["op"]
        transform = _transform(spec, current, seed)
        steps.append((f"{prefix}{i:02d}_{op}", transform))
        if op == "select_k_best":
            current = spec.get("params", {}).get("k", min(10, current))
        elif op == "rfe":
            current = spec.get("params", {}).get("n_features_to_select", max(1, current // 2))
        elif op == "pca":
            current = spec.get("params", {}).get("n_components", min(10, current))
        elif op == "polynomial_features":
            degree = spec.get("params", {}).get("degree", 2)
            current = math.comb(current + degree, degree) - 1
    return steps, current


def _branch_pipeline(branch, n_features, seed):
    branch = _plain(branch)
    steps, output_dim = _transforms(branch.get("transforms", []), n_features, seed, "b")
    if branch.get("model") is not None:
        steps.append(("estimator", _model(branch["model"], seed)))
    else:
        # FeatureUnion passes this branch's feature representation downstream.
        if not steps:
            return FunctionTransformer(validate=False), output_dim
    return Pipeline(steps), output_dim


def compile_composition(composition, base_preprocessor, X_train, cfg):
    """Compile a validated CompositionConfig to a fitted-later sklearn Pipeline.

    The returned outer pipeline always exposes a final step named 'model',
    matching the existing evaluator's interface. X_train is used solely for
    input-column validation; no transformation is fitted outside the CV folds.
    """
    spec = _plain(composition)
    kind = spec["kind"]
    branches = spec.get("branches", [])
    global_steps = spec.get("transforms", [])
    n_features = X_train.shape[1]
    if n_features < 1:
        raise ValueError("No input features")
    complexity = len(global_steps) + sum(len(b.get("transforms", [])) + 1 for b in branches)
    if complexity > MAX_OPERATORS:
        raise ValueError(f"Composition exceeds the {MAX_OPERATORS}-operator limit")
    seed = cfg.get("seed", 42)
    if kind == "pipeline":
        transforms, _ = _transforms(global_steps, n_features, seed, "g")
        model = Pipeline(transforms + [("estimator", _model(spec["model"], seed))])
    elif kind == "feature_union":
        built = [(b["name"], *_branch_pipeline(b, n_features, seed)) for b in branches]
        parts = [(name, pipe) for name, pipe, _ in built]
        union_dim = sum(dimension for _, _, dimension in built)
        transforms, _ = _transforms(global_steps, union_dim, seed, "g")
        model = Pipeline([("feature_union", FeatureUnion(parts))]
                         + transforms + [("estimator", _model(spec["model"], seed))])
    elif kind in ("stacking", "voting"):
        transforms, shared_dim = _transforms(global_steps, n_features, seed, "g")
        estimators = [(b["name"], _branch_pipeline(b, shared_dim, seed)[0]) for b in branches]
        if kind == "stacking":
            final_spec = spec.get("final_estimator") or {
                "name": "logistic_regression", "hyperparameters": {"max_iter": 1000}}
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=_model(final_spec, seed),
                cv=3, passthrough=spec.get("passthrough", False), n_jobs=1)
        else:
            voting = spec.get("voting", "soft")
            if voting == "soft":
                for b in branches:
                    m = b["model"]
                    if m["name"] == "svc" and not m.get("hyperparameters", {}).get("probability", False):
                        raise ValueError("Soft voting with SVC requires probability=True")
            ensemble = VotingClassifier(estimators=estimators, voting=voting, n_jobs=1)
        model = Pipeline(transforms + [("estimator", ensemble)])
    else:
        raise ValueError(f"Unknown composition kind: {kind}")

    return Pipeline([("preprocessing", base_preprocessor), ("model", model)])
