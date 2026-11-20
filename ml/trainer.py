import inspect

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from ml.registry import ENCODERS, MODEL_REGISTRY, NUMERIC_IMPUTERS, SCALER_REGISTRY
from ml.composer import compile_composition

def _as_dict(candidate):
    """Accept a Pydantic CandidateConfig or a plain configuration dict."""
    if isinstance(candidate, dict):
        return candidate
    if callable(getattr(candidate, "model_dump", None)):
        return candidate.model_dump()
    raise TypeError("candidate must be a dict or Pydantic model")

def _choice(value, allowed, field):
    if value not in allowed:
        raise ValueError(f"Invalid {field}: {value!r}. Allowed: {sorted(allowed)}")
    return value

def validate_params(model_name, params):
    # Normalize common LLM boolean mistakes
    for key in ("dual", "bootstrap", "oob_score", "fit_intercept"):
        if key in params and isinstance(params[key], str):
            value = params[key].lower()
            if value in ("true", "yes", "false", "no"):
                params[key] = value in ("true", "yes")

    # Reject incompatible parameter combinations
    if model_name in ("random_forest", "extra_trees"):
        if params.get("oob_score") and not params.get("bootstrap", False):
            raise ValueError("oob_score=True requires bootstrap=True")

    if model_name == "logistic_regression":
        if params.get("dual") and params.get("solver", "lbfgs") != "liblinear":
            raise ValueError("dual=True requires solver='liblinear'")

    return params

def build_pipeline(candidate, X_train, cfg):
    """Build an unfitted pipeline using only training-set column information.

    Optional preprocessing keys: numeric_imputer, numeric_scaler,
    categorical_encoder, drop_columns, categorical_columns.
    """
    candidate = _as_dict(candidate)
    
    prep = candidate.get("preprocessing", {})
    composition = candidate.get("composition")
    model_spec = candidate.get("model")
    model_name = (
        composition["kind"] if composition is not None else
        _choice(model_spec["name"], MODEL_REGISTRY, "model name")
    )
    imputer = _choice(prep.get("numeric_imputer", "median"), NUMERIC_IMPUTERS, "numeric_imputer")
    scaler_name = _choice(prep.get("numeric_scaler", "none"), SCALER_REGISTRY, "numeric_scaler")
    encoder_name = _choice(prep.get("categorical_encoder", "onehot"), ENCODERS, "categorical_encoder")

    drops = prep.get("drop_columns", [])
    categorical_override = prep.get("categorical_columns", [])
    if not isinstance(drops, list) or not isinstance(categorical_override, list):
        raise ValueError("drop_columns and categorical_columns must be lists")
    unknown = (set(drops) | set(categorical_override)) - set(X_train.columns)
    if unknown:
        raise ValueError(f"Unknown feature columns: {sorted(unknown)}")
    if set(drops) & set(categorical_override):
        raise ValueError("A dropped column cannot also be categorical")

    columns = [c for c in X_train.columns if c not in drops]
    if not columns:
        raise ValueError("Cannot drop every feature column")
    categorical = [c for c in columns if c in categorical_override or not
                   (X_train[c].dtype.kind in "biuf")]
    numeric = [c for c in columns if c not in categorical]

    numeric_steps = [("imputer", SimpleImputer(strategy=imputer))]
    if SCALER_REGISTRY[scaler_name] is not None:
        numeric_steps.append(("scaler", SCALER_REGISTRY[scaler_name]()))

    # HistGradientBoosting requires dense input; all other models can use sparse OHE.
    # Level-2 transformations use dense sklearn feature arrays. Never let an
    # incompatible sparse matrix silently reach PCA or StandardScaler.
    dense = bool(composition) or model_name == "hist_gradient_boosting"
    if encoder_name == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    else:
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", cat_encoder),
    ])
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipe, categorical))
    preprocessing = ColumnTransformer(transformers, sparse_threshold=0 if dense else 0.3)

    if composition is not None:
        return compile_composition(composition, preprocessing, X_train[columns], cfg)

    params = model_spec.get("hyperparameters", {})
    if not isinstance(params, dict):
        raise ValueError("hyperparameters must be a dictionary")

    params = dict(params)  # Avoid mutating the original candidate
    for name, value in params.items():
        if isinstance(value, (list, dict)):
            raise ValueError(
                f"Hyperparameter '{name}' must be a single value, "
                f"not {value!r}. Generate separate candidates "
                "for each configuration."
            )

    # Normalize common LLM-generated parameter aliases
    if model_name == "hist_gradient_boosting":
        if "n_estimators" in params:
            if "max_iter" in params:
                raise ValueError(
                    "Specify either n_estimators or max_iter, not both"
                )
            params["max_iter"] = params.pop("n_estimators")

    # Normalize deprecated logistic regression penalty
    if model_name == "logistic_regression" and "penalty" in params:
        penalty = params.pop("penalty")

        if penalty == "l2":
            params.setdefault("l1_ratio", 0.0)
        elif penalty == "l1":
            params.setdefault("l1_ratio", 1.0)
        elif penalty == "elasticnet":
            params.setdefault("l1_ratio", 0.5)
        elif penalty is None:
            params["C"] = float("inf")
        else:
            raise ValueError(f"Unsupported logistic penalty: {penalty!r}")

    params = validate_params(model_name, params)

    cls = MODEL_REGISTRY[model_name]
    valid_params = cls().get_params()

    invalid = set(params) - set(valid_params)
    if invalid:
        raise ValueError(
            f"Unsupported {model_name} hyperparameters: {sorted(invalid)}"
        )

    if "random_state" in valid_params:
        params.setdefault("random_state", cfg.get("seed", 42))

    model = cls(**params)

    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def run_cv(candidate, train_df, target, cfg):
    """Repeated stratified CV; never accesses the holdout."""
    from ml.scoring import cross_validate_pipeline

    candidate = _as_dict(candidate)
    if target not in train_df:
        raise ValueError(f"Target column {target!r} not found")
    X, y = train_df.drop(columns=[target]), train_df[target]
    cv_cfg = cfg.get("cv", {})
    pipe = build_pipeline(candidate, X, cfg)
    return {
        "model_name": (candidate["model"]["name"] if candidate.get("model")
                       else f"composition:{candidate['composition']['kind']}"),
        **cross_validate_pipeline(
            pipe, X, y,
            folds=int(cv_cfg.get("folds", cv_cfg.get("n_splits", 5))),
            repeats=int(cv_cfg.get("repeats", 3)),
            seed=cfg.get("seed", 42),
            scoring=cv_cfg.get("scoring", "f1_macro"),
            n_jobs=cv_cfg.get("n_jobs", 1),
        ),
    }
