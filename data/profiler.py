import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_PROFILE_COLUMNS = 40  # keeps the profile inside the LLM context window
MIN_CLASS_SAMPLES = 2


def load_and_profile(csv_path, target, test_size=0.2, seed=42,  cv_folds=5):
    """Entry point used by the controller: (train_df, test_df, profile)."""
    train_df, test_df = load_and_split(csv_path, target, test_size, seed, min_train_per_class=cv_folds)
    return train_df, test_df, profile_dataset(train_df, target)


def load_and_split(csv_path, target, test_size=0.2, seed=42, min_train_per_class=5):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    empty_columns = [
        col for col in df.columns
        if col != target and df[col].isna().all()
    ]

    if empty_columns:
        print(f"[data] Dropping entirely empty columns: {empty_columns}")
        df = df.drop(columns=empty_columns)
    # sklearn imputers don't handle infinity and one such value fails every candidate
    df = df.replace([np.inf, -np.inf], np.nan)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)[:15]}")
    if df[target].isna().any():
        raise ValueError("Target column contains missing values.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    class_counts = df[target].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Target must have at least 2 classes.")
    rare = class_counts[class_counts < MIN_CLASS_SAMPLES]
    if len(rare):
        raise ValueError(
            f"{len(rare)} of {len(class_counts)} classes have fewer than {MIN_CLASS_SAMPLES} "
            f"samples (is the target continuous rather than a class label?)."
        )

    n_test = math.ceil(len(df) * test_size)
    if min(len(df) - n_test, n_test) < len(class_counts):
        raise ValueError("Train and test sets must each have at least one sample per class.")

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )

    # CV needs every class present in each fold, so check the training side explicitly.
    smallest = int(train_df[target].value_counts().min())
    if smallest < min_train_per_class:
        raise ValueError(
            f"Smallest class has only {smallest} training samples after the split; "
            f"{min_train_per_class}-fold CV needs at least {min_train_per_class}."
        )
    return train_df, test_df


def profile_dataset(train_df, target) -> dict:
    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    y = train_df[target]
    class_counts = y.value_counts()
    profile = {
        "n_rows": len(train_df),
        "n_features": train_df.shape[1] - 1,
        "target": target,
        "n_classes": int(y.nunique()),
        "class_distribution": {
            str(k): round(float(v), 3) for k, v in y.value_counts(normalize=True).items()
        },
        "imbalance_ratio": round(float(class_counts.max() / class_counts.min()), 2),
        "columns": {},
    }

    feature_cols = [c for c in train_df.columns if c != target]
    for col in feature_cols[:MAX_PROFILE_COLUMNS]:
        series = train_df[col]
        n_unique = int(series.nunique())
        info = {
            "dtype": str(series.dtype),
            "missing_pct": round(float(series.isna().mean() * 100), 1),
            "n_unique": n_unique,
            # near 1.0 on a non-target column suggests an identifier: the generator should drop it
            "unique_ratio": round(n_unique / max(len(series), 1), 3),
            "sample_values": [str(v) for v in series.dropna().unique()[:5]],
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = series.astype(float)
            info["type"] = "numeric"
            info["stats"] = {
                "mean": _safe_float(numeric.mean()),
                "std": _safe_float(numeric.std()),
                "min": _safe_float(numeric.min()),
                "max": _safe_float(numeric.max()),
                "median": _safe_float(numeric.median()),
            }
        else:
            info["type"] = "categorical"
            info["top_values"] = {str(k): int(v) for k, v in series.value_counts().head(5).items()}
        profile["columns"][col] = info

    if len(feature_cols) > MAX_PROFILE_COLUMNS:
        profile["n_columns_omitted"] = len(feature_cols) - MAX_PROFILE_COLUMNS
    return profile


def _safe_float(value):
    """Convert numeric statistics to JSON-safe values (None for NaN/inf)."""
    if pd.isna(value) or not math.isfinite(float(value)):
        return None
    return round(float(value), 3)