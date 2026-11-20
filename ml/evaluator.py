"""Agentic final evaluation, using the same scorer as the baseline."""
from ml.trainer import build_pipeline
import pandas as pd
import numpy as np
from ml.scoring import fit_final_pipeline, score_holdout


def evaluate_holdout(candidate, train_df, test_df, target, cfg, *,
                     prediction_path=None, prediction_metadata=None) -> dict:
    if hasattr(candidate, "model_dump"):
        candidate = candidate.model_dump()
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]
    pipe = build_pipeline(candidate, X_train, cfg)
    convergence = fit_final_pipeline(pipe, X_train, y_train)
    cv_cfg = cfg.get("cv", {})
    result = score_holdout(
        pipe, X_test, y_test, cv_cfg.get("scoring", "f1_macro"),
        positive_label=cv_cfg.get("positive_label"),
    )
    if prediction_path is not None:
        # Original CSV row indices permit exact matching across arms in a seed.
        # No feature or patient identifier is needed in the exported predictions.
        out = pd.DataFrame({
            "row_index": [int(i) for i in test_df.index],
            "true_label": y_test.astype(str).to_numpy(),
            "predicted_label": pd.Series(pipe.predict(X_test)).astype(str).to_numpy(),
        })
        if hasattr(pipe, "predict_proba"):
            probabilities = pipe.predict_proba(X_test)
            classes = list(pipe.named_steps["model"].classes_)
            if probabilities.shape[1] == len(classes):
                for j, cls in enumerate(classes):
                    out[f"probability_{cls}"] = probabilities[:, j]
        if prediction_metadata:
            for key, value in prediction_metadata.items():
                out[key] = value
        prediction_path = __import__("pathlib").Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(prediction_path, index=False, float_format="%.12g")
    return {**result, "n_train": len(train_df), "n_test": len(test_df),
            "convergence": convergence}
