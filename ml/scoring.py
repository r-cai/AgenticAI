"""Shared CV, final-fit, and holdout scoring for baseline and agentic experiments."""
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, get_scorer)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


def cross_validate_pipeline(pipe, X, y, *, folds=5, repeats=3, seed=42,
                            scoring='f1_macro', n_jobs=1):
    """Use identical, shuffled repeated stratified partitions for every candidate."""
    if folds < 2 or repeats < 1:
        raise ValueError('folds must be >= 2 and repeats >= 1')
    if min(y.value_counts() if hasattr(y, 'value_counts') else np.unique(y, return_counts=True)[1]) < folds:
        raise ValueError('Every class needs at least folds training observations')
    get_scorer(scoring)
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                                return_train_score=True, n_jobs=n_jobs, error_score='raise')
    valid = np.asarray(scores['test_score'], dtype=float)
    train = np.asarray(scores['train_score'], dtype=float)
    issues = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return {
        'scoring': scoring, 'mean_cv_score': round(float(valid.mean()), 4),
        'std_cv_score': round(float(valid.std()), 4),
        'train_mean': round(float(train.mean()), 4),
        'train_cv_gap': round(float(train.mean() - valid.mean()), 4),
        'fold_scores': [round(float(v), 4) for v in valid],
        'n_folds': folds, 'n_repeats': repeats, 'n_evaluations': folds * repeats,
        'convergence_warning_count': len(issues),
        'convergence_warning': issues[0] if issues else None,
        'cv_fit_time_mean_seconds': round(float(np.mean(scores['fit_time'])), 4),
    }


def fit_final_pipeline(pipe, X_train, y_train):
    """Fit once on all training observations; return descriptive fit statistics."""
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipe.fit(X_train, y_train)
    elapsed = round(time.perf_counter() - start, 4)
    model = pipe.named_steps['model']
    iterations = getattr(model, 'n_iter_', None)
    if iterations is not None:
        iterations = int(np.max(np.asarray(iterations)))
    issues = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return {
        'fit_time_seconds': elapsed, 'iterations_used': iterations,
        'convergence_warning': bool(issues),
        'warning_detail': issues[0] if issues else None,
    }


def score_holdout(pipe, X_test, y_test, scoring='f1_macro', positive_label=None):
    """Score a previously fitted pipeline. Never use these results for tuning."""
    preds = pipe.predict(X_test)
    result = {
        'score': round(float(get_scorer(scoring)(pipe, X_test, y_test)), 4),
        'scoring': scoring,
        'precision': round(float(precision_score(y_test, preds, average='macro', zero_division=0)), 4),
        'accuracy': round(float(accuracy_score(y_test, preds)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y_test, preds)), 4),
        'recall': round(float(recall_score(y_test, preds, average='macro', zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_test, preds, average='macro', zero_division=0)), 4),
    }
    result['f1_macro'] = result['f1_score']
    classes = pipe.named_steps['model'].classes_
    auc = None
    try:
        if len(classes) == 2:
            if hasattr(pipe, 'predict_proba'):
                raw = pipe.predict_proba(X_test)[:, 1]
            elif hasattr(pipe, 'decision_function'):
                raw = pipe.decision_function(X_test)
            else:
                raw = None
            if raw is not None:
                auc = roc_auc_score(y_test, raw, labels=classes)
        elif hasattr(pipe, 'predict_proba'):
            auc = roc_auc_score(y_test, pipe.predict_proba(X_test),
                                labels=classes, multi_class='ovr', average='macro')
    except ValueError:
        pass  # AUC undefined when holdout is missing a class
    result['auc'] = round(float(auc), 4) if auc is not None else None
    result['roc_auc'] = result['auc']
    if positive_label is not None and positive_label in classes:
        result['positive_label'] = positive_label
        result['positive_sensitivity'] = round(float(
            recall_score(y_test, preds, pos_label=positive_label,
                         average='binary', zero_division=0)), 4)
        result['positive_precision'] = round(float(
            precision_score(y_test, preds, pos_label=positive_label,
                            average='binary', zero_division=0)), 4)
        # AUC for the explicitly positive condition (e.g. malignant=1).
        if hasattr(pipe, 'predict_proba') and len(classes) == 2:
            positives = (np.asarray(y_test) == positive_label).astype(int)
            if len(np.unique(positives)) == 2:
                col = list(classes).index(positive_label)
                result['positive_roc_auc'] = round(float(
                    roc_auc_score(positives, pipe.predict_proba(X_test)[:, col])), 4)
    return result
