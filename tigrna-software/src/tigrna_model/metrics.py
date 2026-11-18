from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def safe_correlation(function, observed: np.ndarray, predicted: np.ndarray) -> float:
    try:
        result = function(observed, predicted)
        value = result.statistic if hasattr(result, "statistic") else result[0]
        return float(value) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    high_threshold = float(np.quantile(observed, 0.75))
    high = observed >= high_threshold
    metrics = {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson": safe_correlation(pearsonr, observed, predicted),
        "spearman": safe_correlation(spearmanr, observed, predicted),
        "high_efficiency_threshold": high_threshold,
    }
    try:
        metrics["top_quartile_roc_auc"] = float(roc_auc_score(high, predicted))
        metrics["top_quartile_average_precision"] = float(
            average_precision_score(high, predicted)
        )
    except ValueError:
        metrics["top_quartile_roc_auc"] = float("nan")
        metrics["top_quartile_average_precision"] = float("nan")
    top_n = max(1, int(np.ceil(0.2 * len(observed))))
    selected = np.argsort(predicted)[-top_n:]
    overall_mean = float(np.mean(observed))
    top_mean = float(np.mean(observed[selected]))
    metrics["top20_observed_mean"] = top_mean
    metrics["top20_enrichment"] = top_mean / overall_mean if overall_mean else float("nan")
    return metrics


def group_bootstrap_ci(
    observed: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    iterations: int = 1000,
    seed: int = 20260819,
) -> dict[str, list[float]]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"mae": [], "spearman": [], "pearson": []}
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(iterations):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([group_indices[group] for group in chosen])
        metrics = regression_metrics(observed[index], predicted[index])
        for name in samples:
            if np.isfinite(metrics[name]):
                samples[name].append(metrics[name])
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in samples.items()
        if values
    }
