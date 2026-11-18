from __future__ import annotations

import argparse
import json
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import __version__
from .dataio import load_experiment_workbook, sha256_file
from .features import FEATURE_VERSION, feature_matrix
from .metrics import group_bootstrap_ci, regression_metrics


RANDOM_STATE = 20260819


def candidate_specs() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "ridge": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            {"model__alpha": [1.0, 10.0, 100.0, 1000.0, 10000.0]},
        ),
        "svr_rbf": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__epsilon": [0.02, 0.05, 0.10],
                "model__gamma": ["scale", 0.01],
            },
        ),
        "extra_trees": (
            Pipeline(
                [
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=400,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            {
                "model__max_features": [0.25, 0.5, 1.0],
                "model__min_samples_leaf": [2, 4, 8],
                "model__max_depth": [None, 8],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=300,
                            learning_rate=0.05,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            {
                "model__l2_regularization": [0.1, 1.0, 10.0],
                "model__min_samples_leaf": [5, 10, 20],
                "model__max_leaf_nodes": [7, 15],
            },
        ),
    }


def tune_candidate(
    estimator: Pipeline,
    grid: dict[str, list[object]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    n_jobs: int,
) -> GridSearchCV:
    split_count = min(4, len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("at least two independent locus groups are required")
    search = GridSearchCV(
        clone(estimator),
        grid,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=split_count),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups, model__sample_weight=weights)
    return search


def nested_group_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, object]]]:
    specs = candidate_specs()
    outer = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    candidate_predictions = {name: np.full(len(y), np.nan) for name in specs}
    adaptive_predictions = np.full(len(y), np.nan)
    fold_records: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(outer.split(X, y, groups), start=1):
        inner_results: dict[str, dict[str, object]] = {}
        for name, (estimator, grid) in specs.items():
            search = tune_candidate(
                estimator,
                grid,
                X[train_index],
                y[train_index],
                groups[train_index],
                weights[train_index],
                n_jobs,
            )
            prediction = np.clip(search.predict(X[test_index]), 0.0, 1.0)
            candidate_predictions[name][test_index] = prediction
            inner_results[name] = {
                "inner_cv_mae": float(-search.best_score_),
                "best_params": search.best_params_,
                "estimator": search.best_estimator_,
            }
        selected_name = min(inner_results, key=lambda name: inner_results[name]["inner_cv_mae"])
        adaptive_predictions[test_index] = candidate_predictions[selected_name][test_index]
        fold_records.append(
            {
                "fold": fold,
                "test_groups": sorted(set(groups[test_index].tolist())),
                "test_n": int(len(test_index)),
                "selected_family": selected_name,
                "inner_results": {
                    name: {
                        "inner_cv_mae": result["inner_cv_mae"],
                        "best_params": result["best_params"],
                    }
                    for name, result in inner_results.items()
                },
            }
        )
    return candidate_predictions, adaptive_predictions, fold_records


def fit_final_model(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    n_jobs: int,
) -> GridSearchCV:
    estimator, grid = candidate_specs()[family]
    return tune_candidate(estimator, grid, X, y, groups, weights, n_jobs)


def save_plots(
    output_dir: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    frame: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 6.2))
    colors = frame["cell_line"].map({"293T": "#2563EB", "N2a": "#EA580C"}).fillna("#64748B")
    plt.scatter(observed * 100, predicted * 100, c=colors, alpha=0.78, edgecolor="white", linewidth=0.4)
    limit = max(5.0, float(max(observed.max(), predicted.max()) * 100 * 1.05))
    plt.plot([0, limit], [0, limit], linestyle="--", color="#64748B", linewidth=1)
    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.xlabel("Observed optimized indel efficiency (%)")
    plt.ylabel("Nested group-CV prediction (%)")
    plt.title("TIGR-Tas optimized-efficiency model")
    plt.tight_layout()
    plt.savefig(output_dir / "oof_observed_vs_predicted.png", dpi=220)
    plt.close()

    calibration = pd.DataFrame({"observed": observed, "predicted": predicted})
    calibration["bin"] = pd.qcut(calibration["predicted"], q=5, duplicates="drop")
    summary = calibration.groupby("bin", observed=False).agg(
        predicted_mean=("predicted", "mean"),
        observed_mean=("observed", "mean"),
        observed_std=("observed", "std"),
        n=("observed", "size"),
    )
    plt.figure(figsize=(7.2, 5.4))
    plt.errorbar(
        summary["predicted_mean"] * 100,
        summary["observed_mean"] * 100,
        yerr=summary["observed_std"].fillna(0) * 100 / np.sqrt(summary["n"]),
        fmt="o-",
        color="#0F766E",
        capsize=4,
    )
    plt.xlabel("Mean predicted efficiency in score bin (%)")
    plt.ylabel("Mean observed efficiency (%)")
    plt.title("Out-of-fold calibration by prediction quintile")
    plt.tight_layout()
    plt.savefig(output_dir / "oof_calibration.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 5.4))
    for cell_line, group in frame.assign(label=observed).groupby("cell_line"):
        plt.hist(group["label"] * 100, bins=18, alpha=0.55, label=cell_line)
    plt.xlabel("Optimized indel efficiency (%)")
    plt.ylabel("Number of guides")
    plt.title("Training-label distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=220)
    plt.close()


def render_model_card(
    path: Path,
    metrics: dict[str, object],
    dataset: pd.DataFrame,
    selected_family: str,
    selected_params: dict[str, object],
    interval_radius: float,
) -> None:
    adaptive = metrics["adaptive_nested_group_cv"]
    baseline = metrics["dummy_group_cv"]
    cell_counts = dataset.groupby("cell_line").size().to_dict()
    lines = [
        "# tigRNA 优化体效率模型卡",
        "",
        "## 用途",
        "",
        "针对 1 nt 5′侧翼 + 18 nt target + 1 nt 3′侧翼的 tigRNA 靶序列，输出优化体（工作簿中的 `336-172`）预估 indel 效率、0-100 相对分数和保守的不确定性区间。",
        "",
        "## 数据与标签",
        "",
        f"- 可训练样本：{len(dataset)} 条；按细胞系计数：{cell_counts}。",
        f"- 独立位点组：{dataset['locus_group'].nunique()} 个。",
        "- 主标签：同一行 `indel%(336-172)` 非空测量的算术均值；缺失值不按 0 处理。",
        "- 同一蛋白序列对所有样本恒定，因此只作为版本元数据保存，不进入特征。",
        "",
        "## 验证设计",
        "",
        "使用按基因/位点分组的嵌套 5 折交叉验证。相同位点的 guide 不会同时出现在训练折和测试折，避免重叠序列造成泄漏。模型家族和超参数只在外层训练折内选择。",
        "",
        "## 核心结果",
        "",
        f"- 嵌套分组 CV MAE：{adaptive['mae'] * 100:.2f} 个百分点（95% 组自助法区间 {adaptive['bootstrap_95_ci']['mae'][0] * 100:.2f}-{adaptive['bootstrap_95_ci']['mae'][1] * 100:.2f}）。",
        f"- Spearman：{adaptive['spearman']:.3f}（95% 组自助法区间 {adaptive['bootstrap_95_ci']['spearman'][0]:.3f}-{adaptive['bootstrap_95_ci']['spearman'][1]:.3f}）。",
        f"- Pearson：{adaptive['pearson']:.3f}。",
        f"- 常数基线 MAE：{baseline['mae'] * 100:.2f} 个百分点。",
        f"- 最终模型：`{selected_family}`，参数 `{selected_params}`。",
        f"- 90% 经验残差半径：±{interval_radius * 100:.2f} 个百分点。",
        "",
        "## 重要限制",
        "",
        "- 这是小样本、序列内在特征模型，不含染色质可及性、转录水平、基因组坐标、脱靶或实验批次；预估效率不能替代实测。",
        "- N2a 数据来自单一 PCSK 位点，细胞系与位点混杂；模型不把细胞系作为输入，也不能据此声称跨细胞系校准。",
        "- 论文报告 TasR 无强 TAM/PAM 偏好，但两侧碱基仍作为弱位置特征保留，由正则化控制。",
        "- 新的蛋白改造体、不同 tigRNA scaffold、表达系统或测序流程需要重新校准或增量训练。",
        "",
        "## 评分含义",
        "",
        "`score_0_100` 是预估效率在训练标签分布中的百分位，不是成功概率。候选排序时应同时参考预估效率、不确定性区间、与训练集最近序列的一致性及实验可行性。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    root = args.project_root.resolve()
    xlsx_candidates = sorted((root / "data" / "raw").glob("*.xlsx"))
    xlsx_path = args.xlsx.resolve() if args.xlsx else (xlsx_candidates[0] if xlsx_candidates else None)
    if xlsx_path is None or not xlsx_path.exists():
        raise FileNotFoundError("no input .xlsx found under data/raw")

    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    reports_dir = root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_experiment_workbook(xlsx_path)
    all_rows.to_csv(processed_dir / "all_experimental_rows.csv", index=False)
    dataset = all_rows.loc[all_rows["primary_label_available"]].reset_index(drop=True)
    dataset.to_csv(processed_dir / "training_dataset.csv", index=False)
    if len(dataset) < 30:
        raise ValueError(f"only {len(dataset)} optimized-efficiency labels are available")

    X, names = feature_matrix(dataset["sequence_20"])
    y = dataset["optimized_efficiency"].to_numpy(dtype=float)
    groups = dataset["locus_group"].to_numpy()
    weights = dataset["sample_weight"].to_numpy(dtype=float)

    candidate_predictions, adaptive_predictions, fold_records = nested_group_evaluation(
        X, y, groups, weights, args.n_jobs
    )
    if np.isnan(adaptive_predictions).any():
        raise RuntimeError("nested CV did not produce a prediction for every sample")

    candidate_metrics: dict[str, dict[str, float]] = {}
    for name, prediction in candidate_predictions.items():
        candidate_metrics[name] = regression_metrics(y, prediction)
    selected_family = min(candidate_metrics, key=lambda name: candidate_metrics[name]["mae"])

    final_search = fit_final_model(selected_family, X, y, groups, weights, args.n_jobs)
    final_estimator = final_search.best_estimator_
    final_training_predictions = np.clip(final_estimator.predict(X), 0.0, 1.0)

    dummy_predictions = np.full(len(y), np.nan)
    outer = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_index, test_index in outer.split(X, y, groups):
        dummy = DummyRegressor(strategy="median")
        dummy.fit(X[train_index], y[train_index], sample_weight=weights[train_index])
        dummy_predictions[test_index] = dummy.predict(X[test_index])

    adaptive_metrics = regression_metrics(y, adaptive_predictions)
    adaptive_metrics["bootstrap_95_ci"] = group_bootstrap_ci(
        y, adaptive_predictions, groups, iterations=1000
    )
    dummy_metrics = regression_metrics(y, dummy_predictions)
    interval_radius = float(np.quantile(np.abs(y - adaptive_predictions), 0.90, method="higher"))

    metrics_payload: dict[str, object] = {
        "adaptive_nested_group_cv": adaptive_metrics,
        "dummy_group_cv": dummy_metrics,
        "candidate_nested_group_cv": candidate_metrics,
        "selected_final_family": selected_family,
        "selected_final_params": final_search.best_params_,
        "selected_full_group_cv_mae": float(-final_search.best_score_),
        "conformal_absolute_residual_radius_90": interval_radius,
        "folds": fold_records,
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    comparison_rows = []
    for name, values in candidate_metrics.items():
        comparison_rows.append({"model": name, **values})
    comparison_rows.append({"model": "adaptive_selection", **adaptive_metrics})
    comparison_rows.append({"model": "dummy_median", **dummy_metrics})
    pd.DataFrame(comparison_rows).to_csv(reports_dir / "model_comparison.csv", index=False)

    oof = dataset[
        [
            "guide_id",
            "cell_line",
            "locus_group",
            "sequence_20",
            "optimized_efficiency",
            "optimized_n",
        ]
    ].copy()
    for name, prediction in candidate_predictions.items():
        oof[f"oof_{name}"] = prediction
    oof["oof_adaptive"] = adaptive_predictions
    oof["absolute_error_adaptive"] = np.abs(y - adaptive_predictions)
    oof.to_csv(reports_dir / "oof_predictions.csv", index=False)

    bundle = {
        "estimator": final_estimator,
        "feature_version": FEATURE_VERSION,
        "feature_names": names,
        "model_family": selected_family,
        "model_params": final_search.best_params_,
        "training_label_distribution": np.sort(y),
        "training_sequences": dataset["sequence_20"].tolist(),
        "training_guide_ids": dataset["guide_id"].tolist(),
        "training_cell_lines": dataset["cell_line"].tolist(),
        "conformal_radius_90": interval_radius,
        "target_definition": "mean of non-missing indel%(336-172) measurements",
    }
    joblib.dump(bundle, models_dir / "tigrna_efficiency_model.joblib", compress=3)

    metadata = {
        "model_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "source_workbook": xlsx_path.name,
        "source_workbook_sha256": sha256_file(xlsx_path),
        "n_training_samples": int(len(dataset)),
        "n_locus_groups": int(dataset["locus_group"].nunique()),
        "cell_line_counts": dataset.groupby("cell_line").size().to_dict(),
        "label_min": float(y.min()),
        "label_median": float(np.median(y)),
        "label_mean": float(y.mean()),
        "label_max": float(y.max()),
        "selected_family": selected_family,
        "selected_params": final_search.best_params_,
        "feature_version": FEATURE_VERSION,
        "protein_sequence_role": "constant provenance metadata; not a predictive input",
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_plots(reports_dir, y, adaptive_predictions, dataset)
    render_model_card(
        reports_dir / "model_card.md",
        metrics_payload,
        dataset,
        selected_family,
        final_search.best_params_,
        interval_radius,
    )

    print(json.dumps({"metadata": metadata, "metrics": metrics_payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
