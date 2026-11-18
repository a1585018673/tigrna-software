from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .features import FEATURE_VERSION, feature_matrix
from .metrics import group_bootstrap_ci, regression_metrics


GRID = {
    "model__C": [0.1, 1.0, 10.0],
    "model__epsilon": [0.02, 0.05, 0.10],
    "model__gamma": ["scale", 0.01],
}


def make_estimator() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ]
    )


def within_locus_metrics(
    observed: np.ndarray, predicted: np.ndarray, groups: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        y = observed[index]
        p = predicted[index]
        if len(index) < 3 or len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
            continue
        rank_correlation = float(spearmanr(y, p).statistic)
        top1 = float(y[np.argmax(p)] == y.max())
        comparable = correct = 0
        for left in range(len(index)):
            for right in range(left + 1, len(index)):
                observed_delta = y[left] - y[right]
                predicted_delta = p[left] - p[right]
                if observed_delta != 0 and predicted_delta != 0:
                    comparable += 1
                    correct += observed_delta * predicted_delta > 0
        pairwise = correct / comparable if comparable else np.nan
        rows.append(
            {
                "locus_group": group,
                "n": len(index),
                "spearman": rank_correlation,
                "top1_correct": top1,
                "pairwise_accuracy": pairwise,
            }
        )
    detail = pd.DataFrame(rows)
    metrics = {
        "eligible_locus_groups": int(len(detail)),
        "macro_spearman": float(detail["spearman"].mean()),
        "median_spearman": float(detail["spearman"].median()),
        "top1_accuracy": float(detail["top1_correct"].mean()),
        "macro_pairwise_accuracy": float(detail["pairwise_accuracy"].mean()),
    }
    rng = np.random.default_rng(20260819)
    boot = {name: [] for name in ["macro_spearman", "top1_accuracy", "macro_pairwise_accuracy"]}
    for _ in range(2000):
        sample = detail.iloc[rng.integers(0, len(detail), size=len(detail))]
        boot["macro_spearman"].append(float(sample["spearman"].mean()))
        boot["top1_accuracy"].append(float(sample["top1_correct"].mean()))
        boot["macro_pairwise_accuracy"].append(float(sample["pairwise_accuracy"].mean()))
    metrics["bootstrap_95_ci"] = {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in boot.items()
    }
    return metrics, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    reports = root / "reports"
    models = root / "models"
    frame = pd.read_csv(root / "data" / "processed" / "training_dataset.csv")
    frame = frame.loc[frame["cell_line"].eq("293T")].reset_index(drop=True)
    X, feature_names = feature_matrix(frame["sequence_20"])
    y = frame["optimized_efficiency"].to_numpy(dtype=float)
    groups = frame["locus_group"].to_numpy()
    weights = np.sqrt(frame["optimized_n"].to_numpy(dtype=float))

    outer = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    dummy_oof = np.full(len(y), np.nan)
    fold_records = []
    for fold, (train_index, test_index) in enumerate(outer.split(X, y, groups), start=1):
        search = GridSearchCV(
            make_estimator(),
            GRID,
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=4),
            n_jobs=args.n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(
            X[train_index],
            y[train_index],
            groups=groups[train_index],
            model__sample_weight=weights[train_index],
        )
        oof[test_index] = np.clip(search.predict(X[test_index]), 0.0, 1.0)
        dummy = DummyRegressor(strategy="median")
        dummy.fit(X[train_index], y[train_index], sample_weight=weights[train_index])
        dummy_oof[test_index] = dummy.predict(X[test_index])
        fold_records.append(
            {
                "fold": fold,
                "test_groups": sorted(set(groups[test_index].tolist())),
                "test_n": int(len(test_index)),
                "best_params": search.best_params_,
                "inner_cv_mae": float(-search.best_score_),
            }
        )

    final_search = GridSearchCV(
        make_estimator(),
        GRID,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=5),
        n_jobs=args.n_jobs,
        refit=True,
        error_score="raise",
    )
    final_search.fit(X, y, groups=groups, model__sample_weight=weights)
    final_prediction_reference = np.sort(np.clip(final_search.predict(X), 0.0, 1.0))

    cv_metrics = regression_metrics(y, oof)
    cv_metrics["bootstrap_95_ci"] = group_bootstrap_ci(y, oof, groups, iterations=2000)
    baseline_metrics = regression_metrics(y, dummy_oof)
    ranking_metrics, ranking_detail = within_locus_metrics(y, oof, groups)
    radius = float(np.quantile(np.abs(y - oof), 0.90, method="higher"))

    payload = {
        "model_status": "experimental",
        "calibration_context": "293T, optimized 336-172 construct",
        "n_samples": int(len(frame)),
        "n_locus_groups": int(frame["locus_group"].nunique()),
        "nested_group_cv": cv_metrics,
        "dummy_group_cv": baseline_metrics,
        "within_locus_ranking": ranking_metrics,
        "conformal_absolute_residual_radius_90": radius,
        "final_params": final_search.best_params_,
        "folds": fold_records,
    }
    (reports / "metrics_293T.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    ranking_detail.to_csv(reports / "within_locus_ranking_293T.csv", index=False)
    oof_frame = frame[
        ["guide_id", "locus_group", "sequence_20", "optimized_efficiency", "optimized_n"]
    ].copy()
    oof_frame["oof_prediction"] = oof
    oof_frame["absolute_error"] = np.abs(y - oof)
    oof_frame.to_csv(reports / "oof_predictions_293T.csv", index=False)

    bundle = {
        "estimator": final_search.best_estimator_,
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "model_family": "svr_rbf",
        "model_params": final_search.best_params_,
        "model_status": "experimental",
        "calibration_context": "293T, optimized 336-172 construct",
        "training_label_distribution": np.sort(y),
        "score_reference_predictions": final_prediction_reference,
        "training_sequences": frame["sequence_20"].tolist(),
        "training_guide_ids": frame["guide_id"].tolist(),
        "training_cell_lines": frame["cell_line"].tolist(),
        "conformal_radius_90": radius,
        "target_definition": "mean of non-missing indel%(336-172) measurements",
    }
    joblib.dump(bundle, models / "tigrna_efficiency_model_293T.joblib", compress=3)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_status": "experimental",
        "calibration_context": bundle["calibration_context"],
        "n_training_samples": len(frame),
        "n_locus_groups": frame["locus_group"].nunique(),
        "feature_version": FEATURE_VERSION,
        "final_params": final_search.best_params_,
        "primary_use": "ranking multiple candidate tigRNAs for the same locus",
        "absolute_efficiency_use": "rough reference only; interval is wide",
    }
    (models / "model_metadata_293T.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(7.2, 6.0))
    plt.scatter(y * 100, oof * 100, c="#2563EB", alpha=0.78, edgecolor="white", linewidth=0.5)
    limit = max(float(y.max()), float(oof.max())) * 105
    plt.plot([0, limit], [0, limit], "--", color="#64748B", linewidth=1)
    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.xlabel("Observed optimized indel efficiency (%)")
    plt.ylabel("Nested locus-group CV prediction (%)")
    plt.title("Experimental 293T tigRNA sequence model")
    plt.tight_layout()
    plt.savefig(reports / "oof_293T_observed_vs_predicted.png", dpi=220)
    plt.close()

    ci = ranking_metrics["bootstrap_95_ci"]
    card = f"""# 293T tigRNA 效率模型卡

## 结论

这是实验性同位点候选排序器，不是高置信绝对效率预测器。默认模型只用 293T 的优化体（336-172）数据校准。

## 数据与验证

- 训练样本：{len(frame)} 条，来自 {frame['locus_group'].nunique()} 个独立位点。
- 标签：同一行 `indel%(336-172)` 非空测量的算术均值；缺失不按 0 处理。
- 验证：按基因/位点分组的嵌套 5 折交叉验证；同一位点的 guide 不跨训练折与测试折。

## 结果

- 绝对效率 MAE：{cv_metrics['mae'] * 100:.2f} 个百分点；常数基线 {baseline_metrics['mae'] * 100:.2f} 个百分点。
- 整体 Spearman：{cv_metrics['spearman']:.3f}（95% 组自助法区间 {cv_metrics['bootstrap_95_ci']['spearman'][0]:.3f} 到 {cv_metrics['bootstrap_95_ci']['spearman'][1]:.3f}）。
- 25 个可评估位点内，宏平均 Spearman：{ranking_metrics['macro_spearman']:.3f}（95% 区间 {ci['macro_spearman'][0]:.3f} 到 {ci['macro_spearman'][1]:.3f}）。
- 同位点成对排序正确率：{ranking_metrics['macro_pairwise_accuracy']:.3f}（95% 区间 {ci['macro_pairwise_accuracy'][0]:.3f} 到 {ci['macro_pairwise_accuracy'][1]:.3f}）。
- 选中同位点实测第一名的比例：{ranking_metrics['top1_accuracy']:.3f}（95% 区间 {ci['top1_accuracy'][0]:.3f} 到 {ci['top1_accuracy'][1]:.3f}）。
- 90% 经验绝对残差半径：±{radius * 100:.2f} 个百分点。

## 推荐用法

对同一个基因/位点一次输入多条候选，优先参考 `batch_rank_0_100` 和 `score_0_100` 排序，同时保留多条进入实验。`predicted_efficiency_pct` 只能作粗略参考。

## 限制

- 没有染色质可及性、基因组坐标、转录水平、脱靶和批次信息，无法解释主要的跨位点差异。
- N2a 只有一个 PCSK 位点，不能建立可验证的 N2a 专用模型。
- 蛋白序列在所有样本中相同，只作为版本元数据，不能充当学习特征。
- 新蛋白改造、tigRNA scaffold、表达或测序流程变化后必须重新校准。
"""
    (reports / "model_card_293T.md").write_text(card, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
