from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def entropy_score(sequence: str) -> float:
    values = []
    for base in "ACGT":
        fraction = sequence.count(base) / len(sequence)
        if fraction:
            values.append(-fraction * math.log2(fraction))
    return 50.0 * sum(values)


def max_run(sequence: str) -> int:
    best = current = 1
    for previous, base in zip(sequence, sequence[1:]):
        if previous == base:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def base_run(sequence: str, wanted: str) -> int:
    best = current = 0
    for base in sequence:
        if base == wanted:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def homopolymer_safety(run: int) -> float:
    if run <= 2:
        return 100.0
    if run == 3:
        return 70.0
    if run == 4:
        return 35.0
    return 0.0


def clean_number(value):
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: prepare_scoring_data.py <all_rows.csv> <scores.csv> <oof.csv> <output.json>"
        )
    rows_path, scores_path, oof_path, output_path = map(Path, sys.argv[1:])
    rows = pd.read_csv(rows_path)
    scores = pd.read_csv(scores_path)
    oof = pd.read_csv(oof_path)

    score_columns = [
        "id",
        "predicted_efficiency_pct",
        "prediction_interval_90_low_pct",
        "prediction_interval_90_high_pct",
        "score_0_100",
        "nearest_training_guide",
        "nearest_sequence_identity_pct",
        "model_status",
        "calibration_context",
    ]
    data = rows.merge(scores[score_columns], left_on="guide_id", right_on="id", how="left")
    data = data.merge(
        oof[["guide_id", "oof_prediction", "absolute_error"]],
        on="guide_id",
        how="left",
    )
    if data["predicted_efficiency_pct"].isna().any():
        missing = data.loc[data["predicted_efficiency_pct"].isna(), "guide_id"].tolist()
        raise ValueError(f"missing model scores for {missing}")

    data["target"] = data["target"].str.upper()
    data["gc_fraction"] = data["target"].map(
        lambda seq: (seq.count("G") + seq.count("C")) / len(seq)
    )
    data["gc_balance_score"] = (100.0 - 200.0 * (data["gc_fraction"] - 0.5).abs()).clip(0, 100)
    data["complexity_score"] = data["target"].map(entropy_score)
    data["max_homopolymer"] = data["target"].map(max_run)
    data["homopolymer_safety_score"] = data["max_homopolymer"].map(homopolymer_safety)
    data["max_at_run"] = data["target"].map(
        lambda seq: max(base_run(seq, "A"), base_run(seq, "T"))
    )
    data["at_run_risk"] = data["max_at_run"].map(
        lambda run: "高" if run >= 4 else ("中" if run == 3 else "低")
    )
    data["evidence_score"] = (data["optimized_n"].clip(0, 3) / 3.0 * 100.0)

    data["within_locus_score"] = 50.0
    for _, index in data.groupby(["cell_line", "locus_group"]).groups.items():
        index = list(index)
        values = data.loc[index, "predicted_efficiency_pct"]
        if len(index) == 1:
            ranked = pd.Series([50.0], index=index)
        else:
            ranked = (values.rank(method="average") - 1.0) / (len(index) - 1.0) * 100.0
        data.loc[index, "within_locus_score"] = ranked

    data["composite_score"] = (
        0.55 * data["within_locus_score"]
        + 0.20 * data["score_0_100"]
        + 0.10 * data["gc_balance_score"]
        + 0.10 * data["complexity_score"]
        + 0.05 * data["homopolymer_safety_score"]
    )
    data["priority"] = pd.cut(
        data["composite_score"],
        bins=[-np.inf, 45, 65, 80, np.inf],
        labels=["D 低优先", "C 备选", "B 较高", "A 高优先"],
        right=False,
    ).astype(str)
    data["prediction_note"] = np.where(
        data["cell_line"].eq("N2a"),
        "293T模型外推到N2a；仅用于相对排序",
        np.where(
            data["oof_prediction"].notna(),
            "已有严格OOF回测；当前模型分来自全量拟合",
            "未实测；293T全量拟合模型预测",
        ),
    )

    output_columns = [
        "guide_id",
        "locus_group",
        "cell_line",
        "tam5",
        "target",
        "tam3",
        "sequence_20",
        "wt_efficiency",
        "wt_n",
        "optimized_efficiency",
        "optimized_n",
        "optimized_std",
        "predicted_efficiency_pct",
        "prediction_interval_90_low_pct",
        "prediction_interval_90_high_pct",
        "score_0_100",
        "within_locus_score",
        "oof_prediction",
        "absolute_error",
        "gc_fraction",
        "gc_balance_score",
        "complexity_score",
        "max_homopolymer",
        "homopolymer_safety_score",
        "at_run_risk",
        "evidence_score",
        "composite_score",
        "priority",
        "prediction_note",
    ]
    data = data[output_columns].sort_values(
        ["cell_line", "locus_group", "composite_score"], ascending=[True, True, False]
    )

    def record(row: pd.Series) -> dict[str, object]:
        return {
            "guide_id": row.guide_id,
            "locus_group": row.locus_group,
            "cell_line": row.cell_line,
            "tam5": row.tam5,
            "target": row.target,
            "tam3": row.tam3,
            "sequence_20": row.sequence_20,
            "wt_efficiency": clean_number(row.wt_efficiency),
            "wt_n": int(row.wt_n),
            "optimized_efficiency": clean_number(row.optimized_efficiency),
            "optimized_n": int(row.optimized_n),
            "optimized_std": clean_number(row.optimized_std),
            "predicted_efficiency": float(row.predicted_efficiency_pct) / 100.0,
            "interval_low": float(row.prediction_interval_90_low_pct) / 100.0,
            "interval_high": float(row.prediction_interval_90_high_pct) / 100.0,
            "model_score": float(row.score_0_100),
            "within_locus_score": float(row.within_locus_score),
            "oof_prediction": clean_number(row.oof_prediction),
            "oof_absolute_error": clean_number(row.absolute_error),
            "gc_fraction": float(row.gc_fraction),
            "gc_balance_score": float(row.gc_balance_score),
            "complexity_score": float(row.complexity_score),
            "max_homopolymer": int(row.max_homopolymer),
            "homopolymer_safety_score": float(row.homopolymer_safety_score),
            "at_run_risk": row.at_run_risk,
            "evidence_score": float(row.evidence_score),
            "composite_score": float(row.composite_score),
            "priority": row.priority,
            "prediction_note": row.prediction_note,
        }

    records = [record(row) for _, row in data.iterrows()]
    top = sorted(records, key=lambda item: item["composite_score"], reverse=True)[:15]
    payload = {
        "records": records,
        "top": top,
        "summary": {
            "total": len(records),
            "count_293T": int((data["cell_line"] == "293T").sum()),
            "count_N2a": int((data["cell_line"] == "N2a").sum()),
            "optimized_measured": int(data["optimized_efficiency"].notna().sum()),
            "high_priority": int((data["priority"] == "A 高优先").sum()),
            "median_model_score": float(data["score_0_100"].median()),
            "median_predicted_efficiency": float(data["predicted_efficiency_pct"].median()) / 100.0,
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
