from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import feature_matrix, sequence_from_parts


ALIASES = {
    "id": ["id", "guide_id", "name"],
    "tam5": ["tam5", "5tam", "5_tam", "5'tam", "five_prime_tam"],
    "target": ["target", "target_sequence", "sequence"],
    "tam3": ["tam3", "3tam", "3_tam", "3'tam", "three_prime_tam"],
}


def _normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    lookup = {str(column).strip().lower().replace(" ", "_"): column for column in frame.columns}
    result: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                result[canonical] = lookup[alias]
                break
    return result


def _similarity(sequence: str, training_sequences: list[str]) -> tuple[float, int]:
    identities = [sum(a == b for a, b in zip(sequence, candidate)) / 20.0 for candidate in training_sequences]
    index = int(np.argmax(identities))
    return float(identities[index]), index


def score_records(frame: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    columns = _normalized_columns(frame)
    missing = [name for name in ["tam5", "target", "tam3"] if name not in columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    ids = (
        frame[columns["id"]].astype(str).tolist()
        if "id" in columns
        else [f"candidate_{index + 1}" for index in range(len(frame))]
    )
    sequences = [
        sequence_from_parts(row[columns["tam5"]], row[columns["target"]], row[columns["tam3"]])
        for _, row in frame.iterrows()
    ]
    X, _ = feature_matrix(sequences)
    prediction = np.clip(bundle["estimator"].predict(X), 0.0, 1.0)
    radius = float(bundle["conformal_radius_90"])
    reference = np.asarray(
        bundle.get("score_reference_predictions", bundle["training_label_distribution"]),
        dtype=float,
    )
    training_sequences = list(bundle["training_sequences"])
    training_ids = list(bundle["training_guide_ids"])
    rows = []
    for guide_id, sequence, value in zip(ids, sequences, prediction):
        identity, nearest_index = _similarity(sequence, training_sequences)
        score = 100.0 * np.searchsorted(reference, value, side="right") / len(reference)
        rows.append(
            {
                "id": guide_id,
                "tam5": sequence[0],
                "target": sequence[1:19],
                "tam3": sequence[19],
                "sequence_20": sequence,
                "predicted_efficiency_pct": value * 100.0,
                "prediction_interval_90_low_pct": max(0.0, value - radius) * 100.0,
                "prediction_interval_90_high_pct": min(1.0, value + radius) * 100.0,
                "score_0_100": score,
                "nearest_training_guide": training_ids[nearest_index],
                "nearest_sequence_identity_pct": identity * 100.0,
                "model_family": bundle["model_family"],
                "model_status": bundle.get("model_status", "experimental"),
                "calibration_context": bundle.get("calibration_context", "pooled research model"),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) > 1:
        result["batch_rank_0_100"] = (
            result["predicted_efficiency_pct"].rank(method="average", pct=True) * 100.0
        )
    else:
        result["batch_rank_0_100"] = pd.Series([None], dtype="object")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score tigRNA target sequences")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--id", default="candidate_1")
    parser.add_argument("--tam5", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--tam3", default=None)
    args = parser.parse_args()

    default_293t = args.project_root / "models" / "tigrna_efficiency_model_293T.joblib"
    fallback_pooled = args.project_root / "models" / "tigrna_efficiency_model.joblib"
    model_path = args.model or (default_293t if default_293t.exists() else fallback_pooled)
    bundle = joblib.load(model_path)
    if args.input:
        frame = pd.read_csv(args.input)
        scored = score_records(frame, bundle)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.output, index=False)
            print(f"saved {len(scored)} predictions to {args.output}")
        else:
            print(scored.to_csv(index=False))
        return
    if not all([args.tam5, args.target, args.tam3]):
        parser.error("provide --input CSV or all of --tam5, --target, --tam3")
    frame = pd.DataFrame(
        [{"id": args.id, "tam5": args.tam5, "target": args.target, "tam3": args.tam3}]
    )
    scored = score_records(frame, bundle)
    print(json.dumps(scored.iloc[0].to_dict(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
