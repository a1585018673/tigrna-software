from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .features import sequence_from_parts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_replicates(row: pd.Series, columns: range) -> list[float]:
    values: list[float] = []
    for column in columns:
        value = pd.to_numeric(row.iloc[column], errors="coerce")
        if pd.notna(value):
            values.append(float(value))
    return values


def _summary(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return np.nan, np.nan, 0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return mean, std, len(values)


def load_experiment_workbook(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    records: list[dict[str, object]] = []
    for sheet_name, sheet in sheets.items():
        if sheet.shape[1] < 10:
            raise ValueError(f"sheet {sheet_name!r} has fewer than 10 columns")
        for excel_row, (_, row) in enumerate(sheet.iloc[1:].iterrows(), start=2):
            guide_id_raw = row.iloc[0]
            if pd.isna(guide_id_raw) or not str(guide_id_raw).strip():
                continue
            guide_id = str(guide_id_raw).strip()
            tam5 = str(row.iloc[1]).strip()
            target = str(row.iloc[2]).strip()
            tam3 = str(row.iloc[3]).strip()
            sequence_20 = sequence_from_parts(tam5, target, tam3)
            wt_values = _numeric_replicates(row, range(4, 7))
            optimized_values = _numeric_replicates(row, range(7, 10))
            wt_mean, wt_std, wt_n = _summary(wt_values)
            opt_mean, opt_std, opt_n = _summary(optimized_values)
            locus = re.sub(r"-tig\d+$", "", guide_id, flags=re.IGNORECASE)
            records.append(
                {
                    "guide_id": guide_id,
                    "cell_line": str(sheet_name),
                    "locus_group": locus,
                    "excel_row": excel_row,
                    "tam5": tam5.upper(),
                    "target": target.upper(),
                    "tam3": tam3.upper(),
                    "sequence_20": sequence_20,
                    "wt_efficiency": wt_mean,
                    "wt_std": wt_std,
                    "wt_n": wt_n,
                    "optimized_efficiency": opt_mean,
                    "optimized_std": opt_std,
                    "optimized_n": opt_n,
                    "wt_replicates": ";".join(f"{value:.6g}" for value in wt_values),
                    "optimized_replicates": ";".join(
                        f"{value:.6g}" for value in optimized_values
                    ),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("no guide rows were found")
    duplicate_counts = frame.groupby("sequence_20")["sequence_20"].transform("size")
    frame["sequence_duplicate_count"] = duplicate_counts.astype(int)
    frame["primary_label_available"] = frame["optimized_n"] > 0
    frame["sample_weight"] = np.sqrt(frame["optimized_n"].clip(lower=1))
    return frame
