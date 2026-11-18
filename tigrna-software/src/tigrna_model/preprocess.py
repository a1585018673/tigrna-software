"""Preprocess the experimental workbook without altering label definitions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .dataio import load_experiment_workbook, sha256_file


def prepare(root: Path, xlsx: Path | None = None) -> dict:
    candidates = sorted((root / 'data/raw').glob('*.xlsx'))
    if xlsx is None and len(candidates) != 1:
        raise ValueError('Expected one raw workbook, or pass --xlsx explicitly')
    source = xlsx or candidates[0]
    all_rows = load_experiment_workbook(source)
    for column in ['wt_efficiency', 'optimized_efficiency']:
        values = all_rows[column].dropna()
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f'{column} must contain fractions in [0,1]; no automatic unit conversion')
    output = root / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    (root / 'models').mkdir(exist_ok=True)
    (root / 'reports').mkdir(exist_ok=True)
    all_rows.to_csv(output / 'all_experimental_rows.csv', index=False)
    training = all_rows[all_rows.primary_label_available].reset_index(drop=True)
    training.to_csv(output / 'training_dataset.csv', index=False)
    summary = {}
    for line, frame in all_rows.groupby('cell_line', sort=False):
        measured = frame[frame.primary_label_available]
        frame.to_csv(output / f'all_rows_{line}.csv', index=False)
        measured.to_csv(output / f'training_{line}.csv', index=False)
        summary[line] = {'all_rows': len(frame), 'labelled_rows': len(measured),
                         'labelled_locus_groups': measured.locus_group.nunique(),
                         'missing_optimized_labels': int((~frame.primary_label_available).sum())}
    measurements = []
    sheets = pd.read_excel(source, sheet_name=None, header=None, engine='openpyxl')
    for _, row in all_rows.iterrows():
        raw = sheets[row.cell_line].iloc[int(row.excel_row) - 1]
        for construct, columns in [('WT', range(4, 7)), ('336-172', range(7, 10))]:
            for replicate, col in enumerate(columns, 1):
                value = pd.to_numeric(raw.iloc[col], errors='coerce')
                measurements.append({'guide_id': row.guide_id, 'cell_line': row.cell_line,
                    'excel_row': int(row.excel_row), 'excel_column': chr(65 + col),
                    'construct': construct, 'measurement_slot': replicate,
                    'efficiency_fraction': None if pd.isna(value) else float(value),
                    'missing': bool(pd.isna(value))})
    pd.DataFrame(measurements).to_csv(output / 'replicate_measurements.csv', index=False)
    report = {'raw_workbook_sha256': sha256_file(source), 'all_rows': len(all_rows),
              'training_rows': len(training), 'cell_lines': summary,
              'measurement_slots': len(measurements),
              'missing_labels_are_zero_imputed': False}
    (output / 'dataset_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--xlsx', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project_root.resolve(), args.xlsx), indent=2))


if __name__ == '__main__':
    main()
