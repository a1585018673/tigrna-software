"""Run the supplied model on the real example CSV and compare expected predictions."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
from tigrna_model.score import score_records

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT/'run_results/demo_predictions.csv')
args = parser.parse_args()
start = time.perf_counter()
model = joblib.load(ROOT/'models/tigrna_efficiency_model_293T.joblib')
result = score_records(pd.read_csv(ROOT/'examples/candidates.csv'), model)
expected = pd.read_csv(ROOT/'examples/expected_predictions_293T.csv')
assert result.id.tolist() == expected.id.tolist()
for column in result.select_dtypes(include='number').columns:
    np.testing.assert_allclose(result[column].astype(float), expected[column].astype(float),
                               atol=1e-8, rtol=1e-8, equal_nan=True, err_msg=column)
args.output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(args.output, index=False)
print(json.dumps({'rows': len(result), 'expected_output_matches': True,
                  'elapsed_seconds': time.perf_counter()-start}, indent=2))
