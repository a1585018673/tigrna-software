"""Check pooled saved OOF metrics and final SVR refit; not a four-family CV rerun."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tigrna_model.features import feature_matrix
from tigrna_model.metrics import regression_metrics
data=pd.read_csv(ROOT/'data/processed/training_dataset.csv',float_precision='round_trip')
# Rank metrics are tie-sensitive: recover the original binary floats from the
# full-precision CSV, rather than using pandas' default fast float conversion.
oof=pd.read_csv(ROOT/'reports/oof_predictions.csv',float_precision='round_trip')
metrics=json.loads((ROOT/'reports/metrics.json').read_text())
assert data.guide_id.tolist()==oof.guide_id.tolist()
for family in ['ridge','svr_rbf','extra_trees','hist_gradient_boosting']:
    actual=regression_metrics(oof.optimized_efficiency.to_numpy(),oof[f'oof_{family}'].to_numpy())
    for name,val in actual.items():
        np.testing.assert_allclose(val,metrics['candidate_nested_group_cv'][family][name],rtol=1e-8,atol=1e-10,equal_nan=True)
actual=regression_metrics(oof.optimized_efficiency.to_numpy(),oof.oof_adaptive.to_numpy())
for name,val in actual.items():
    np.testing.assert_allclose(val,metrics['adaptive_nested_group_cv'][name],rtol=1e-8,atol=1e-10,equal_nan=True)
saved=joblib.load(ROOT/'models/tigrna_efficiency_model.joblib')
assert saved['model_family']=='svr_rbf'
params={k.removeprefix('model__'):v for k,v in saved['model_params'].items()}
refit=Pipeline([('scale',StandardScaler()),('model',SVR(kernel='rbf',**params))])
x,_=feature_matrix(data.sequence_20)
refit.fit(x,data.optimized_efficiency,model__sample_weight=data.sample_weight)
delta=float(np.max(np.abs(refit.predict(x)-saved['estimator'].predict(x))))
assert delta<1e-10
report={'saved_125_row_oof_metrics_verified':True,'selected_pooled_svr_refit_verified':True,
        'maximum_training_prediction_difference_fraction':delta,
        'full_four_family_nested_cv_rerun_in_this_bundle_check':False,
        'reason':'CPU-only instance has a 0.5 CPU cgroup quota; original four-family OOF results are retained, not relabelled as a new rerun.'}
(ROOT/'reports/pooled_reference_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
