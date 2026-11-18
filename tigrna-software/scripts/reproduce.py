"""Recompute preprocessing/CV in a separate output directory; never overwrite supplied results."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['293t','full'], default='293t')
parser.add_argument('--n-jobs', type=int, default=2)
parser.add_argument('--output', type=Path, default=ROOT/'run_results/reproduction')
args = parser.parse_args()
out = args.output.resolve()
if out.exists():
    raise FileExistsError('Choose a new --output directory; existing results will not be overwritten')
if out == ROOT or ROOT in out.parents and not (ROOT/'run_results') in out.parents:
    raise ValueError('Write inside run_results or an external, new directory')
out.mkdir(parents=True)
shutil.copytree(ROOT/'data/raw', out/'data/raw')
(out/'models').mkdir()
(out/'reports').mkdir()
env = os.environ.copy()
env['PYTHONPATH'] = str(ROOT/'src')
for name,folder in [('MPLCONFIGDIR','matplotlib'),('JOBLIB_TEMP_FOLDER','joblib'),('TMPDIR','tmp')]:
    location=out/'cache'/folder
    location.mkdir(parents=True,exist_ok=True)
    env[name]=str(location)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    env[name] = '1'
timings = {}
def run(label, command):
    before=time.perf_counter()
    with (out/f'{label}.log').open('w',encoding='utf-8') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    timings[label]=time.perf_counter()-before
    print(label, f'{timings[label]:.2f}s', flush=True)
run('preprocess',[sys.executable,'-m','tigrna_model.preprocess','--project-root',str(out)])
if args.mode == 'full':
    run('pooled_training',[sys.executable,'-m','tigrna_model.train','--project-root',str(out),'--n-jobs',str(args.n_jobs)])
run('293t_training',[sys.executable,'-m','tigrna_model.train_293t','--project-root',str(out),'--n-jobs',str(args.n_jobs)])
import numpy as np
import pandas as pd
comparisons={}
for name in ['oof_predictions_293T.csv','within_locus_ranking_293T.csv']+(['oof_predictions.csv','model_comparison.csv'] if args.mode=='full' else []):
    actual=pd.read_csv(out/'reports'/name)
    expected=pd.read_csv(ROOT/'reports'/name)
    assert actual.shape == expected.shape and list(actual)==list(expected), name
    max_difference=0.0
    for col in expected:
        if pd.api.types.is_numeric_dtype(expected[col]):
            np.testing.assert_allclose(actual[col],expected[col],rtol=1e-7,atol=1e-8,equal_nan=True,err_msg=f'{name}:{col}')
            difference=(actual[col]-expected[col]).abs().max()
            if pd.notna(difference): max_difference=max(max_difference,float(difference))
        else:
            assert actual[col].fillna('').tolist() == expected[col].fillna('').tolist(), (name,col)
    comparisons[name]={'matches':True,'max_absolute_difference':max_difference}
report={'mode':args.mode,'n_jobs':args.n_jobs,'timings_seconds':timings,'comparisons':comparisons}
(out/'reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
