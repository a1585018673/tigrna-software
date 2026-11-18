# tigrna-software
A machine learning-based tigRNA efficiency predictor using support vector regression (SVR). It extracts 482 sequence features to estimate editing efficiency, rank candidate sequences, and support data preprocessing, model training, validation, and reproducibility.
# Experimental tigRNA efficiency predictor

Software version: 0.1.0.

This package contains the original model source, the original experimental workbook,
293T and N2a data, fitted models, out-of-fold predictions, and reproduction utilities.
The default predictor is a **293T-calibrated, sequence-only support-vector regressor
(SVR)** for the optimized 336-172 construct. A pooled 293T/N2a research model is also
included. Both models provide experimental sequence-based efficiency estimates and
relative candidate rankings.

## 1. System requirements

- Tested environment: Ubuntu 22.04.5 LTS, Linux x86-64, Python 3.12.3. Exact direct
  dependency versions are in `requirements.txt`. Environment lock files and system
  details are in `environment/`.
- Python >=3.11 is required by the package. The recorded execution tests used Python
  3.12.3 on Linux; Windows and macOS execution have not been tested. Windows
  installation commands are supplied below for users of that platform.
- Standard CPU hardware; no GPU, accelerator, genome index or external database is
  required. Recommended resources: 2 CPU cores and 4 GB RAM for prediction/default
  training, with more cores optional for the pooled four-family comparison.
- Internet is needed to install dependencies, not for prediction or training after
  installation. Only load the supplied trusted `.joblib` files; joblib/pickle files
  from untrusted sources can execute code.

## 2. Installation guide

Run from the extracted repository directory. Paths in commands are relative.

Linux/macOS:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install "setuptools>=68"
python -m pip install --no-build-isolation --no-deps -e .
```

Windows PowerShell:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install "setuptools>=68"
python -m pip install --no-build-isolation --no-deps -e .
```

The environment needs setuptools for the editable install. Alternatively, after
installing requirements, `scripts/demo.py` and `scripts/reproduce.py` work directly
without an editable install. Do not install into the operating system's Python.

If the default package index stalls, select a reachable trusted index explicitly,
for example:

```bash
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 30 --retries 1 -r requirements.txt
```

Typical installation time is approximately 2-10 minutes with a working package mirror
(an estimate). The measured dependency-installation stage took 407.29 seconds on the
reference instance, excluding earlier network retries. Installation and execution
timings are recorded in `reports/fresh_install_report.json`.

## 3. Demo

```bash
python scripts/demo.py
python -m unittest discover -s tests -v
```

The demo uses `examples/candidates.csv` and writes
`run_results/demo_predictions.csv`. It compares every numeric prediction with
`examples/expected_predictions_293T.csv` (tolerance 1e-8). These are real example
sequences for demonstrating the interface, **not an independent test set**.
Expected output is a CSV containing efficiency (%), empirical residual bounds,
0-100 relative score, nearest-training-sequence identity, and within-batch rank.
The demo process took 3.77 seconds in the reference environment and 14.68 seconds in
the newly installed environment on the CPU-limited test instance. Runtime includes
interpreter and model-loading overhead and varies with the available CPU resources.

## 4. Instructions for use

Batch prediction:
```bash
python -m tigrna_model.score --project-root . --input examples/candidates.csv --output run_results/predictions.csv
```

Single prediction:
```bash
python -m tigrna_model.score --project-root . --id demo_1 --tam5 G --target GACATTGGAGGTACCCGT --tam3 A
```

Input columns: `id,tam5,target,tam3`. `target` must contain 18 DNA bases, and each
TAM/flanking field one base. All values are oriented in the target's 5'-to-3'
direction. U is normalised to T by the legacy validator. Non-ACGT symbols fail.
The predictor itself accepts all flank bases. The later experimental policy
excluding 5' A or 3' T is a separate design filter, not an intrinsic model claim.

The default model is `models/tigrna_efficiency_model_293T.joblib`. The optional
pooled 293T/N2a research model can be selected with
`--model models/tigrna_efficiency_model.joblib`. This pooled model is not an
independently validated N2a-specific predictor.

For preprocessing new training data, the workbook schema is: guide ID in column A;
5' flank, 18nt target and 3' flank in columns B-D; WT measurement slots in E-G; and
optimized-construct measurement slots in H-J. Efficiency values are fractions in
[0,1], and worksheet names identify the cell lines. Run:

```bash
python -m tigrna_model.preprocess --project-root . --xlsx path/to/experiments.xlsx
```

Use a separate project directory for new experimental datasets. Calibration is
specific to the protein, scaffold and experimental context used for training.

## 5. Reproduction

Default 293T preprocessing and nested-CV retraining:
```bash
python scripts/reproduce.py --mode 293t --n-jobs 2
```

Full pooled-family comparison plus default-model retraining:
```bash
python scripts/reproduce.py --mode full --n-jobs 2 --output run_results/full_reproduction
```

Outputs go into a new directory; supplied `models/` and `reports/` are not overwritten.
The script compares reproduced numeric OOF tables with the supplied reference files
and fails on disagreement. Runtime depends strongly on parallelism and library
versions. The completed default 293T reproduction took 69.90 seconds for training
and evaluation, plus 3.08 seconds for preprocessing, using one worker on an instance
with an approximately 0.5-CPU allocation. Both reproduced comparison tables matched
the original numeric results exactly; see `reports/reproduction_report.json`.

The saved pooled-model OOF metrics and final selected SVR fit were verified
separately. The full four-family pooled nested-CV search was not rerun to completion
in that verification; its original results are retained in `reports/`.

Additional verification commands:

```bash
python scripts/verify_pooled_reference.py
python scripts/check_manifest.py
```

## 6. Package contents and scientific limits

- `src/tigrna_model/`: prediction, sequence features, training, metrics and preprocessing.
- `data/raw/`: unchanged source workbook; `data/processed/`: all records, training
  subsets by cell line and long-format replicate slots (including missing values).
- `models/`: original fitted default and pooled models and their metadata.
- `reports/`: original nested-CV metrics and OOF predictions, not training-set fits.
- `examples/`: small input, expected result and historical 154-prediction regression fixture.
- `tests/`, `scripts/`: tests, demo and reproduction commands.
- `legacy/`: sequence enumeration and historical composite-score utilities.
- `environment/`: tested system information and dependency lock files.
- `verification/reproduced_reports/`: independently reproduced default-model results.
- `MANIFEST.sha256`: file-integrity checksums; run `python scripts/check_manifest.py`.

There are 154 design records (123 in 293T and 31 in N2a), of which 125 have an
optimized-construct label: 98 from 35 loci in 293T and 27 from one N2a locus.
Missing labels are not zeros.
The 293T nested-group-CV MAE is about 11.06 percentage points and Spearman about
0.311; the median baseline MAE is about 11.29 points. These results are modest,
not evidence of high-accuracy absolute prediction. The original 90% absolute-residual
radius is about 24.83 points; it is an empirical OOF residual summary and does not
guarantee 90% coverage on a new experiment. Model-specific results and limitations
are described in `reports/model_card_293T.md` and `reports/model_card.md`.

N2a contributes 27 labelled guides from a single PCSK locus. These data do not
establish generalization to other N2a loci or calibration across cell lines.

No raw sequencing reads, read alignments or indel-calling pipeline are supplied in
the available source materials. This package reproduces modelling from the workbook,
not the upstream generation of the experimental efficiency measurements.

## 7. Model and data definitions

Each input is represented by 482 features: 80 position-specific base indicators,
304 position-specific adjacent-dinucleotide indicators, four target base counts,
16 target dinucleotide frequencies, 64 target trinucleotide frequencies and
14 scalar sequence descriptors. The exact definitions and feature order are in
`src/tigrna_model/features.py`.

The default model uses StandardScaler followed by an RBF-kernel SVR. Its selected
parameters are C=0.1, epsilon=0.02 and gamma='scale'. Training sample weights are the
square root of the number of available optimized-construct measurement slots.
The primary label is their nonmissing arithmetic mean; explicit numerical zeros
are retained. Standard deviation uses ddof=1 when at least two measurements exist.

Default validation uses five outer locus-group folds and four inner group folds.
The final default model is selected with five group folds over the 293T training
data. The same locus is not split between a training fold and its test fold.
OOF predictions are held-out predictions; full-data fitted predictions and demo
outputs are not independent validation results.

Key processed data files:

| File in `data/processed/` | Contents |
| --- | --- |
| `all_experimental_rows.csv` | All 154 designs, including 29 without an optimized label |
| `training_dataset.csv` | All 125 labelled designs in workbook order |
| `all_rows_293T.csv`, `all_rows_N2a.csv` | All-design exports by cell line: 123 and 31 rows |
| `training_293T.csv`, `training_N2a.csv` | Labelled exports by cell line: 98 and 27 rows |
| `replicate_measurements.csv` | 924 measurement slots with source row/column and missing-value flags |
| `dataset_summary.json` | Counts and the source-workbook SHA256 |

`guide_id` is the source identifier; `cell_line` is the worksheet name; `locus_group`
removes a trailing `-tig<number>` from that identifier. `excel_row` is one-based.
`wt_efficiency` and `optimized_efficiency` are mean fractions, `_std` columns are
sample standard deviations, and `_n` columns count available measurements.
Measurement slots do not themselves establish biological-replicate independence.
The long-format export preserves blank slots explicitly, rather than converting
them to zeros. The original workbook is the measurement source of record.

For the default model, `score_0_100` is a percentile relative to its training-set
model predictions. The pooled model uses its training-label distribution as the
legacy score reference. Neither score is a success probability. `batch_rank_0_100`
is an average-rank percentile within the supplied batch and is unavailable for a
singleton. Nearest-sequence identity is positional 20nt identity against the training
set, not a genome-wide off-target score.

The historical composite score combines fixed heuristic weights: 55% locus-relative
rank, 20% model percentile, 10% GC balance, 10% sequence complexity and 5% homopolymer
safety. These coefficients are not trained parameters or calibrated efficiency
estimates. The supplied model does not include chromatin, expression or off-target
features, and it does not use the constant protein sequence as a predictive input.



No software or data redistribution licence is specified in this distribution.
Third-party dependencies retain their respective upstream licences.
