# Static Network-Flow Anomaly Detection

This repository trains two independent anomaly detection models from one attack-free week of UWF Zeek flows:

- an RBF one-class support vector machine (OC-SVM); and
- a grouped cross-entropy autoencoder (AE) for numerical and categorical flow features.

It also evaluates a transparent benign-tail fusion of their anomaly scores. There is no incremental learning, automatic retraining, dashboard, TCN, or hidden production policy here.

## What the experiment means

The earliest seven observed days are the trusted baseline. Rows are ordered by `(ts, uid)`; the first 95% train the preprocessors and models, and the final 5% calibrate benign-only alert thresholds. All later pre-2025 rows form the development period. Rows from 2025 onward form a locked retrospective holdout.

Labels and identifiers never enter either model. Labels are used only to calculate metrics, search research hyperparameters on development data, and answer retrospective questions such as “what threshold would have met 99% precision?” Such label-aware thresholds cannot be selected in an unlabeled live environment.

## Requirements

- Python 3.14
- `wget`
- About 2 GB for the 50 source Parquet shards, plus working space for prepared data and score caches
- Substantial CPU time for exact RBF OC-SVM fitting and scoring; the compact tests finish in seconds

Create the environment from this directory:

```bash
python3.14 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools, use `python -m pip install -e '.[dev]'`. A flat `requirements.txt` is also provided for environments that prefer it.

## Download the exact UWF inputs

The URL list contains exactly the 50 primary Parquet shards used by this experiment. It excludes CSV subsets, PCAPs, and metric sidecars.

```bash
wget --continue --input-file=data/uwf_urls.txt \
  --directory-prefix=data/raw \
  --force-directories --no-host-directories --cut-dirs=1
```

The resulting layout is `data/raw/UWF-ZeekData*/parquet/<week>/part-....parquet`. Re-running the command resumes incomplete downloads.

The files come from the [UWF Network Traffic Flow Datasets](https://datasets.uwf.edu/). UWF describes the data as MITRE ATT&CK-labelled Zeek traffic and distributes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the appropriate UWF paper listed on the dataset page when publishing results.

## End-to-end quickstart

All paths are resolved from the repository root, so commands work regardless of the current shell directory.

```bash
# 1. Validate, normalize, deduplicate, and split the downloaded corpus.
ndr prepare

# 2. Fit both models and their attack-free validation thresholds.
ndr train --model all

# 3. Evaluate deployment-like standalone thresholds.
ndr evaluate --model ocsvm --period development
ndr evaluate --model ae --period development

# 4. Select alpha and a matched-recall threshold on development labels.
ndr search --model fusion

# 5. Apply the frozen fusion choice to the retrospective 2025 holdout.
ndr evaluate --model fusion --period holdout
```

## Models and features

Both detectors start from six traffic-volume fields, six connection categories, IP direction, port roles, and derived rate/ratio features. Raw identifiers, IP addresses, labels, timestamps, raw duration, and raw ports are excluded from the final matrices.

The OC-SVM matrix has 31 numeric and 52 one-hot columns (83 total). Its promoted settings are `kernel=rbf`, `gamma=0.02`, `nu=0.001`, with a benign validation threshold at `q=0.9995`.

The AE matrix adds one explicit unknown level to each of the 11 categorical groups (94 columns total). Its encoder is `256 → 128 → 64 → 32`; the decoder mirrors it and ends in one numerical head plus 11 categorical heads. Training uses 50% numerical MSE and 50% grouped categorical cross-entropy. Scoring uses 10% numerical error and 90% categorical error.

## Focused searches

Searches use development labels and therefore produce research choices, not benign-only deployment calibration.

```bash
ndr search --model ocsvm
ndr search --model ae
ndr search --model fusion
```

- OC-SVM: six `(gamma, nu)` pairs and three nearby benign threshold quantiles.
- AE: nine trained networks across three training weights and three seeds; each network is evaluated at three scoring weights. The architecture and cross-entropy loss stay fixed.
- Fusion: `alpha=0.000, 0.025, …, 1.000`, where alpha is the OC-SVM weight. It minimizes false positives while preserving the development OC-SVM true-positive count. Ties prefer precision, F1, then the larger OC-SVM weight.

For each detector `m`, validation scores are converted to a finite empirical tail strength:

```text
p_m(x) = (1 + count(validation_score_m >= score_m(x))) / (n + 1)
t_m(x) = -log(p_m(x))
fusion(x) = alpha * t_ocsvm(x) + (1 - alpha) * t_ae(x)
```

The fusion alpha and threshold are frozen before holdout evaluation. If model artifacts change, rerun the fusion search.

## Retrospective precision or recall constraints

These commands use labels from the requested period. They are useful for analysis and capacity planning, but the returned threshold is not available prospectively in a live NDR system.

```bash
ndr operating-point --model ocsvm --period holdout --min-precision 0.99
ndr operating-point --model ae --period holdout --min-recall 0.998
ndr operating-point --model fusion --period holdout --min-recall 0.998
```

With `--min-precision`, the command maximizes recall. With `--min-recall`, it maximizes precision. It fails rather than silently relaxing an infeasible constraint. Precision depends on attack prevalence, so always interpret it with false-positive rate and alerts per 10,000 events.

## Throughput benchmark

```bash
ndr benchmark
```

The benchmark trains each architecture on 100k, 200k, and 400k rows. It fits `seconds = a × rows^b` for OC-SVM and a linear curve for AE training. Frozen-model inference uses 10k, 100k, and 1M rows, one warmup, and three timed repetitions. Timings exclude model loading and feature transformation. Results include environment versions and OC-SVM support-vector counts.

This is intentionally expensive. For a smoke run:

```bash
ndr benchmark --training-rows 1000 2000
```

## Outputs

```text
artifacts/
  models/          # generated locally; ignored by git
  preprocessors/   # generated locally; ignored by git
  thresholds/      # generated locally; ignored by git
  score_cache/     # identity-aligned Parquet; ignored by git
  reports/         # compact CSV/YAML evidence intended for publication
data/
  raw/             # downloads; ignored by git
  processed/       # chronological corpus; ignored by git
```

Score caches are keyed by `(uid, ts)` and are rejected if identities are missing or duplicated. Evaluation reports share the same metric names for OC-SVM, AE, and fusion. See [METRICS.md](METRICS.md) for the manually curated reference results and interpretation.

## Testing

```bash
pytest
```

The tests use tiny synthetic Parquet files. They cover aliases, canonical deduplication, the exact chronological split, attack-free baseline enforcement, feature contracts, unknown categories, both model round trips, tied-score thresholds, fusion, constrained operating points, search grids, scaling fits, and the CLI workflow.

## Common problems

- **Missing source file:** rerun the `wget --continue` command and check that `--cut-dirs=1` was used.
- **Baseline contains attacks:** do not suppress the error. Confirm that all 50 manifest files were downloaded and that their labels/schema were not modified.
- **Unexpected feature width:** delete locally generated preprocessors and models, then retrain. Never mix the 83-column OC-SVM preprocessor with the 94-column AE.
- **Fusion artifact missing:** train both models, then run `ndr search --model fusion`.
- **Out of memory:** lower `data.memory_limit` and `data.threads` in `configs/default.yaml`; do not change the chronological split.

## Reproducibility and limitations

- Random seeds and grids are explicit in YAML.
- The 2025 data was inspected during model development. It is a locked retrospective holdout, not a pristine unseen test.
- Benign-tail values saturate beyond the validation maximum. This keeps scores finite but creates ties at extreme values.
- Exact RBF OC-SVM runtime grows much faster than AE runtime as training rows increase.
- No fitted binaries or raw UWF data are distributed. Published reports must name their config, period, threshold mode, and runtime environment.
