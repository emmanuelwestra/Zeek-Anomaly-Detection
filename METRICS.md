# Reference metrics

This file is maintained by hand. The CLI writes machine-readable evidence to `artifacts/reports/`; it does not generate or overwrite this narrative.

## What is currently verified

The fixed corpus contains 865,941 benign training rows, 45,576 benign chronological validation rows, and 12,068,782 later rows. Development is pre-2025; holdout begins on 2025-01-01.

The grouped cross-entropy AE settings were selected on development across seeds 7, 42, and 123. The selected values were a 0.50 numerical training weight and a 0.10 numerical scoring weight.

The following comparison uses analyst labels from the development period to select, for each model, the threshold with the highest precision subject to recall ≥99.6%. Each selected development threshold is then frozen and applied unchanged to the 2025 holdout. This simulates periodic threshold adjustment from older analyst-labeled benign and attack traffic while preserving a later evaluation period.

Threshold selection is label-aware and retrospective. The development rows are in-sample operating-point results; the holdout rows measure threshold transfer. Holdout labels are used only to score the frozen thresholds, not to select them.

| Model | Period | Rows | Threshold | Accuracy | Precision | Recall | F1 | MCC | AUROC | AUPRC | FPR | TP | TN | FP | FN | Alerts / 10k | Window precision | Window recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OC-SVM | Development selection | 9,050,126 | -145.405105 | 91.2126% | 77.6401% | 99.6000% | 0.8726 | 0.8217 | 0.9498 | 0.8529 | 12.4186% | 2,723,419 | 5,531,440 | 784,330 | 10,937 | 3,875.9 | 35.7320% | 97.6271% |
| AE | Development selection | 9,050,126 | 0.0428372994 | 79.4145% | 59.5205% | 99.6110% | 0.7452 | 0.6455 | 0.9802 | 0.9450 | 29.3294% | 2,723,719 | 4,463,390 | 1,852,380 | 10,637 | 5,056.4 | 43.2135% | 99.8305% |
| Fusion (`alpha=0.30`) | Development selection | 9,050,126 | 8.426870655 | 92.9613% | 81.3078% | 99.6014% | 0.8953 | 0.8529 | 0.9569 | 0.8348 | 9.9134% | 2,723,457 | 5,689,662 | 626,108 | 10,899 | 3,701.1 | 40.3927% | 97.6271% |
| OC-SVM | 2025 holdout (frozen) | 3,018,656 | -145.405105 | 96.5458% | 93.5383% | 99.9916% | 0.9666 | 0.9331 | 0.9972 | 0.9906 | 6.8924% | 1,507,545 | 1,406,842 | 104,143 | 126 | 5,339.1 | 12.5541% | 66.6667% |
| AE | 2025 holdout (frozen) | 3,018,656 | 0.0428372994 | 55.5612% | 52.9171% | 99.9960% | 0.6921 | 0.2436 | 0.9969 | 0.9927 | 88.7761% | 1,507,611 | 169,592 | 1,341,393 | 60 | 9,438.0 | 12.3644% | 65.5172% |
| Fusion (`alpha=0.30`) | 2025 holdout (frozen) | 3,018,656 | 8.426870655 | 99.6527% | 99.3177% | 99.9916% | 0.9965 | 0.9931 | 0.9977 | 0.9955 | 0.6854% | 1,507,545 | 1,500,628 | 10,357 | 126 | 5,028.4 | 12.5813% | 66.6667% |

At the 99.6% development recall floor, fusion has the best development precision (81.3078%), FPR (9.9134%), and alert rate (3,701.1 per 10,000). Relative to OC-SVM, it removes 158,222 development false positives (20.17%) while recording 38 additional true positives.

On holdout, the frozen fusion and OC-SVM thresholds record the same 1,507,545 true positives and 126 false negatives. Fusion removes 93,786 false positives (90.06%) relative to OC-SVM and reduces the alert rate from 5,339.1 to 5,028.4 per 10,000 events. AE records 66 fewer false negatives than fusion, but its 88.7761% FPR and 9,438.0 alerts per 10,000 events make this operating point operationally unusable. Hourly-window recall remains only 66.67% for fusion and OC-SVM, so near-perfect event recall does not establish reliable window-level localization.

The separately optimized holdout reports under `artifacts/reports/*_holdout_recall_0.996_operating_point.yaml` are holdout-oracle analyses: their thresholds use holdout labels and are retrospective ceilings, not evidence of prospective generalization. The actual recalls in those reports exceed 99.6% because deterministic thresholds cannot split tied-score groups.

An aggregate static OC-SVM evaluation across the full post-baseline corpus produced 99.7034% recall, 87.2362% precision, F1 0.9305, 7.9065% FPR, 618,823 false positives, and 17,395 effective support vectors at `gamma=0.02`, `nu=0.001`, and benign-validation `q=0.9995`. It is not placed in the period table because it combines development and holdout; mixing that scope with the rows above would be misleading.

## 99.6%-recall operating-point status

The fusion model uses the repository's 94-feature cross-entropy AE with `alpha=0.30`. Its 99.6%-recall threshold is `8.426870654816677`, selected from development labels by `operating-point --model fusion --period development --min-recall 0.996`. The corresponding OC-SVM and AE thresholds are `-145.40510450263335` and `0.042837299406528466`.

The source operating-point reports are:

- `artifacts/reports/ocsvm_development_recall_0.996_operating_point.yaml`
- `artifacts/reports/ae_development_recall_0.996_operating_point.yaml`
- `artifacts/reports/fusion_development_recall_0.996_operating_point.yaml`

The frozen-threshold holdout metrics above were recomputed from `artifacts/score_cache/holdout-*.parquet` with the repository's strict `score > threshold` rule. They are not the output of the standard `evaluate` command, which reads the promoted thresholds under `artifacts/thresholds/`. Fusion is the clear aggregate event-level winner under the 99.6% development recall-floor rule, but segment-level checks and future untouched data are still required before making a deployment claim.

## Throughput

Selected-pairing training and inference throughput must be measured with `ndr benchmark` on the publishing machine. Results from the MSE-AE architecture are not comparable and are therefore excluded. Report the median model-only inference events/second, the three training timings, the OC-SVM power exponent, the AE linear fit, support-vector counts, and the recorded hardware/software environment from:

- `artifacts/reports/benchmark.csv`
- `artifacts/reports/benchmark_fits.yaml`

Throughput is hardware-dependent. Feature transformation, model loading, and disk I/O are explicitly outside the timed region.

## Interpretation

- Use benign-validation quantile metrics when discussing a threshold calibrated without attack labels.
- Use the development-constrained/frozen-holdout comparison when discussing periodic recalibration from older analyst labels. Preserve a strict time boundary between the labeled tuning period and the evaluation period.
- Treat thresholds selected on holdout labels as retrospective oracle ceilings only.
- Prefer AUPRC to accuracy when comparing rankings on imbalanced traffic.
- Always pair precision with FPR and alerts per 10,000 events; precision changes with attack prevalence.
- Compare development and holdout separately. The current 2025 holdout is retrospective because it was inspected during earlier research.
