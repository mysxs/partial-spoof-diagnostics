# Reading the released results

The files are grouped by measurement setup. Percentages use a 0–100 scale. Lower EER and higher F1/recall are better. The released per-seed values are the source of the summaries.

## PS localization audit

[`results/seed_results.csv`](../results/seed_results.csv) has one row per split × model × seed (24 rows). [`results/summary.json`](../results/summary.json) holds means and **sample** standard deviations, model identities, and scope notes. Individual [`results/dev_*.json`](../results/) and [`results/eval_*.json`](../results/) files retain selected epochs, development-selected thresholds, counts, and SHA-256 identities. The audit covers 24,844 PS development and 71,237 PS evaluation utterances using the original labels and split lists.

| Split | Model | Frame EER (%) ↓ | Frame F1 (%) ↑ | Event F1 (%) ↑ |
| --- | --- | ---: | ---: | ---: |
| Development | Hard-target frame | 5.330 ± 0.127 | 93.889 ± 0.144 | 69.472 ± 0.757 |
| Development | Soft-target frame | 5.565 ± 0.202 | 93.623 ± 0.229 | 67.961 ± 0.775 |
| Development | Phone units | 6.203 ± 0.176 | 92.898 ± 0.200 | 63.435 ± 0.107 |
| Development | Word units | 7.033 ± 0.096 | 91.958 ± 0.109 | 61.968 ± 0.333 |
| Evaluation (retrospective) | Hard-target frame | 11.123 ± 0.020 | 85.945 ± 0.010 | 55.479 ± 0.505 |
| Evaluation (retrospective) | Soft-target frame | 11.326 ± 0.132 | 85.657 ± 0.135 | 54.199 ± 0.572 |
| Evaluation (retrospective) | Phone units | 11.237 ± 0.224 | 86.030 ± 0.314 | 49.888 ± 0.394 |
| Evaluation (retrospective) | Word units | 12.514 ± 0.182 | 84.624 ± 0.311 | 48.169 ± 0.559 |

Frame F1/recall uses development-selected thresholds. Event proposals are positive spans on the **20-ms** prediction grid, matched one to one against original **10-ms** reference events at IoU ≥ 0.5. Short-event recall is for reference events below 100 ms. Boundary MAE is conditional on **matched** events. The CSV/JSON also report frame recall, event precision/recall, short-event recall, and boundary MAE. See [`audit_localization_table.py`](../scripts/audit_localization_table.py) and [`event_strata.py`](../scripts/event_strata.py) for the exact implementation.

The evaluation rows audit frozen outputs after the study was completed. They are not a prospectively held-out model-selection result. The audit verifies checkpoint, prediction, history, configuration, label, list, and metric-code hashes and checks the stored development threshold against evaluation metadata.

## Other comparisons

| Record | Contents | Scope |
| --- | --- | --- |
| [`MATCHED_CONTROLS_SUMMARY.json`](../MATCHED_CONTROLS_SUMMARY.json) | 18 per-seed frame/phone × original/augmentation/consistency runs; counts, thresholds, metrics, input hashes | Numerical summaries; no public checkpoints/predictions |
| [`MATCHED_CONTROLS_INTERACTION.json`](../MATCHED_CONTROLS_INTERACTION.json) | Speaker-bootstrap interaction contrasts | Read its `limitations` field with the estimates |
| [`E4_summary_20260919.csv`](../E4_summary_20260919.csv) | Source-locked transfer/intervention comparisons | Distinct arms and datasets; check each row's selection field |
| [`RECFPRF_SOURCE_CHECK.json`](../RECFPRF_SOURCE_CHECK.json) | reCFPRF source commit/file hashes, adapter hash, official-kernel parity check | Five RangeEER duration checks are not a full continuous-reference RangeEER result |

[`PROVENANCE.md`](../PROVENANCE.md) explains these records. Negative results, including the weaker word-unit branch, remain visible.
