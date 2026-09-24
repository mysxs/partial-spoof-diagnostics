# Partial Spoof Diagnostics

[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE) [![Release: research audit](https://img.shields.io/badge/release-research%20audit-0969da)](docs/REPRODUCIBILITY.md)

**Code and result records for _Separating Boundary Artifacts from Synthetic Content in Partial Speech Spoofing: A Controlled Study_.** This repository exposes the localization audit, selected training entry points, and numerical provenance as ordinary GitHub files. It retains negative findings and explains the limits of retrospective analysis.

| Start here | Contents |
| --- | --- |
| [Results guide](docs/RESULTS.md) | Three-seed localization table, metric definitions, links to raw rows |
| [Reproducibility guide](docs/REPRODUCIBILITY.md) | Input layout, commands, integrity checks, release scope |
| [Provenance](PROVENANCE.md) | reCFPRF source commit/checksums and matched-control records |
| [Citation](CITATION.cff) | Machine-readable repository citation |

> [!IMPORTANT]
> The PartialSpoof **evaluation** localization table is a retrospective audit of frozen predictions. Checkpoint and threshold selection used development data; evaluation labels were not used to retune them. This release provides audit code and selected training code, not a one-command reproduction of every paper experiment.

## Results at a glance

These values come from [`results/summary.json`](results/summary.json): **three seeds**, **last WavLM layer**, PS **evaluation** frozen-prediction audit. Values are mean ± sample standard deviation, in percent.

| Model | Frame EER ↓ | Frame F1 ↑ | Event F1 ↑ |
| --- | ---: | ---: | ---: |
**Code and result records for _Temporal Anchors and Editing Sensitivity in Partial Speech Spoofing: A Controlled Study_.** This repository exposes localization audits, controlled intervention results, training entry points, and numerical provenance as ordinary GitHub files. It retains negative findings and explains the limits of retrospective analysis.55.479 ± 0.505 |
| Soft-target frame | 11.326 ± 0.132 | 85.657 ± 0.135 | 54.199 ± 0.572 |
| Phone units | 11.237 ± 0.224 | 86.030 ± 0.314 | 49.888 ± 0.394 |
| Word units | 12.514 ± 0.182 | 84.624 ± 0.311 | 48.169 ± 0.559 |

The word-unit branch is retained even though it underperforms. E4 transfer/intervention and matched-control records have different setups; they should not be collapsed into this table. See the [results guide](docs/RESULTS.md) for development rows, per-seed values, matching rules, and limitations.

## Repository map

```text
scripts/                         extraction, training, and audit entry points
tests/                           deterministic event-matching tests
results/                         three-seed PS audit rows and summaries
docs/                            reproduction and result guides
MATCHED_CONTROLS_SUMMARY.json    18 per-seed matched-control records
MATCHED_CONTROLS_INTERACTION.json speaker-bootstrap contrasts and limitations
E4_summary_20260919.csv         source-locked transfer/intervention summary
RECFPRF_SOURCE_CHECK.json        reCFPRF source and adapter hashes
PROVENANCE.md                   human-readable source notes
MANIFEST.sha256.json            checksums for current direct release files
partial-spoof-diagnostics-source.zip  earlier frozen source snapshot
```

The key scripts are [`audit_localization_table.py`](scripts/audit_localization_table.py), [`event_strata.py`](scripts/event_strata.py), [`evaluate_event_strata.py`](scripts/evaluate_event_strata.py), [`extract_wavlm.py`](scripts/extract_wavlm.py), [`train_frame.py`](scripts/train_frame.py), and [`train_units.py`](scripts/train_units.py). The release does not contain every E4/cross-domain pipeline stage; its JSON/CSV files are the disclosed records for those comparisons.

## Quick start: inspect and verify

```bash
git clone https://github.com/mysxs/partial-spoof-diagnostics.git
cd partial-spoof-diagnostics
python scripts/verify_manifest.py
python -m pip install -r requirements.txt
python -m pytest -q tests/test_event_strata.py
```

The checksum command requires only Python's standard library. The tests cover event matching and duration/boundary strata; they do not run model inference. To recompute the PS localization table from verified official data and frozen experiment outputs, follow the exact layout and command in the [reproducibility guide](docs/REPRODUCIBILITY.md).

## Data, weights, and scope

The study uses PartialSpoof, HAD, LlamaPartialSpoof, WavLM, Whisper large-v3, and Montreal Forced Aligner components. Obtain the required third-party resources from their original providers. This release contains no audio, dataset labels, model weights, cached features, frozen predictions, or private execution paths. The published aggregate records can be inspected immediately; a full audit rerun requires the original frozen checkpoints/predictions and official data view. [Required inputs and limits](docs/REPRODUCIBILITY.md#what-is-and-is-not-released) are stated explicitly.

Original code is released under the [MIT license](LICENSE). If you use it, see GitHub's **Cite this repository** control or [`CITATION.cff`](CITATION.cff). Reproducibility questions can be filed under [Issues](https://github.com/mysxs/partial-spoof-diagnostics/issues) with the command and integrity-check output, without uploading restricted data.


## Reviewer revision audits (2026-09-24)

The [new result records](results/reviewer_revision_20260924) cover matched nonlinguistic anchor controls, a three-seed layer-6 localization audit, source-locked cross-corpus seed records, and the full operating-point curves ([PDF](results/reviewer_revision_20260924/operating_points.pdf), [CSV](results/reviewer_revision_20260924/operating_points.csv)). The [audit scripts](scripts/reviewer_revision_20260924) are also available.

The intervention-specific LlamaPartialSpoof A follow-up froze 200 constructed same-source-speaker cases from 40 speakers, nine existing frame checkpoints, and PartialSpoof-development thresholds before scoring. Its [protocol](results/reviewer_revision_20260924/protocol.json) and [paired speaker/seed statistics](results/reviewer_revision_20260924/statistics.json) retain both primary contrasts and the 50-ms smoothing result. These are constructed controls, not official Llama A benchmark metrics or a pristine blind corpus: broader Llama A transfer had already been inspected, and donor texts/generators were not matched. GG excess false alarms improve, while the synthetic-core miss-rate difference is uncertain; smoothing reverses the PartialSpoof direction. Negative and heterogeneous results are retained.
