# Reproducibility guide

These commands assume execution at the repository root and Python 3. This guide separates checks possible with a fresh clone from those requiring original experiment artifacts.

## 1. Verify the public package

```bash
python scripts/verify_manifest.py
python -m pip install -r requirements.txt
python -m pytest -q tests/test_event_strata.py
```

The manifest checker hashes every listed file. [`requirements.txt`](../requirements.txt) lists audit/test libraries; training and extraction also require a compatible CUDA PyTorch stack and upstream WavLM code. The manifest does not hash itself or the ZIP. The ZIP contains an earlier frozen source snapshot with its own embedded manifest; use the **direct GitHub tree** for current files.

## 2. Prepare official inputs

Keep the official PartialSpoof lists and labels unchanged. The localization audit reads these paths below `--data-root`:

```text
database/dev/dev.lst
database/eval/eval.lst
database/segment_labels/dev_seglab_0.01.npy
database/segment_labels/dev_seglab_0.02.npy
database/segment_labels/eval_seglab_0.01.npy
database/segment_labels/eval_seglab_0.02.npy
```

The `.npy` files are the original dictionaries keyed by utterance ID. The code checks list sizes (24,844 / 71,237), unique IDs, binary labels, prediction coverage, and array shapes. It retains the label-array tails. Feature extraction and training additionally need train/dev lists, 20-ms labels, audio, aligned-unit caches, and feature caches.

`--official-repo` must expose `sandbox/eval_asvspoof.py` with `compute_eer`. The audit hashes that implementation; preserve its upstream commit and hash in any new run record.

## 3. Recompute the frozen-prediction PS audit

The original run root uses these folders for seeds 1–3:

```text
frame_baselines_v2/temporal_seed{1,2,3}/
frame_soft_v2/temporal_seed{1,2,3}/
unit_models_v2/phone_seed{1,2,3}/
unit_models_v2/word_seed{1,2,3}/
eval_frames_v2/base/<training-folder>/<run-name>/predictions.npz
eval_units_v2/full/<training-folder>/<run-name>/predictions.npz
```

Training runs need `history.json`, `config.json`, `best.pt`, `complete.json`, and `dev_predictions.npz`; evaluation prediction directories need `complete.json`. The evaluator validates checkpoint/prediction identities and the development-locked threshold.

```bash
python scripts/audit_localization_table.py \
  --run-root /path/to/frozen/run \
  --data-root /path/to/verified/PartialSpoof \
  --official-repo /path/to/official/eer/repo \
  --out /path/to/new/audit/output
```

Output contains a JSON and per-utterance CSV per split/model/seed, plus `summary.json` and `seed_results.csv`. The supplied [summary](../results/summary.json) and [per-seed CSV](../results/seed_results.csv) allow comparison without private experiment files. This is a re-evaluation of frozen outputs, **not** retraining or evaluation-set threshold tuning.

## Selected training and feature entry points

The published source includes [`extract_wavlm.py`](../scripts/extract_wavlm.py), [`train_frame.py`](../scripts/train_frame.py), and [`train_units.py`](../scripts/train_units.py). Each has `--help`; running them requires CUDA, official labels, the official EER checkout, and prepared feature/unit caches. For example:

```bash
python scripts/train_frame.py \
  --root /path/to/verified/PartialSpoof \
  --features /path/to/wavlm/features \
  --official-repo /path/to/official/eer/repo \
  --out /path/to/new/frame/runs \
  --architectures temporal --seeds 1 2 3
```

Use a new output directory. The code selects checkpoints and thresholds on development data. Its header describes a **frozen-WavLM probe**, not every fine-tuned or proposal-refinement baseline in the paper.

## What is and is not released

| Available in GitHub | Needed for a full rerun but external to this release |
| --- | --- |
| Audit and selected training/extraction scripts; event tests | Official data/protocols and permitted access |
| Three-seed aggregate and per-seed PS localization records | Frozen checkpoints, predictions, feature/unit caches |
| Matched-control JSON, E4 CSV, reCFPRF hashes | Official EER implementation, model weights, alignment models, compatible CUDA environment |
| Integrity manifest | Exact original private run environment and all stage-specific commands |

The aggregate records can be inspected, source files verified, and the audit rerun when external frozen artifacts are available. A fresh clone does not independently regenerate every transfer/intervention record or paper baseline. Exact package versions, model checksums, dataset manifests, and commands should accompany newly executed runs; they are not inferred from aggregate files.

## Evaluation protocol

Keep the original train/development/evaluation partitions and labels fixed. Development data selects epochs and thresholds. PS evaluation localization is **retrospective**. External HAD and LlamaPartialSpoof E4 records use source-development selection as recorded there. Do not use test labels to tune models or omit negative outcomes.
