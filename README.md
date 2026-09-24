# Partial-spoof diagnostics and reproducibility package

Repository: https://github.com/mysxs/partial-spoof-diagnostics

Companion code for *Separating Boundary Artifacts from Synthetic Content in Partial Speech Spoofing: A Controlled Study*.

This directory contains the experiment-side code used for the ICASSP study. It does **not** redistribute PartialSpoof, HAD, LlamaPartialSpoof, forced-alignment models, or WavLM weights. Obtain those resources from their official sources and follow their licenses.

## Contents

- `scripts/audit_localization_table.py`: recomputes PS development diagnostics and the retrospective PS evaluation table from frozen predictions. It never fits a threshold on PS evaluation.
- `scripts/event_strata.py`: deterministic 20-ms proposal construction and one-to-one IoU matching against the released 10-ms labels.
- `scripts/extract_wavlm.py`: feature extraction used by the frozen WavLM front end; supply local paths in the command line/configuration used for your run.
- `scripts/train_frame.py`, `scripts/train_units.py`: frame and aligned-unit training entry points.
- `scripts/evaluate_event_strata.py`: localization summaries.
- `tests/test_event_strata.py`: unit tests for event matching and duration bins.

## Reproducing the reported PS audit

Install the audit dependencies with `python -m pip install -r requirements.txt`. Training and feature extraction additionally require a compatible PyTorch/WavLM stack and separately obtained data and models. This initial release covers the PS localization audit and selected training entry points; it is not yet a complete end-to-end reproduction package for every experiment in the paper.

Prepare a verified PS protocol view containing the official train/dev/eval lists and labels, and retain frozen checkpoints and predictions. Then run:

```bash
python scripts/audit_localization_table.py \
  --run-root /path/to/frozen/run \
  --data-root /path/to/partialspoof/verified_protocol_view \
  --official-repo /path/to/official-eer-kernel \
  --out results/ps_localization
```

The audit writes one JSON and one per-utterance CSV for each arm/seed/split, together with `summary.json` and `seed_results.csv`. SHA-256 identities cover checkpoints, predictions, configs, histories, label arrays, ID lists, the matching code, and the official EER implementation. The PS evaluation rows are a retrospective audit of predictions produced in the completed study; they are not a prospectively held-out benchmark.

The experiment used Python 3, NumPy, SciPy, PyTorch, `s3prl`/WavLM-compatible code, Whisper large-v3, and Montreal Forced Aligner. Exact package versions, model checksums, dataset manifests, and run-specific commands must be recorded alongside each frozen run; private hosts and credentials are intentionally omitted from this package.

```bash
python -m pytest tests/test_event_strata.py
```

## Data and model provenance

Use the official publication/download pages cited by the paper. Do not commit raw datasets, speaker lists, model weights, cached features, predictions, credentials, or internal filesystem paths. Original code is released under the MIT license in `LICENSE`; third-party datasets and models retain their own licenses.
