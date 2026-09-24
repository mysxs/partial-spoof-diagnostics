# Provenance and raw result records

## reCFPRF

The released reCFPRF evaluation is source-locked to commit `847347aaec6f65c3c6d2f17c63515b826b94feb3`. The adapter SHA-256 is `19bbf127c496a2d5b9c3ca3a238117e1ecad1166712bc32e47d066d2f416364d`. The source-file SHA-256 values for `SegmentEER.py`, `RangeEER.py`, and `rttm_tool.py` are recorded in [`RECFPRF_SOURCE_CHECK.json`](RECFPRF_SOURCE_CHECK.json). The official 20-ms SegmentEER wrapper and kernel agree at `0.017704054563277845` on the 24,844-utterance PS development view. The supplied checkpoint was evaluated without retraining or its proposal-refinement stage.

The RangeEER entries in the check file are five duration checks only; the paper therefore does not claim a full continuous-reference RangeEER result.

## Matched controls

[`MATCHED_CONTROLS_SUMMARY.json`](MATCHED_CONTROLS_SUMMARY.json) contains the per-seed raw confusion counts, locked development thresholds, selected epochs, metrics, input hashes, and case counts for all 18 frame/phone × original/augmentation/consistency runs. Private checkpoint paths have been removed; numerical records are unchanged. [`MATCHED_CONTROLS_INTERACTION.json`](MATCHED_CONTROLS_INTERACTION.json) contains the speaker-bootstrap contrasts and limitations. [`E4_summary_20260919.csv`](E4_summary_20260919.csv) is the source-locked summary used for the transfer and intervention tables.
