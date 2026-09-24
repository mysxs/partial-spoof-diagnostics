"""Paired speaker/seed audit for the frozen Llama-A intervention follow-up."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rate(counts, kind):
    tp, fp, fn, tn = counts
    return (fp / (fp + tn)) if kind == "fpr" else (fn / (tp + fn))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["cases"] == 200 and protocol["speakers"] == 40
    threshold_names = ["dev_fpr_0.05", "dev_fpr_0.01", "original_dev_eer"]
    conditions = ["original", "augmentation", "consistency"]
    records = {}
    speakers = None
    for point in threshold_names:
        for condition in conditions:
            for seed in (1, 2, 3):
                path = args.evaluation / "frame" / f"{condition}_seed{seed}" / f"{point}_cases.json"
                cases = json.loads(path.read_text())
                assert len(cases) == 200
                current_speakers = sorted({c["speaker"] for c in cases})
                assert len(current_speakers) == 40
                if speakers is None:
                    speakers = current_speakers
                assert speakers == current_speakers
                matrix = np.zeros((len(speakers), 4, 4), dtype=np.int64)
                for case in cases:
                    si = speakers.index(case["speaker"])
                    for ci, name in enumerate(("genuine_full", "gg_raw_full", "gf_raw_core", "gf_smooth50_core")):
                        matrix[si, ci] += case["counts"][name]
                records[(point, condition, seed)] = matrix

    def summarize(point, seed_idx, speaker_idx):
        rows = {}
        for condition in conditions:
            counts = sum((records[(point, condition, seed)][speaker_idx].sum(axis=0)
                          for seed in seed_idx), np.zeros((4, 4), dtype=np.int64))
            rows[condition] = {
                "genuine_fpr": rate(counts[0], "fpr"),
                "gg_fpr": rate(counts[1], "fpr"),
                "gg_excess_fpr": rate(counts[1], "fpr") - rate(counts[0], "fpr"),
                "gf_core_fnr": rate(counts[2], "fnr"),
                "gf_smooth50_core_fnr": rate(counts[3], "fnr"),
            }
        contrasts = {
            "augmentation_minus_original_gg_excess_fpr": rows["augmentation"]["gg_excess_fpr"] - rows["original"]["gg_excess_fpr"],
            "augmentation_minus_original_gf_core_fnr": rows["augmentation"]["gf_core_fnr"] - rows["original"]["gf_core_fnr"],
            "consistency_minus_augmentation_gg_excess_fpr": rows["consistency"]["gg_excess_fpr"] - rows["augmentation"]["gg_excess_fpr"],
            "consistency_minus_augmentation_gf_core_fnr": rows["consistency"]["gf_core_fnr"] - rows["augmentation"]["gf_core_fnr"],
            "original_smoothing_minus_raw_gf_core_fnr": rows["original"]["gf_smooth50_core_fnr"] - rows["original"]["gf_core_fnr"],
        }
        return rows, contrasts

    rng = np.random.default_rng(20260924)
    result = {"protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
              "cases": 200, "speakers": 40, "seeds": [1, 2, 3],
              "bootstrap_draws": 2000, "thresholds": {}}
    for point in threshold_names:
        point_rows, estimate = summarize(point, [1, 2, 3], np.arange(40))
        draws = {key: [] for key in estimate}
        for _ in range(2000):
            seed_idx = rng.integers(1, 4, size=3)
            speaker_idx = rng.integers(0, 40, size=40)
            _, trial = summarize(point, seed_idx, speaker_idx)
            for key, value in trial.items():
                draws[key].append(value)
        contrasts = {}
        for key, value in estimate.items():
            sample = np.asarray(draws[key])
            contrasts[key] = {
                "estimate": value,
                "paired_speaker_seed_95ci": np.quantile(sample, [0.025, 0.975]).tolist(),
                "bootstrap_two_sided_p": float(min(1, 2 * min(np.mean(sample <= 0), np.mean(sample >= 0)))),
            }
        if point == "dev_fpr_0.05":
            primary = ["augmentation_minus_original_gg_excess_fpr", "augmentation_minus_original_gf_core_fnr"]
            ordered = sorted(primary, key=lambda key: contrasts[key]["bootstrap_two_sided_p"])
            adjusted = {}
            running = 0.0
            for i, key in enumerate(ordered):
                running = max(running, (2-i) * contrasts[key]["bootstrap_two_sided_p"])
                adjusted[key] = min(1.0, running)
            for key in primary:
                contrasts[key]["holm_adjusted_bootstrap_p"] = adjusted[key]
        result["thresholds"][point] = {"rates": point_rows, "contrasts": contrasts}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["thresholds"]["dev_fpr_0.05"], indent=2))


if __name__ == "__main__":
    main()
