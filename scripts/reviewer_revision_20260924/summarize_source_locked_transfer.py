#!/usr/bin/env python3
"""Aggregate saved three-seed source-threshold confusion counts without refitting."""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "final": {
        "PS eval": "results/eval_frames_v2/base.json",
        "HAD": "results/external/had/frame_results.json",
        "Llama A": "results/source_locked_transfer/raw/llama_a_final.json",
        "Llama B": "results/llama_b_last/summary.json",
    },
    "layer 6": {
        "PS eval": "results/ssl_layers_v2/layer6/eval_ps/summary.json",
        "HAD": "results/ssl_layers_v2/layer6/eval_had/summary.json",
        "Llama A": "results/llama_a_frame/layer6_summary.json",
        "Llama B": "results/llama_b_layers/layer6/summary.json",
    },
    "layer 12": {
        "PS eval": "results/ssl_layers_v2/layer12/eval_ps/summary.json",
        "HAD": "results/ssl_layers_v2/layer12/eval_had/summary.json",
        "Llama A": "results/llama_a_frame/layer12_summary.json",
        "Llama B": "results/llama_b_layers/layer12/summary.json",
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "results/source_locked_transfer"
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], {}
    identity_by_system_seed = {}
    for system, datasets in SOURCES.items():
        for dataset, relpath in datasets.items():
            path = ROOT / relpath
            source = json.loads(path.read_text())
            provenance[relpath] = digest(path)
            selected = {}
            for key, item in source.items():
                expected_parent = ("/frame_baselines_v2/" if system == "final" else
                                   f"/ssl_layers_v2/layer{system.split()[-1]}/models/")
                if expected_parent not in key:
                    continue
                match = re.search(r"/temporal_seed([123])/best\.pt$", key)
                if match is None:
                    continue
                seed = int(match.group(1))
                assert seed not in selected
                selected[seed] = item
            assert sorted(selected) == [1, 2, 3], (relpath, selected)
            ids_hash = {item["identity"]["ids_sha256"] for item in selected.values()}
            assert len(ids_hash) == 1, relpath
            for seed, item in selected.items():
                identity = item["identity"]
                cp_hash = identity["checkpoint_sha256"]
                threshold = item["dev_locked_spoof_threshold"]
                if (system, seed) in identity_by_system_seed:
                    assert identity_by_system_seed[system, seed] == (cp_hash, threshold)
                else:
                    identity_by_system_seed[system, seed] = (cp_hash, threshold)
                tp, fp, fn, tn = (item[k] for k in ("tp", "fp", "fn", "tn"))
                assert abs(fp / (fp + tn) - item["fpr_at_dev_threshold"]) < 1e-12
                assert abs(fn / (fn + tp) - item["fnr_at_dev_threshold"]) < 1e-12
                rows.append({"system": system, "dataset": dataset, "seed": seed,
                             "checkpoint_sha256": cp_hash, "ids_sha256": next(iter(ids_hash)),
                             "dev_locked_spoof_threshold": threshold,
                             "eer_pct": 100 * item["segment20_eer"],
                             "fpr_pct": 100 * item["fpr_at_dev_threshold"],
                             "fnr_pct": 100 * item["fnr_at_dev_threshold"],
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "source_json": relpath, "source_sha256": provenance[relpath]})
    with (out / "seed_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for system in SOURCES:
        for dataset in SOURCES[system]:
            subset = [row for row in rows if row["system"] == system and row["dataset"] == dataset]
            assert len(subset) == 3
            summary.append({"system": system, "dataset": dataset,
                            **{name: {"mean": statistics.mean(row[name] for row in subset),
                                      "sd": statistics.stdev(row[name] for row in subset)}
                               for name in ("eer_pct", "fpr_pct", "fnr_pct")}})
    (out / "summary.json").write_text(json.dumps({"summary": summary, "source_sha256": provenance,
        "policy": "All operating thresholds and checkpoints saved on PS development; no external tuning or polarity flip. Three seed means and sample SD. FPR/FNR use pooled 20-ms frames."}, indent=2))
    print("COMPLETE", len(rows), "seed rows")
    for row in summary:
        print(row["system"], row["dataset"],
              *(f"{row[m]['mean']:.2f}±{row[m]['sd']:.2f}" for m in ("eer_pct", "fpr_pct", "fnr_pct")))


if __name__ == "__main__":
    main()
