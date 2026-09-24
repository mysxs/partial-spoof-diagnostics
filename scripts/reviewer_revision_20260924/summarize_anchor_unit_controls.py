#!/usr/bin/env python3
"""Audit phone vs nonlinguistic unit-output controls from frozen checkpoints."""
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "results/anchor_unit_controls"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = ROOT / "raw"
    groups = {}
    source_hashes = {}
    ids_hashes = set()
    for arm in ("phone", "permuted", "fixed80"):
        path = raw / f"{arm}_eval.json"
        source_hashes[path.name] = sha(path)
        data = json.loads(path.read_text())
        items = {}
        for key, value in data.items():
            if arm == "phone" and "/unit_models_v2/phone_seed" not in key:
                continue
            seed = int(key.split("/phone_seed")[-1].split("/")[0])
            assert seed in (1, 2, 3) and seed not in items
            assert value["utterances"] == 71237
            ids_hashes.add(value["identity"]["ids_sha256"])
            meta = value["identity"]["unit_metadata"]
            assert meta["feature_metadata"]["layer"] == "last"
            if arm != "phone":
                assert meta["mode"] == arm
                assert meta["source_metadata"]["interval_storage"] == "float64"
            items[seed] = value
        assert sorted(items) == [1, 2, 3], (arm, items)
        groups[arm] = items
    assert len(ids_hashes) == 1
    metadata = {arm: json.loads((raw / f"{arm}_train_metadata.json").read_text())
                for arm in groups}
    source_hashes.update({f"{arm}_train_metadata.json": sha(raw / f"{arm}_train_metadata.json")
                          for arm in groups})
    common_features = [metadata[arm]["feature_metadata"] for arm in groups]
    assert all(m == common_features[0] for m in common_features)
    results = []
    for arm, items in groups.items():
        values = [100 * items[seed]["segment20_eer"] for seed in (1, 2, 3)]
        results.append({"arm": arm, "layer": "last", "seeds": [1, 2, 3],
                        "eer_mean_pct": statistics.mean(values),
                        "eer_sample_sd_pct": statistics.stdev(values),
                        "seed_eer_pct": values,
                        "checkpoint_sha256": [items[s]["identity"]["checkpoint_sha256"] for s in (1, 2, 3)]})
    (ROOT / "summary.json").write_text(json.dumps({"results": results,
        "control": "Permuted intervals preserve each utterance's phone width multiset but change boundary positions; all arms use unit outputs, duration-weighted occupancy BCE, the same frozen final-layer WavLM features, PS-dev checkpoint selection, and original PS-eval references.",
        "limitations": "These controls are final-layer, not layer-6, and PS evaluation is retrospective. Width permutation changes acoustic alignment and does not isolate semantic content by itself.",
        "ids_sha256": next(iter(ids_hashes)), "source_sha256": source_hashes}, indent=2))
    for row in results:
        print(row["arm"], f"{row['eer_mean_pct']:.3f}±{row['eer_sample_sd_pct']:.3f}")


if __name__ == "__main__":
    main()
