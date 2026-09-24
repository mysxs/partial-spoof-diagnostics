#!/usr/bin/env python3
"""Audit frozen layer-6 PS predictions at saved development thresholds.

This is a retrospective audit. It never selects checkpoints or thresholds on
PS evaluation and retains the released 10-ms reference labels unchanged.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from event_strata import duration_bin, match_events, spans


ARMS = {
    "Soft frame": ("layer6_language_models/soft_frame/models", "temporal", "layer6_language_eval/soft_frame_ps/models"),
    "Phone unit": ("layer6_language_models/phone_unit/models", "phone", "layer6_language_eval/phone_unit_ps/models"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_history(run):
    history = json.loads((run / "history.json").read_text())
    return min(history, key=lambda row: row["segment20_eer"])


def main():
    parser = argparse.ArgumentParser()
    for key in ("run-root", "data-root", "official-repo", "out"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    root, data, out = Path(args.run_root), Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.official_repo)
    from sandbox.eval_asvspoof import compute_eer

    rows = []
    for split in ("dev", "eval"):
        ids_path = data / "database" / split / f"{split}.lst"
        label10_path = data / "database/segment_labels" / f"{split}_seglab_0.01.npy"
        label20_path = data / "database/segment_labels" / f"{split}_seglab_0.02.npy"
        ids = ids_path.read_text().splitlines()
        expected = {"dev": 24844, "eval": 71237}[split]
        assert len(ids) == len(set(ids)) == expected
        labs10 = np.load(label10_path, allow_pickle=True).item()
        labs20 = np.load(label20_path, allow_pickle=True).item()
        y_by_id = {uid: np.asarray(labs20[uid]).astype(np.int8) == 0 for uid in ids}
        ref_by_id = {uid: spans(np.asarray(labs10[uid]).astype(np.int8) == 0, .01) for uid in ids}
        y = np.concatenate([y_by_id[uid] for uid in ids])
        label_hashes = {"ids_sha256": sha256(ids_path), "labels10_sha256": sha256(label10_path),
                        "labels20_sha256": sha256(label20_path)}
        for arm, (model_dir, prefix, eval_dir) in ARMS.items():
            for seed in (1, 2, 3):
                run = root / model_dir / f"{prefix}_seed{seed}"
                best = selected_history(run)
                threshold = float(best["dev_selected_spoof_threshold"])
                pred_path = (run / "dev_predictions.npz" if split == "dev" else
                             root / eval_dir / f"{prefix}_seed{seed}" / "predictions.npz")
                assert pred_path.exists(), pred_path
                source = None if split == "dev" else json.loads((pred_path.parent / "complete.json").read_text())
                checkpoint_hash = sha256(run / "best.pt")
                if source is not None:
                    assert source["identity"]["checkpoint_sha256"] == checkpoint_hash
                    assert abs(source["dev_locked_spoof_threshold"] - threshold) < 1e-12
                    assert source["utterances"] == expected
                identity = {"arm": arm, "seed": seed, "split": split, "checkpoint_sha256": checkpoint_hash,
                            "predictions_sha256": sha256(pred_path), "history_sha256": sha256(run / "history.json"),
                            "matching_sha256": sha256(Path(__file__).with_name("event_strata.py")), **label_hashes}
                dest = out / f"{split}_{arm.lower().replace(' ', '_')}_seed{seed}.json"
                if dest.exists():
                    row = json.loads(dest.read_text())
                    assert row["identity"] == identity
                    rows.append(row)
                    print("REUSED", dest.name, flush=True)
                    continue
                tp = fp = fn = tn = nr = npred = nm = nshort = nshortmatch = 0
                start_err = end_err = 0.
                score_parts = []
                with np.load(pred_path, allow_pickle=False) as pred:
                    assert set(pred.files) == set(ids)
                    for uid in ids:
                        score, target = pred[uid], y_by_id[uid]
                        assert score.shape == target.shape and np.isfinite(score).all()
                        assert np.all((0 <= score) & (score <= 1))
                        score_parts.append(score)
                        mask = score > threshold
                        tp += int(np.count_nonzero(mask & target))
                        fp += int(np.count_nonzero(mask & ~target))
                        fn += int(np.count_nonzero(~mask & target))
                        tn += int(np.count_nonzero(~mask & ~target))
                        ref, proposal = ref_by_id[uid], spans(mask, .02)
                        matches = match_events(ref, proposal)
                        short = {i for i, (s, e) in enumerate(ref) if duration_bin(e - s) == "lt100"}
                        nr += len(ref)
                        npred += len(proposal)
                        nm += len(matches)
                        nshort += len(short)
                        nshortmatch += sum(i in short for i, _, _ in matches)
                        start_err += sum(abs(ref[i, 0] - proposal[j, 0]) for i, j, _ in matches)
                        end_err += sum(abs(ref[i, 1] - proposal[j, 1]) for i, j, _ in matches)
                scores = np.concatenate(score_parts)
                eer, _ = compute_eer(1 - scores[~y], 1 - scores[y])
                target_eer = best["segment20_eer"] if split == "dev" else source["segment20_eer"]
                assert abs(eer - target_eer) < 1e-10, (arm, seed, split, eer, target_eer)
                if source is not None:
                    assert (tp, fp, fn, tn) == tuple(source[k] for k in ("tp", "fp", "fn", "tn"))
                row = {"identity": identity, "arm": arm, "split": split, "seed": seed, "layer": 6,
                       "selected_epoch": best["epoch"], "threshold": threshold,
                       "utterances": expected, "frames": len(y), "frame_eer_pct": 100 * float(eer),
                       "frame_f1_pct": 100 * 2 * tp / (2 * tp + fp + fn),
                       "frame_recall_pct": 100 * tp / (tp + fn),
                       "event_precision_pct": 100 * nm / npred,
                       "event_recall_pct": 100 * nm / nr,
                       "event_f1_pct": 100 * 2 * nm / (nr + npred),
                       "short_event_recall_pct": 100 * nshortmatch / nshort,
                       "matched_boundary_mae_ms": 1000 * (start_err + end_err) / (2 * nm),
                       "reference_events": nr, "predicted_events": npred, "matched_events": nm,
                       "short_reference_events": nshort, "short_matched_events": nshortmatch,
                       "tp": tp, "fp": fp, "fn": fn, "tn": tn}
                dest.write_text(json.dumps(row, indent=2))
                rows.append(row)
                print("PASS", split, arm, seed, "EER", row["frame_eer_pct"],
                      "event F1", row["event_f1_pct"], flush=True)
                del scores, score_parts
    metrics = ("frame_eer_pct", "frame_f1_pct", "frame_recall_pct", "event_precision_pct",
               "event_recall_pct", "event_f1_pct", "short_event_recall_pct", "matched_boundary_mae_ms")
    summary = []
    for split in ("dev", "eval"):
        for arm in ARMS:
            group = [r for r in rows if r["split"] == split and r["arm"] == arm]
            assert len(group) == 3
            summary.append({"split": split, "arm": arm, "layer": 6, "seeds": [1, 2, 3],
                            **{metric: {"mean": statistics.mean(r[metric] for r in group),
                                        "sd": statistics.stdev(r[metric] for r in group)} for metric in metrics}})
    (out / "summary.json").write_text(json.dumps({"models": rows, "summary": summary,
        "scope": "Frozen-prediction retrospective audit; original labels; saved PS-dev thresholds; IoU>=0.5."}, indent=2))
    with (out / "seed_results.csv").open("w", newline="") as stream:
        fields = ["split", "arm", "seed", "layer", "selected_epoch", "threshold", *metrics]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print("COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
