#!/usr/bin/env python3
"""Freeze Llama-A follow-up analysis before extracting features or scoring."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ps-validation", required=True)
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    out, old, runroot = Path(args.out), Path(args.ps_validation), Path(args.run_root)
    target = out / "protocol.json"
    assert not target.exists(), "Protocol was already frozen"
    control = out / "controls"
    complete = json.loads((control / "complete.json").read_text())
    assert complete["cases"] == 200 and complete["variants"] == 800
    cases = json.loads((control / "controls.json").read_text())
    ids = (control / "ids.txt").read_text().splitlines()
    assert len(cases) == 200 and len(ids) == len(set(ids)) == 800
    assert len({c["protocol_speaker"] for c in cases}) == 40
    for case in cases:
        speaker = case["protocol_speaker"]
        assert all(uid.split("_")[1] == speaker for uid in
                   (case["base_id"], case["genuine_donor"], case["fake_donor"]))
        assert set(case["variants"]) == {"genuine", "gg_raw", "gf_raw", "gf_smooth50"}
    for uid in ids:
        assert (control / "con_wav" / f"{uid}.wav").stat().st_size > 44
    prior = json.loads((old / "protocol.json").read_text())
    calibration = old / "calibration/complete.json"
    assert json.loads(calibration.read_text())["split"] == "dev"
    checkpoints = {}
    for condition in ("original", "augmentation", "consistency"):
        for seed in (1, 2, 3):
            key = f"frame/{condition}/seed{seed}"
            path = runroot / "frame_interventions_v2" / f"{condition}_seed{seed}" / "best.pt"
            checkpoints[key] = sha(path)
            assert checkpoints[key] == prior["checkpoints"][key]
    protocol = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "Official LlamaPartialSpoof A archive, checksum verified; only generated controls scored here",
        "cases": 200, "speakers": 40, "seed": 20260924,
        "model_selection": "All 9 frame checkpoints and operating thresholds selected previously on PS development",
        "checkpoints": checkpoints,
        "source_thresholds": ["original_dev_eer", "dev_fpr_0.01", "dev_fpr_0.05"],
        "primary": ["At dev 5% FPR, augmentation minus original GG excess FPR",
                    "At dev 5% FPR, augmentation minus original GF-core FNR"],
        "secondary": ["The same contrasts at dev 1% FPR", "Consistency minus augmentation",
                      "50ms smoothing response", "full development-threshold grid"],
        "inference": "Frozen final-layer WavLM Large and 9 saved frame heads, no Llama-A tuning or polarity flip",
        "statistics": "Three training seeds; speaker-and-seed paired bootstrap with 2000 draws; report both primary contrasts together, with Holm correction for two primary endpoints; secondary intervals descriptive",
        "controls_sha256": {name: sha(control / name) for name in
                            ("controls.json", "ids.txt", "labels20_spoof1.npz", "core_masks20.npz", "complete.json")},
        "calibration_sha256": sha(calibration),
        "prior_protocol_sha256": sha(old / "protocol.json"),
        "builder_sha256": complete["script_sha256"],
        "source_label_sha256": complete["label_sha256"],
        "source_archive_md5": complete["archive_md5"],
        "limitations": ["The broader paper already inspected Llama-A transfer EER, so this is an intervention-specific pre-scored follow-up, not a pristine blind corpus.",
                        "Constructed controls are not official Llama-A benchmark samples or labels.",
                        "Source-speaker ID is inferred from official filenames; texts and synthesis generators are not matched.",
                        "The source 400ms donor and boundary-excluded core policy match PS controls, but source-corpus covariates remain confounded."]}
    target.write_text(json.dumps(protocol, indent=2))
    print("FROZEN", target, sha(target), flush=True)


if __name__ == "__main__":
    main()
