#!/usr/bin/env python3
"""Construct frozen GG/GF controls from official LlamaPartialSpoof A audio.

Donors share the LibriSpeech source-speaker ID encoded in the official filenames.
They are not text matched; these controls test transfer of the PS-trained frame
models, not causal isolation of a unique splice cue.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def splice(base, donor, start, width):
    alpha = np.ones(len(donor), dtype=np.float64)
    fade = min(width, len(donor) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, endpoint=True)
        alpha[:fade] = ramp
        alpha[-fade:] = ramp[::-1]
    result = base.copy()
    end = start + len(donor)
    result[start:end] = (1 - alpha) * base[start:end] + alpha * donor
    return result, alpha


def rms(signal):
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "con_wav").mkdir(exist_ok=True)
    assert not (out / "complete.json").exists(), "A completed construction must not be overwritten"
    label_path = root / "label_R01TTS.0.a.txt"
    archive_marker = root / "R01TTS.0.a.tgz.extracted.json"
    archive = json.loads(archive_marker.read_text())
    assert archive["md5"] == "685acfe986b50baaf3e25e9d5e3091a4"
    by_speaker = defaultdict(lambda: {"genuine": [], "partial": []})
    for line in label_path.read_text().splitlines():
        fields = line.split()
        uid, duration, whole = fields[:3]
        speaker = uid.split("_")[1]
        intervals = []
        for token in fields[3:]:
            start, end, label = token.split("-")
            intervals.append((float(start), float(end), label))
        if whole == "bonafide" and (uid.startswith("dev-clean_") or uid.startswith("test-clean_")):
            by_speaker[speaker]["genuine"].append((uid, float(duration)))
        elif whole == "spoof" and "partial" in uid:
            fake_regions = [(start, end) for start, end, label in intervals
                            if label == "spoof" and end - start >= .44]
            if fake_regions:
                by_speaker[speaker]["partial"].append((uid, fake_regions))
    eligible = sorted(s for s, group in by_speaker.items()
                      if len(group["genuine"]) >= 2 and group["partial"])
    assert eligible
    rng = np.random.default_rng(args.seed)
    audio_cache = {}

    def audio(uid):
        if uid not in audio_cache:
            path = root / "R01TTS.0.a" / f"{uid}.wav"
            waveform, sr = sf.read(path, dtype="float32")
            assert sr == 16000 and waveform.ndim == 1 and len(waveform)
            audio_cache[uid] = waveform
        return audio_cache[uid]

    records, labels, masks, skipped = [], {}, {}, []
    # Round-robin speakers keeps the source distribution from collapsing to one voice.
    per_speaker = {speaker: list(rng.permutation(by_speaker[speaker]["genuine"])) for speaker in eligible}
    speakers = list(rng.permutation(eligible))
    rounds = 0
    while len(records) < args.cases and any(per_speaker.values()):
        rounds += 1
        for speaker in speakers:
            if len(records) >= args.cases:
                break
            pool = per_speaker[speaker]
            if not pool:
                continue
            base_id, _ = pool.pop()
            base = audio(base_id)
            count = 6400  # 400 ms at 16 kHz
            if len(base) < count + 3200:
                skipped.append((base_id, "short base"))
                continue
            start_candidates = rng.integers(1600, len(base) - count - 1600 + 1, size=16)
            starts = [int(start) for start in start_candidates if rms(base[start:start + count]) >= .008]
            if not starts:
                skipped.append((base_id, "silent host"))
                continue
            start = starts[0]
            genuine_candidates = [uid for uid, _ in by_speaker[speaker]["genuine"] if uid != base_id]
            rng.shuffle(genuine_candidates)
            genuine_choice = None
            for uid in genuine_candidates[:40]:
                wav = audio(uid)
                if len(wav) < count:
                    continue
                for offset in rng.integers(0, len(wav) - count + 1, size=8):
                    donor = wav[offset:offset + count]
                    if rms(donor) >= .008:
                        genuine_choice = (uid, int(offset), donor.copy())
                        break
                if genuine_choice:
                    break
            fake_choice = None
            partial_candidates = by_speaker[speaker]["partial"].copy()
            rng.shuffle(partial_candidates)
            for uid, regions in partial_candidates[:80]:
                wav = audio(uid)
                for begin, end in regions:
                    lo, hi = int(np.ceil((begin + .02) * 16000)), int(np.floor((end - .02) * 16000))
                    if hi - lo < count:
                        continue
                    for offset in rng.integers(lo, hi - count + 1, size=5):
                        donor = wav[offset:offset + count]
                        if rms(donor) >= .008:
                            fake_choice = (uid, int(offset), donor.copy())
                            break
                    if fake_choice:
                        break
                if fake_choice:
                    break
            if not genuine_choice or not fake_choice:
                skipped.append((base_id, "donor unavailable"))
                continue
            target_rms = rms(base[start:start + count])
            gains = {"gg": target_rms / rms(genuine_choice[2]),
                     "gf": target_rms / rms(fake_choice[2])}
            if not all(.25 <= gain <= 4 for gain in gains.values()):
                skipped.append((base_id, "extreme gain"))
                continue
            donors = {"gg": genuine_choice[2] * gains["gg"],
                      "gf": fake_choice[2] * gains["gf"]}
            if max(float(np.max(np.abs(signal))) for signal in (base, *donors.values())) >= .999:
                skipped.append((base_id, "clipping"))
                continue
            case_id = f"llama_a_{len(records):05d}"
            variants = {}
            n20 = int(np.ceil(len(base) / 320))
            edges = np.arange(n20) * 320
            valid = np.ones(n20, dtype=bool)
            for boundary in (start, start + count):
                valid &= ~((edges < boundary + 960) & (edges + 320 > boundary - 960))
            for variant, donor_type, fade in (("genuine", None, 0), ("gg_raw", "gg", 0),
                                               ("gf_raw", "gf", 0), ("gf_smooth50", "gf", 800)):
                if donor_type is None:
                    waveform, alpha = base.copy(), np.zeros(count)
                else:
                    waveform, alpha = splice(base, donors[donor_type], start, fade)
                provenance = np.zeros(len(base), dtype=np.float32)
                if donor_type == "gf":
                    provenance[start:start + count] = alpha
                binary = (np.pad(provenance, (0, n20 * 320 - len(base)))
                          .reshape(n20, 320) > 0).any(axis=1).astype(np.uint8)
                uid = f"{case_id}_{variant}"
                sf.write(out / "con_wav" / f"{uid}.wav", waveform, 16000, subtype="FLOAT")
                labels[uid], masks[uid] = binary, valid.copy()
                variants[variant] = {"uttid": uid, "fake_frames": int(binary.sum()),
                                     "width_ms": fade / 16}
            records.append({"case": case_id, "protocol_speaker": speaker, "base_id": base_id,
                            "genuine_donor": genuine_choice[0], "fake_donor": fake_choice[0],
                            "insert_start_sample": start, "insert_end_sample": start + count,
                            "rms_gains": gains, "variants": variants})
            if len(records) % 25 == 0:
                print("Constructed", len(records), "cases", flush=True)
    assert len(records) == args.cases, (len(records), args.cases)
    (out / "controls.json").write_text(json.dumps(records, indent=2))
    (out / "ids.txt").write_text("\n".join(labels) + "\n")
    np.savez(out / "labels20_spoof1.npz", **labels)
    np.savez(out / "core_masks20.npz", **masks)
    result = {"split": "eval", "source": "LlamaPartialSpoof A official archive",
              "cases": len(records), "variants": len(labels),
              "speakers": len({r["protocol_speaker"] for r in records}),
              "seed": args.seed, "label_sha256": sha256(label_path),
              "archive_md5": archive["md5"], "script_sha256": sha256(__file__),
              "skipped": skipped,
              "limitations": ["Constructed controls, not official benchmark samples or labels.",
                              "Speaker identity inferred from released source-filename field.",
                              "Donor text and synthesis method are not matched.",
                              "Source corpus has already been seen in the broader transfer study; this is an intervention-specific follow-up, not a pristine blind benchmark."]}
    (out / "complete.json").write_text(json.dumps(result, indent=2))
    print("COMPLETE", len(records), "cases", result["speakers"], "speakers", flush=True)


if __name__ == "__main__":
    main()
