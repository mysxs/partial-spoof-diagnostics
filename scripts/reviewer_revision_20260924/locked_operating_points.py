"""Calibrate on official development predictions only; never accept test labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

TARGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

def official_spoof_targets(values):
    raw = np.asarray(values)
    assert np.isin(raw.astype(str), ['0', '1']).all(), 'Unexpected official label'
    return raw.astype(np.int8) == 0

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def threshold_at_fpr(genuine_scores, target):
    s = np.asarray(genuine_scores)
    assert s.ndim == 1 and len(s) and np.isfinite(s).all()
    assert 0 <= target < 1
    allowed = int(np.floor(len(s) * target))
    index = len(s) - allowed - 1
    return float(np.partition(s, index)[index])

def main():
    ap = argparse.ArgumentParser()
    for k in ['run-root', 'data-root', 'out']:
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); root = Path(a.run_root); data = Path(a.data_root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    labelpath = data/'database/segment_labels/dev_seglab_0.02.npy'
    idpath = data/'database/dev/dev.lst'
    ids = idpath.read_text().splitlines()
    labels = np.load(labelpath, allow_pickle=True).item()
    result = {'targets': TARGETS, 'positive': 'official label == 0',
              'decision': 'spoof_score > threshold', 'split': 'dev',
              'ids_sha256': sha(idpath), 'labels_sha256': sha(labelpath),
              'script_sha256': sha(__file__), 'models': {}}
    for arch, directory in [('frame', 'frame_interventions_v2'), ('phone', 'unit_interventions_v2')]:
        for condition in ['original', 'augmentation', 'consistency']:
            for seed in [1, 2, 3]:
                run = root/directory/f'{condition}_seed{seed}'
                predpath = run/'dev_predictions.npz'
                dest = out/f'{arch}_{condition}_seed{seed}.json'
                identity = {str(p.name): sha(p) for p in [predpath, run/'best.pt', run/'config.json']}
                if dest.exists():
                    item = json.loads(dest.read_text())
                    assert item['identity'] == identity
                    assert item['labels_sha256'] == result['labels_sha256']
                else:
                    with np.load(predpath) as pred:
                        assert set(pred.files) == set(ids) and len(pred.files) == len(ids)
                        ys, ss = [], []
                        for uid in ids:
                            y = official_spoof_targets(labels[uid]); s = pred[uid]
                            assert s.shape == y.shape and np.isfinite(s).all()
                            ys.append(y); ss.append(s)
                    y = np.concatenate(ys); s = np.concatenate(ss)
                    assert y.any() and (~y).any()
                    negative = s[~y]; positive = s[y]; points = {}
                    for target in TARGETS:
                        threshold = threshold_at_fpr(negative, target)
                        fpr = float(np.mean(negative > threshold))
                        assert fpr <= target + 1e-12
                        points[str(target)] = {'threshold': threshold, 'dev_fpr': fpr,
                            'dev_fnr': float(np.mean(positive <= threshold))}
                    item = {'identity': identity, 'labels_sha256': result['labels_sha256'],
                            'utterances': len(ids), 'frames': len(y), 'points': points}
                    dest.write_text(json.dumps(item, indent=2))
                key = f'{arch}/{condition}/seed{seed}'; result['models'][key] = item
                print(key, item['points']['0.05'], flush=True)
    (out/'complete.json').write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
