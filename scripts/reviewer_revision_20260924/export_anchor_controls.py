#!/usr/bin/env python3
"""Label-independent anchor controls with the canonical encoder and soft targets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from export_unit_features import feature_pool, occupancy, overlap_weights


def control_intervals(intervals, duration, uid, mode):
    if mode == 'fixed80':
        edges = np.r_[np.arange(0, duration - 1e-8, .08), duration]
    elif mode == 'permuted':
        widths = np.diff(np.asarray(intervals, dtype='float64'), axis=1)[:, 0]
        seed = int.from_bytes(hashlib.sha256(('anchor_control_v1:' + uid).encode()).digest()[:8], 'little')
        widths = np.random.default_rng(seed).permutation(widths)
        edges = np.r_[0, np.cumsum(widths)]
        edges[-1] = duration
    else:
        raise ValueError(mode)
    assert edges[0] == 0 and np.all(np.diff(edges) > 0)
    return np.column_stack([edges[:-1], edges[1:]])


def main():
    ap = argparse.ArgumentParser()
    for name in ['source', 'features', 'root', 'split', 'out']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    source, features, root, out = map(Path, [a.source, a.features, a.root, a.out])
    assert (source / 'complete.json').exists() and (features / 'complete.json').exists()
    old = json.loads((source / 'metadata.json').read_text())
    meta = json.loads((features / 'metadata.json').read_text())
    assert meta['padding_policy'] == 'exact_convolution_lengths'
    ids = (root / 'database' / a.split / (a.split + '.lst')).read_text().splitlines()
    if a.limit:
        ids = ids[:a.limit]
    labels = np.load(root / 'database/segment_labels' / (a.split + '_seglab_0.01.npy'), allow_pickle=True).item()
    modes = ['fixed80', 'permuted']
    for mode in modes:
        folder = out / mode / a.split
        folder.mkdir(parents=True, exist_ok=True)
        config = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'source_metadata': old, 'feature_metadata': meta, 'mode': mode,
                  'split': a.split, 'limit': a.limit,
                  'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                  'target': 'raw_10ms_duration_occupancy',
                  'note': 'Train phone mode only: original word arrays are inert compatibility fields.'}
        path = folder / 'metadata.json'
        if path.exists():
            assert json.loads(path.read_text()) == config
        else:
            path.write_text(json.dumps(config, indent=2))
    counts = {m: 0 for m in modes}
    for index, uid in enumerate(ids):
        with np.load(source / (uid + '.npz'), allow_pickle=False) as f:
            row = {k: f[k] for k in f.files}
        duration = float(row['duration'])
        x = None
        for mode in modes:
            dest = out / mode / a.split / (uid + '.npz')
            units = control_intervals(row['phone_intervals'], duration, uid, mode)
            counts[mode] += len(units)
            if dest.exists():
                continue
            if x is None:
                x = np.load(features / (uid + '.npy'), allow_pickle=False)
            result = dict(row)
            result['phone_intervals'] = units.astype('float64')
            result['phone_kind'] = np.zeros(len(units), dtype='uint8')
            result['phone_target'] = occupancy(labels[uid], units)
            result['phone_x'] = feature_pool(x, units, duration, center=meta['first_center_seconds'], step=meta['stride_samples']/16000)
            # Refresh word acoustics too, avoiding stale acoustic provenance even though unused.
            result['word_x'] = feature_pool(x, row['word_intervals'], duration, center=meta['first_center_seconds'], step=meta['stride_samples']/16000)
            result['phone_to_word'] = overlap_weights(units, np.r_[row['word_intervals'][:, 0], row['word_intervals'][-1, 1]]).astype('float32')
            assert np.isfinite(result['phone_x']).all()
            temp = dest.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f, **result)
            temp.replace(dest)
        if (index + 1) % 1000 == 0:
            print(a.split, index + 1, '/', len(ids), flush=True)
    for mode in modes:
        (out / mode / a.split / 'complete.json').write_text(json.dumps({
            'utterances': len(ids), 'units': counts[mode], 'mode': mode,
            'note': 'All requested trials retained; no reference-driven anchor selection.'}, indent=2))
    print('Complete', a.split, len(ids), counts, flush=True)


if __name__ == '__main__':
    main()
