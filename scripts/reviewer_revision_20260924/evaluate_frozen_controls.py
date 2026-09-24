"""Supplementary eval-split controls; fixed checkpoints and dev-only thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_frame import FrameHead, batches, align_logits
from evaluate_matched_controls import predict_phone, summarize_predictions, joint_bootstrap
from locked_operating_points import sha

def main():
    ap = argparse.ArgumentParser()
    for k in ['controls', 'run-root', 'calibration', 'out', 'protocol']:
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--architectures', nargs='+', choices=['frame', 'phone'], default=['frame', 'phone'])
    a = ap.parse_args(); torch.set_num_threads(8)
    root, runroot, out = Path(a.controls), Path(a.run_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(a.protocol).read_text())
    calibration = json.loads(Path(a.calibration).read_text())
    assert calibration['split'] == 'dev'
    cases = json.loads((root/'controls.json').read_text())
    assert json.loads((root/'complete.json').read_text())['split'] == 'eval'
    assert len(cases) == spec['cases']
    ids = (root/'ids.txt').read_text().splitlines()
    labels = np.load(root/'labels20_spoof1.npz'); masks = np.load(root/'core_masks20.npz')
    assert set(ids) == set(labels.files) == set(masks.files)
    fm = json.loads((root/'features_v2/metadata.json').read_text())
    hashes = {name: sha(root/name) for name in ['controls.json', 'labels20_spoof1.npz', 'core_masks20.npz', 'features_v2/metadata.json']}
    for arch in a.architectures:
        rows = []
        for uid in ids:
            if arch == 'frame':
                rows.append((uid, np.load(root/'features_v2'/(uid+'.npy')), labels[uid].astype('float32')))
            else:
                with np.load(root/'units'/(uid+'.npz')) as f:
                    rows.append((uid, f['phone_x'], f['phone_intervals']))
        directory = 'frame_interventions_v2' if arch == 'frame' else 'unit_interventions_v2'
        for condition in ['original', 'augmentation', 'consistency']:
            for seed in [1, 2, 3]:
                key = f'{arch}/{condition}/seed{seed}'
                ckpt = runroot/directory/f'{condition}_seed{seed}'/'best.pt'
                checkpoint_hash = sha(ckpt)
                assert checkpoint_hash == spec['checkpoints'][key] == calibration['models'][key]['identity']['best.pt']
                dest = out/arch/f'{condition}_seed{seed}'; dest.mkdir(parents=True, exist_ok=True)
                identity = {'checkpoint': checkpoint_hash, 'controls': hashes, 'calibration': sha(a.calibration),
                            'script': sha(__file__), 'protocol': sha(a.protocol)}
                if arch == 'phone':
                    identity['units_metadata'] = sha(root/'units/metadata.json')
                state = torch.load(ckpt, map_location='cpu', weights_only=False); cfg = state['config']
                trainmeta = cfg['feature_metadata'][0] if arch == 'frame' else cfg['unit_metadata'][0]['feature_metadata']
                for field in ['model', 'source_sha256', 'normalize', 'stride_samples', 'layer', 'padding_policy']:
                    assert fm[field] == trainmeta[field], field
                assert cfg['condition'] == condition and cfg['seed'] == seed
                predpath = dest/'predictions.npz'
                if predpath.exists():
                    assert json.loads((dest/'prediction_identity.json').read_text()) == identity
                    with np.load(predpath) as f:
                        predictions = {uid: f[uid] for uid in ids}
                else:
                    model = FrameHead(rows[0][1].shape[1], 'temporal').cuda()
                    model.load_state_dict(state['model']); model.eval()
                    if arch == 'phone':
                        predictions = predict_phone(model, rows, labels)
                    else:
                        predictions = {}
                        with torch.inference_mode():
                            for batch, x, y, m, lengths, mask in batches(rows, 32, list(range(len(rows)))):
                                scores = torch.sigmoid(align_logits(model(x, mask), lengths, y.shape[1], **cfg['geometry'])).cpu().numpy()
                                for i, (uid, _, yy) in enumerate(batch):
                                    predictions[uid] = scores[i, :len(yy)]
                    (dest/'prediction_identity.json').write_text(json.dumps(identity, indent=2))
                    with (dest/'predictions.tmp').open('wb') as f:
                        np.savez(f, **predictions)
                    (dest/'predictions.tmp').replace(predpath)
                    del model; torch.cuda.empty_cache()
                assert set(predictions) == set(ids)
                points = {'original_dev_eer': float(state['metrics']['dev_selected_spoof_threshold'])}
                points.update({f'dev_fpr_{k}': v['threshold'] for k, v in calibration['models'][key]['points'].items()})
                result = {'identity': identity, 'thresholds': points, 'results': {}, 'predictions_sha256': sha(predpath)}
                for name, threshold in points.items():
                    summary, per_case = summarize_predictions(cases, labels, masks, predictions, threshold)
                    result['results'][name] = summary
                    (dest/(name+'_cases.json')).write_text(json.dumps(per_case))
                (dest/'complete.json').write_text(json.dumps(result, indent=2))
                print(key, result['results']['dev_fpr_0.05']['effects'], flush=True)
        del rows
    # Only assemble the interaction when both complete architecture sets exist.
    all_results = {}
    for arch in ['frame', 'phone']:
        for condition in ['original', 'augmentation', 'consistency']:
            for seed in [1, 2, 3]:
                p = out/arch/f'{condition}_seed{seed}'/'complete.json'
                if p.exists():
                    all_results[f'{arch}/{condition}/seed{seed}'] = json.loads(p.read_text())
    (out/'summary.json').write_text(json.dumps(all_results, indent=2))
    if len(all_results) == 18:
        interactions = {}
        for point in ['original_dev_eer', 'dev_fpr_0.01', 'dev_fpr_0.05']:
            records = {(arch, condition, seed): json.loads((out/arch/f'{condition}_seed{seed}'/(point+'_cases.json')).read_text())
                       for arch in ['frame', 'phone'] for condition in ['original', 'augmentation', 'consistency'] for seed in [1, 2, 3]}
            interactions[point] = joint_bootstrap(records, [1, 2, 3], draws=2000)
        (out/'interaction.json').write_text(json.dumps(interactions, indent=2))
    print('Evaluation completed', a.architectures, flush=True)

if __name__ == '__main__':
    main()
