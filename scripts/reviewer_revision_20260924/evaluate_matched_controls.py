#!/usr/bin/env python3
"""Matched 2x3 control evaluation with paired speaker-and-seed uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_frame import FrameHead, batches, align_logits
from train_units import project_scores
from evaluate_controls import confusion, rates

CONDITIONS=['original','augmentation','consistency']


def effects(counts):
    r={k:rates(v) for k,v in counts.items()}
    return {'genuine_fpr':r['genuine_full']['fpr'],'gg_raw_fpr':r['gg_raw_full']['fpr'],
        'gg_minus_genuine_fpr':r['gg_raw_full']['fpr']-r['genuine_full']['fpr'],
        'gf_raw_fnr_core':r['gf_raw_core']['fnr'],
        'gf_smooth50_fnr_core':r['gf_smooth50_core']['fnr'],
        'smooth50_minus_raw_fnr_core':r['gf_smooth50_core']['fnr']-r['gf_raw_core']['fnr']}


def summarize_predictions(cases,labels,masks,predictions,threshold):
    per_case=[];total={}
    for case in cases:
        counts={}
        for name,variant in case['variants'].items():
            uid=variant['uttid'];y=labels[uid];scores=predictions[uid]
            assert len(y)==len(scores) and np.isfinite(scores).all()
            for scope,mask in [('full',np.ones(len(y),bool)),('core',masks[uid].astype(bool))]:
                key=name+'_'+scope;c=confusion(y,scores>threshold,mask);counts[key]=c.tolist()
                total[key]=total.get(key,np.zeros(4,dtype='int64'))+c
        per_case.append({'case':case['case'],'speaker':case['protocol_speaker'],'counts':counts})
    return {'conditions':{k:rates(v) for k,v in total.items()},'effects':effects(total)},per_case


def contrasts(group):
    result={}
    for metric in next(iter(group.values())):
        for architecture in ['frame','phone']:
            for new,base in [('augmentation','original'),('consistency','augmentation')]:
                name=f'{architecture}:{new}-minus-{base}:{metric}'
                result[name]=group[(architecture,new)][metric]-group[(architecture,base)][metric]
        result['interaction:consistency-minus-augmentation:'+metric]=(group[('phone','consistency')][metric]-group[('phone','augmentation')][metric])-(group[('frame','consistency')][metric]-group[('frame','augmentation')][metric])
    return result


def joint_bootstrap(records,seeds,draws=2000):
    speakers=sorted({c['speaker'] for c in next(iter(records.values()))});condition_keys=list(next(iter(records.values()))[0]['counts'])
    matrices={}
    for key,cases in records.items():
        matrix=np.zeros((len(speakers),len(condition_keys),4),dtype='int64')
        for c in cases:
            i=speakers.index(c['speaker'])
            for j,name in enumerate(condition_keys):matrix[i,j]+=c['counts'][name]
        matrices[key]=matrix
    def summarize(seed_indices,speaker_indices):
        group={}
        for architecture in ['frame','phone']:
            for condition in CONDITIONS:
                counts=sum((matrices[(architecture,condition,seeds[i])][speaker_indices].sum(axis=0) for i in seed_indices),np.zeros((len(condition_keys),4),dtype='int64'))
                group[(architecture,condition)]=effects(dict(zip(condition_keys,counts)))
        return contrasts(group)
    estimates=summarize(np.arange(len(seeds)),np.arange(len(speakers)))
    rng=np.random.default_rng(20260919);samples={key:[] for key in estimates}
    for _ in range(draws):
        result=summarize(rng.integers(len(seeds),size=len(seeds)),rng.integers(len(speakers),size=len(speakers)))
        for key,value in result.items():samples[key].append(value)
    return {'speakers':len(speakers),'seeds':seeds,'bootstrap_draws':draws,
        'contrasts':{k:{'estimate':v,'paired_speaker_seed_95ci':np.quantile(samples[k],[.025,.975]).tolist()} for k,v in estimates.items()},
        'limitations':['Exploratory paired intervals; few speakers and three seeds limit precision.',
            'All comparisons preserve the same speaker and seed draws across conditions.',
            'Multiple contrasts are not multiplicity-adjusted; do not select only favorable comparisons.',
            'Negative interaction means a larger reduction for phone models for the named error contrast; examine absolute error rates too.']}


def predict_phone(model,rows,labels,batch_size=32):
    predictions={}
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];length=max(len(r[1]) for r in batch)
            x=np.zeros((len(batch),length,batch[0][1].shape[1]),'float16');mask=np.zeros((len(batch),length),bool)
            for i,(_,features,units) in enumerate(batch):
                n=len(features);x[i,:n]=features;mask[i,:n]=True;assert (np.diff(units,axis=1)>0).all()
            scores=torch.sigmoid(model(torch.from_numpy(x).cuda().float(),torch.from_numpy(mask).cuda())).cpu().numpy()
            for i,(uid,features,units) in enumerate(batch):predictions[uid]=project_scores(scores[i,:len(features)],units,len(labels[uid]))
    return predictions


def main():
    ap=argparse.ArgumentParser()
    for key in ['controls','frame-runs','unit-runs','out']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2,3]);a=ap.parse_args();torch.set_num_threads(8)
    root,out=Path(a.controls),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert json.loads((root/'complete.json').read_text())['split']=='dev'
    assert (root/'features_v2/complete.json').exists() and (root/'units/complete.json').exists()
    cases=json.loads((root/'controls.json').read_text());ids=(root/'ids.txt').read_text().splitlines()
    labels=np.load(root/'labels20_spoof1.npz');masks=np.load(root/'core_masks20.npz')
    featuremeta=json.loads((root/'features_v2/metadata.json').read_text())
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['controls.json','labels20_spoof1.npz','core_masks20.npz','units/metadata.json','features_v2/metadata.json']}
    records={};all_results={}
    for architecture,folder in [('frame',Path(a.frame_runs)),('phone',Path(a.unit_runs))]:
        rows=[]
        for uid in ids:
            if architecture=='frame':rows.append((uid,np.load(root/'features_v2'/(uid+'.npy')),labels[uid].astype('float32')))
            else:
                with np.load(root/'units'/(uid+'.npz')) as f:rows.append((uid,f['phone_x'],f['phone_intervals']))
        for condition in CONDITIONS:
            for seed in a.seeds:
                run=folder/f'{condition}_seed{seed}';checkpoint=run/'best.pt';assert (run/'complete.json').exists()
                dest=out/architecture/run.name;dest.mkdir(parents=True,exist_ok=True)
                identity={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'controls_hashes':hashes,
                          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                if (dest/'complete.json').exists():
                    result=json.loads((dest/'complete.json').read_text());assert result['identity']==identity
                    per_case=json.loads((dest/'cases.json').read_text())
                else:
                    state=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=state['config']
                    trainmeta=cfg['feature_metadata'][0] if architecture=='frame' else cfg['unit_metadata'][0]['feature_metadata']
                    assert featuremeta['padding_policy']==trainmeta['padding_policy']=='exact_convolution_lengths'
                    for field in ['model','source_sha256','normalize','stride_samples','layer']:assert trainmeta[field]==featuremeta[field],field
                    assert cfg['condition']==condition and cfg['seed']==seed
                    model=FrameHead(rows[0][1].shape[1],'temporal').cuda();model.load_state_dict(state['model']);model.eval()
                    threshold=float(state['metrics']['dev_selected_spoof_threshold'])
                    if architecture=='phone':predictions=predict_phone(model,rows,labels)
                    else:
                        predictions={}
                        with torch.inference_mode():
                            for batch,x,y,m,lengths,fm in batches(rows,32,list(range(len(rows)))):
                                scores=torch.sigmoid(align_logits(model(x,fm),lengths,y.shape[1],**cfg['geometry'])).cpu().numpy()
                                for i,(uid,_,yy) in enumerate(batch):predictions[uid]=scores[i,:len(yy)]
                    result,per_case=summarize_predictions(cases,labels,masks,predictions,threshold)
                    result.update(identity=identity,checkpoint=str(checkpoint),locked_dev_threshold=threshold,dev_epoch=state['metrics']['epoch'],selected_dev_metrics=state['metrics'],cases=len(cases))
                    np.savez(dest/'predictions.npz',**predictions);(dest/'cases.json').write_text(json.dumps(per_case))
                    (dest/'complete.json').write_text(json.dumps(result,indent=2));del model;torch.cuda.empty_cache()
                records[(architecture,condition,seed)]=per_case;all_results[f'{architecture}/{condition}/seed{seed}']=result
                print(architecture,condition,seed,json.dumps(result['effects']),flush=True)
    analysis=joint_bootstrap(records,a.seeds)
    (out/'interaction.json').write_text(json.dumps(analysis,indent=2));(out/'summary.json').write_text(json.dumps(all_results,indent=2))
    print('Matched evaluation complete',flush=True)

if __name__=='__main__':main()
