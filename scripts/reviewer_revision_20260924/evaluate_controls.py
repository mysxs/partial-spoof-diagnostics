#!/usr/bin/env python3
"""Paired control diagnostics at the previously selected dev threshold."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_frame import FrameHead, batches, align_logits


def confusion(y,p,mask):
    y=y.astype(bool);p=p.astype(bool)
    return np.array([(mask&y&p).sum(),(mask&~y&p).sum(),(mask&y&~p).sum(),(mask&~y&~p).sum()],dtype='int64')


def rates(c):
    tp,fp,fn,tn=map(int,c)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'fpr':fp/(fp+tn) if fp+tn else None,'fnr':fn/(fn+tp) if fn+tp else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--controls',required=True)
    ap.add_argument('--runs',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--features',default='')
    a=ap.parse_args();torch.set_num_threads(8)
    root=Path(a.controls);runs=Path(a.runs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    feature_root=Path(a.features) if a.features else root/'features'
    assert (feature_root/'complete.json').exists()
    cases=json.loads((root/'controls.json').read_text());labels=np.load(root/'labels20_spoof1.npz')
    masks=np.load(root/'core_masks20.npz');ids=(root/'ids.txt').read_text().splitlines()
    rows=[(uid,np.load(feature_root/(uid+'.npy')),labels[uid].astype('float32')) for uid in ids]
    all_results={}
    checkpoints=sorted(runs.glob('*_seed*/best.pt'));assert checkpoints
    for checkpoint in checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=state['config'];model=FrameHead(rows[0][1].shape[1],cfg['architecture']).cuda()
        model.load_state_dict(state['model']);model.eval();predictions={}
        threshold=state['metrics']['dev_selected_spoof_threshold']
        with torch.inference_mode():
            for batch,x,y,m,lengths,fm in batches(rows,32,list(range(len(rows)))):
                z=align_logits(model(x,fm),lengths,y.shape[1],**cfg['geometry'])
                scores=torch.sigmoid(z).cpu().numpy()
                for i,(uid,_,yy) in enumerate(batch):predictions[uid]=scores[i,:len(yy)]
        np.savez(out/(checkpoint.parent.name+'_scores.npz'),**predictions)
        per_case=[];speaker_stats={};totals={}
        for case in cases:
            counts={}
            for name,variant in case['variants'].items():
                uid=variant['uttid'];y=labels[uid];p=predictions[uid]>threshold
                for scope,mask in [('full',np.ones(len(y),bool)),('core',masks[uid])]:
                    key=name+'_'+scope;c=confusion(y,p,mask);counts[key]=c.tolist()
                    totals[key]=totals.get(key,np.zeros(4,dtype='int64'))+c
                    speaker_stats.setdefault(case['protocol_speaker'],{}).setdefault(key,np.zeros(4,dtype='int64'))
                    speaker_stats[case['protocol_speaker']][key]+=c
            per_case.append({'case':case['case'],'speaker':case['protocol_speaker'],'counts':counts})
        # Resample speakers, preserving all paired conditions within a cluster.
        speakers=sorted(speaker_stats);rng=np.random.default_rng(20260919);draws=[]
        for _ in range(2000):
            chosen=rng.choice(speakers,len(speakers),replace=True)
            pooled={k:sum((speaker_stats[s][k] for s in chosen),np.zeros(4,dtype='int64')) for k in totals}
            r={k:rates(v) for k,v in pooled.items()}
            effect=[r['gg_raw_full']['fpr']-r['genuine_full']['fpr'],
                r['gf_smooth20_core']['fnr']-r['gf_raw_core']['fnr'],
                r['gf_smooth50_core']['fnr']-r['gf_raw_core']['fnr']]
            for width in (5,20,50):
                if f'gg_smooth{width}_full' in r:effect.append(r[f'gg_smooth{width}_full']['fpr']-r['gg_raw_full']['fpr'])
            draws.append(effect)
        names=['gg_minus_genuine_fpr_full','smooth20_minus_raw_fnr_core','smooth50_minus_raw_fnr_core']
        draws=np.asarray(draws);summaries={k:rates(v) for k,v in totals.items()}
        effects=[summaries['gg_raw_full']['fpr']-summaries['genuine_full']['fpr'],
                 summaries['gf_smooth20_core']['fnr']-summaries['gf_raw_core']['fnr'],
                 summaries['gf_smooth50_core']['fnr']-summaries['gf_raw_core']['fnr']]
        for width in (5,20,50):
            if f'gg_smooth{width}_full' in summaries:
                names.append(f'gg_smooth{width}_minus_raw_fpr_full')
                effects.append(summaries[f'gg_smooth{width}_full']['fpr']-summaries['gg_raw_full']['fpr'])
        result={'checkpoint':str(checkpoint),'dev_epoch':state['metrics']['epoch'],'locked_dev_threshold':threshold,
            'cases':len(cases),'speakers':len(speakers),'conditions':summaries,
            'paired_effects':{k:{'estimate':effects[i],'speaker_bootstrap_95ci':np.quantile(draws[:,i],[.025,.975]).tolist()} for i,k in enumerate(names)},
            'limitations':['Constructed same-speaker but different-content controls; not official benchmark results.',
              'Threshold inherited from original dev; no tuning on controls.',
              'Core excludes a fixed 60ms band around insertion boundaries in every condition.',
              'Speaker bootstrap can be unstable with few speaker clusters; speaker count is reported.']}
        (out/(checkpoint.parent.name+'.json')).write_text(json.dumps(result,indent=2))
        (out/(checkpoint.parent.name+'_cases.json')).write_text(json.dumps(per_case))
        all_results[checkpoint.parent.name]=result
        print(checkpoint.parent.name,json.dumps(result['paired_effects']),flush=True)
    (out/'summary.json').write_text(json.dumps(all_results,indent=2))

if __name__=='__main__':main()
