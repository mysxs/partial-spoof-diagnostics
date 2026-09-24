#!/usr/bin/env python3
"""Acoustic unit caches with raw-label occupancy targets and explicit fallback.

Unit boundaries and input features never depend on spoof reference labels.
Missing/invalid alignments use a fixed 80 ms grid, retaining official trials.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def overlap_weights(intervals,edges):
    a=np.asarray(intervals,dtype='float64')
    overlap=np.maximum(0,np.minimum(a[:,1,None],edges[None,1:])-np.maximum(a[:,0,None],edges[None,:-1]))
    sums=overlap.sum(axis=1)
    assert (sums>0).all(),'Empty unit-to-time intersection'
    return overlap/sums[:,None]


def partition(anchors,duration,gap_max=.08):
    """Disjoint ordered units, including every non-speech interval."""
    result=[];kind=[];cursor=0.
    for s,e in sorted(anchors)+[(duration,duration)]:
        s=float(np.clip(s,0,duration));e=float(np.clip(e,0,duration))
        if s<cursor-1e-5:raise ValueError('Overlapping linguistic anchors')
        s=max(s,cursor)
        if s>cursor+1e-8:
            n=max(1,int(np.ceil((s-cursor)/gap_max)))
            b=np.linspace(cursor,s,n+1)
            result.extend(zip(b[:-1],b[1:]));kind.extend([0]*n)
        if e>s+1e-8:
            result.append((s,e));kind.append(1)
        cursor=max(cursor,e)
    assert result and abs(sum(e-s for s,e in result)-duration)<1e-5
    return np.asarray(result),np.asarray(kind,dtype='uint8')


def feature_pool(features,units,duration,center=.0125,step=.02):
    centers=center+np.arange(len(features))*step
    edges=np.r_[0,(centers[:-1]+centers[1:])/2,duration]
    assert np.all(np.diff(edges)>0),(len(features),duration)
    return (overlap_weights(units,edges)@features.astype('float32')).astype('float16')


def occupancy(values,units):
    labels=np.asarray(values).astype('int8')
    assert np.isin(labels,[0,1]).all()
    fake=(labels==0).astype(float)
    grid=np.arange(len(fake)+1)*.01;cum=np.r_[0,np.cumsum(fake)*.01]
    integrals=np.interp(units,grid,cum)
    return np.clip(np.diff(integrals,axis=1)[:,0]/np.diff(units,axis=1)[:,0],0,1).astype('float32')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--split',required=True);ap.add_argument('--features',required=True)
    ap.add_argument('--textgrids',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--ids',default='')
    a=ap.parse_args()
    import soundfile as sf
    from praatio import textgrid
    root=Path(a.root);features=Path(a.features);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ids=Path(a.ids).read_text().splitlines() if a.ids else (root/'database'/a.split/(a.split+'.lst')).read_text().splitlines()
    grid_paths={p.stem:p for p in Path(a.textgrids).rglob('*.TextGrid')}
    metadata=json.loads((features/'metadata.json').read_text())
    labels=np.load(root/'database/segment_labels'/f'{a.split}_seglab_0.01.npy',allow_pickle=True).item()
    config={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'split':a.split,'feature_metadata':metadata,'gap_max_seconds':.08,'fallback':'80ms_grid',
        'textgrids':a.textgrids,'ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    if (out/'metadata.json').exists():assert json.loads((out/'metadata.json').read_text())==config,'Provenance mismatch'
    else:(out/'metadata.json').write_text(json.dumps(config,indent=2))
    failures=[];count_phone=count_word=0;records=[]
    for index,uid in enumerate(ids):
        destination=out/(uid+'.npz');record_path=out/(uid+'.json')
        if destination.exists() and record_path.exists():
            row=json.loads(record_path.read_text());records.append(row)
            if row['fallback']:failures.append(row)
            count_phone+=row['phone_n'];count_word+=row['word_n'];continue
        duration=sf.info(str(root/'database'/a.split/'con_wav'/(uid+'.wav'))).duration
        fallback=None
        try:
            tg=textgrid.openTextgrid(str(grid_paths[uid]),includeEmptyIntervals=True)
            if abs(float(tg.maxTimestamp)-duration)>.02:raise ValueError('TextGrid/audio duration mismatch')
            phone=[(x.start,x.end) for x in tg.getTier('phones').entries if x.label not in ('','sil','sp') and x.end>x.start]
            word=[(x.start,x.end) for x in tg.getTier('words').entries if x.label and x.end>x.start]
            if not phone or not word:raise ValueError('Empty phone or word alignment')
            pu,pk=partition(phone,duration);wu,wk=partition(word,duration)
        except Exception as exc:
            fallback=str(exc);pu,pk=partition([],duration);wu,wk=partition([],duration)
        x=np.load(features/(uid+'.npy'),allow_pickle=False)
        kwargs={'center':metadata['first_center_seconds'],'step':metadata['stride_samples']/16000}
        px=feature_pool(x,pu,duration,**kwargs);wx=feature_pool(x,wu,duration,**kwargs)
        # Word cells form a partition, hence this map is a true duration mean.
        pw=overlap_weights(pu,np.r_[wu[:,0],wu[-1,1]]).astype('float32')
        temp=out/(uid+'.tmp')
        with temp.open('wb') as f:
            np.savez(f,phone_x=px,word_x=wx,phone_intervals=pu.astype('float32'),word_intervals=wu.astype('float32'),
                phone_kind=pk,word_kind=wk,phone_target=occupancy(labels[uid],pu),word_target=occupancy(labels[uid],wu),
                phone_to_word=pw,duration=np.array(duration),label_horizon=np.array(len(labels[uid])*.01))
        temp.replace(destination)
        row={'uttid':uid,'fallback':fallback,'phone_n':len(pu),'word_n':len(wu),'duration':duration,
             'label_horizon':len(labels[uid])*.01,'textgrid':str(grid_paths.get(uid,''))}
        record_path.write_text(json.dumps(row));records.append(row)
        if fallback:failures.append(row)
        count_phone+=len(pu);count_word+=len(wu)
        if (index+1)%1000==0:print(a.split,index+1,'/',len(ids),'fallback',len(failures),flush=True)
    report={'utterances':len(ids),'fallbacks':failures,'phone_units':count_phone,'word_units':count_word,
            'notes':['No reference-driven anchor selection.','All official utterances retained, including failures.',
                     'Speech/gap type is retained; spn is treated as a spoken unknown phone.']}
    (out/'utterances.json').write_text(json.dumps(records))
    (out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
