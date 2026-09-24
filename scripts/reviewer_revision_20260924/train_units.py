#!/usr/bin/env python3
"""Frozen acoustic unit models, duration-weighted occupancy, complete dev trials.

Phone+word context has a same-parameter duplicated-phone control. Both use the
same duration-mean word auxiliary loss; no noisy-OR occupancy aggregation.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_frame import FrameHead, spoof_targets


def project_scores(prob,units,n20):
    units=np.asarray(units,dtype='float64');prob=np.asarray(prob,dtype='float64')
    boundaries=np.r_[units[:,0],units[-1,1]]
    mass=np.r_[0,np.cumsum(prob*np.diff(boundaries))]
    edges=np.arange(n20+1)*.02
    values=np.interp(edges,boundaries,mass)+np.maximum(0,edges-boundaries[-1])*prob[-1]
    return np.clip(np.diff(values)/.02,0,1).astype('float32')


def reverse_word_map(phone,word):
    overlap=np.maximum(0,np.minimum(word[:,1,None],phone[None,:,1])-np.maximum(word[:,0,None],phone[None,:,0]))
    return (overlap/overlap.sum(axis=1,keepdims=True)).astype('float32')


def load_data(root,caches,split,limit):
    ids=(root/'database'/split/(split+'.lst')).read_text().splitlines()
    if limit:ids=np.random.default_rng(20260919).permutation(ids)[:limit].tolist()
    references=np.load(root/'database/segment_labels'/f'{split}_seglab_0.02.npy',allow_pickle=True).item()
    rows=[]
    for i,uid in enumerate(ids):
        with np.load(caches/split/(uid+'.npz'),allow_pickle=False) as f:r={k:f[k] for k in f.files}
        r['uttid']=uid;r['reference']=spoof_targets(references[uid])
        r['context_x']=(r['phone_to_word']@r['word_x'].astype('float32')).astype('float16')
        r['word_from_phone']=reverse_word_map(r['phone_intervals'],r['word_intervals'])
        assert np.isfinite(r['context_x']).all()
        rows.append(r)
        if i%5000==0:print('Loaded',split,i+1,'/',len(ids),flush=True)
    assert 0<sum(r['reference'].sum() for r in rows)<sum(len(r['reference']) for r in rows)
    return rows


def unit_batches(data,order,batch,mode):
    key='word' if mode=='word' else 'phone'
    for start in range(0,len(order),batch):
        rows=[data[i] for i in order[start:start+batch]]
        length=max(len(r[key+'_x']) for r in rows);nw=max(len(r['word_x']) for r in rows)
        dim=rows[0][key+'_x'].shape[1]*(2 if mode in ['phone_word','phone_duplicate'] else 1)
        x=np.zeros((len(rows),length,dim),'float16');target=np.zeros((len(rows),length),'float32')
        duration=np.zeros_like(target);word_target=np.zeros((len(rows),nw),'float32');word_duration=np.zeros_like(word_target)
        mapping=np.zeros((len(rows),nw,length),'float32')
        for j,r in enumerate(rows):
            xx=r[key+'_x'];n=len(xx);w=len(r['word_x'])
            if mode in ['phone_word','phone_duplicate']:
                xx=np.concatenate([xx,r['context_x'] if mode=='phone_word' else xx],axis=1)
            x[j,:n]=xx;target[j,:n]=r[key+'_target'];duration[j,:n]=np.diff(r[key+'_intervals'])[:,0]
            word_target[j,:w]=r['word_target'];word_duration[j,:w]=np.diff(r['word_intervals'])[:,0]
            if key=='phone':mapping[j,:w,:n]=r['word_from_phone']
        arrays=[x,target,duration,word_target,word_duration,mapping]
        tensors=[torch.from_numpy(v).to('cuda',dtype=torch.float32) for v in arrays]
        yield rows,tensors


def evaluate(model,data,mode,batch,compute_eer):
    model.eval();predictions={};all_y=[];all_s=[]
    key='word' if mode=='word' else 'phone'
    with torch.inference_mode():
        for rows,(x,target,duration,wt,wd,mapping) in unit_batches(data,list(range(len(data))),batch,mode):
            p=torch.sigmoid(model(x,duration>0)).cpu().numpy()
            for i,r in enumerate(rows):
                scores=project_scores(p[i,:len(r[key+'_x'])],r[key+'_intervals'],len(r['reference']))
                predictions[r['uttid']]=scores;all_y.append(r['reference']);all_s.append(scores)
    y=np.concatenate(all_y).astype(bool);s=np.concatenate(all_s)
    eer,threshold=compute_eer(1-s[~y],1-s[y]);threshold=1-float(threshold)
    p=s>threshold;tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum())
    return {'segment20_eer':float(eer),'dev_selected_spoof_threshold':threshold,
            'dev_f1_at_selected_threshold':2*tp/max(2*tp+fp+fn,1),'utterances':len(data),'frames':len(y)},predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--units',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--official-repo',required=True)
    ap.add_argument('--modes',nargs='+',default=['phone','word','phone_word','phone_duplicate'],choices=['phone','word','phone_word','phone_duplicate'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--lr',type=float,default=.0003);ap.add_argument('--word-loss',type=float,default=.2)
    ap.add_argument('--hard-targets',action='store_true');ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();torch.set_num_threads(8);assert torch.cuda.is_available()
    sys.path.insert(0,a.official_repo)
    from sandbox.eval_asvspoof import compute_eer
    root=Path(a.root);units=Path(a.units);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    metadata=[json.loads((units/s/'metadata.json').read_text()) for s in ['train','dev']]
    assert metadata[0]['script_sha256']==metadata[1]['script_sha256']
    train=load_data(root,units,'train',a.limit);dev=load_data(root,units,'dev',a.limit)
    assert not ({r['uttid'] for r in train}&{r['uttid'] for r in dev})
    for mode in a.modes:
        for seed in a.seeds:
            run=out/f'{mode}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
            if (run/'complete.json').exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
            torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
            dim=train[0]['phone_x'].shape[1]*(2 if mode in ['phone_word','phone_duplicate'] else 1)
            model=FrameHead(dim,'temporal').cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0001)
            cfg=vars(a)|{'mode':mode,'seed':seed,'architecture':'temporal','input_dim':dim,
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'frame_head_sha256':hashlib.sha256(Path(__file__).with_name('train_frame.py').read_bytes()).hexdigest(),
                'unit_metadata':metadata,'parameters':sum(p.numel() for p in model.parameters()),
                'limitations':['Frozen backbone; no learned offsets or BAIC in this ablation.',
                  'Duration-weighted 10ms occupancy differs from hard 20ms frame targets; matched soft-target frame control is required.',
                  'Fixed-grid fallbacks retain failed-alignment trials.']}
            (run/'config.json').write_text(json.dumps(cfg,indent=2));history=[];best=float('inf');t0=time.time()
            if (run/'last.pt').exists():
                state=torch.load(run/'last.pt',map_location='cuda',weights_only=False)
                assert state['config']==cfg,'Resume configuration changed'
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);history=state['history'];best=state['best']
            for epoch in range(len(history),a.epochs):
                model.train();total_loss=total_seconds=0.
                order=np.random.default_rng(seed*1000+epoch).permutation(len(train)).tolist()
                for step,(_,tensors) in enumerate(unit_batches(train,order,a.batch,mode)):
                    x,target,duration,wt,wd,mapping=tensors
                    if a.hard_targets:target=(target>=.5-1e-7).float();wt=(wt>=.5-1e-7).float()
                    opt.zero_grad(set_to_none=True);z=model(x,duration>0)
                    loss=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*duration).sum()/duration.sum()
                    if mode in ['phone_word','phone_duplicate'] and a.word_loss:
                        wp=torch.bmm(mapping,torch.sigmoid(z).unsqueeze(-1)).squeeze(-1).clamp(1e-6,1-1e-6)
                        loss+=a.word_loss*(F.binary_cross_entropy(wp,wt,reduction='none')*wd).sum()/wd.sum()
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    seconds=float(duration.sum());total_seconds+=seconds;total_loss+=float(loss.detach())*seconds
                    if step%200==0:print(mode,seed,epoch+1,step,'loss',float(loss.detach()),flush=True)
                metrics,predictions=evaluate(model,dev,mode,a.batch,compute_eer)
                row={'epoch':epoch+1,'train_loss':total_loss/total_seconds,'elapsed_s':time.time()-t0,**metrics}
                history.append(row);print(mode,seed,json.dumps(row),flush=True)
                if metrics['segment20_eer']<best:
                    best=metrics['segment20_eer'];torch.save({'model':model.state_dict(),'config':cfg,'metrics':row},run/'best.pt')
                    np.savez(run/'dev_predictions.npz',**predictions)
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'best':best,'config':cfg},run/'last.tmp')
                (run/'last.tmp').replace(run/'last.pt');(run/'history.json').write_text(json.dumps(history,indent=2))
            (run/'complete.json').write_text(json.dumps({'best_dev_eer':best,'epochs':len(history),'elapsed_s':time.time()-t0},indent=2))

if __name__=='__main__':main()
