#!/usr/bin/env python3
"""Matched frozen-WavLM frame baselines; dev-only checkpoint selection.

These are transparent probes, not a reproduction of CFPRF or a fine-tuned SSL
baseline. Predictions are mapped by physical feature centers, not length resize.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrameHead(nn.Module):
    def __init__(self, dim, architecture):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.architecture=architecture
        if architecture=='linear':self.head=nn.Linear(dim,1)
        else:
            self.project=nn.Linear(dim,128)
            self.convs=nn.ModuleList([nn.Conv1d(128,128,5,padding=2*d,dilation=d) for d in (1,2,4)])
            self.head=nn.Linear(128,1)

    def forward(self,x,mask):
        x=self.norm(x)
        if self.architecture=='temporal':
            x=F.gelu(self.project(x))*mask[:,:,None]
            for conv in self.convs:
                x=(x+F.gelu(conv(x.transpose(1,2)).transpose(1,2)))*mask[:,:,None]
        return self.head(x).squeeze(-1)


def align_logits(logits,lengths,nout,center=.0125,step=.02):
    """Interpolate onto 20 ms reference centers, with constant edge extension."""
    pos=((torch.arange(nout,device=logits.device)+.5)*.02-center)/step
    pos=pos[None,:].expand(len(lengths),-1).clamp(min=0)
    pos=torch.minimum(pos,(lengths-1)[:,None])
    lo=pos.floor().long();hi=torch.minimum(lo+1,(lengths-1)[:,None])
    w=pos-lo
    return logits.gather(1,lo)*(1-w)+logits.gather(1,hi)*w


def spoof_targets(values):
    values=np.asarray(values).astype(np.int8)
    assert np.isin(values,[0,1]).all(),'Unexpected official reference label'
    return (values==0).astype('float32')


def load_split(root,features,split,limit=0,soft_targets=False):
    ids=(root/'database'/split/(split+'.lst')).read_text().splitlines()
    if limit:
        # Fixed randomized pilot, never the first all-bonafide block.
        ids=np.random.default_rng(20260919).permutation(ids)[:limit].tolist()
    labels=np.load(root/'database/segment_labels'/f'{split}_seglab_0.02.npy',allow_pickle=True).item()
    raw=np.load(root/'database/segment_labels'/f'{split}_seglab_0.01.npy',allow_pickle=True).item() if soft_targets else None
    data=[];nbytes=0
    for i,uid in enumerate(ids):
        x=np.load(features/split/(uid+'.npy'),allow_pickle=False)
        y=spoof_targets(labels[uid])
        if soft_targets:
            v=spoof_targets(raw[uid]);starts=np.arange(0,len(v),2)
            soft=np.add.reduceat(v,starts)/np.minimum(2,len(v)-starts)
            assert len(soft)==len(y)
            y=soft.astype('float32')
        assert x.ndim==2 and len(x)>0 and len(y)>0 and np.isfinite(x).all(),uid
        data.append((uid,x,y));nbytes+=x.nbytes+y.nbytes
        if i%5000==0:print(f'Loaded {split} {i+1}/{len(ids)}',flush=True)
    print(f'{split}: {len(data)} utterances, {nbytes/1e9:.3f} GB RAM',flush=True)
    positives=sum(float(y.sum()) for _,_,y in data)
    total=sum(len(y) for _,_,y in data)
    assert 0<positives<total,(split,positives,total)
    print(f'{split}: {positives}/{total} spoof frames',flush=True)
    return data


def batches(data,batch_size,order):
    for start in range(0,len(order),batch_size):
        rows=[data[i] for i in order[start:start+batch_size]]
        nf=max(len(r[1]) for r in rows);ny=max(len(r[2]) for r in rows)
        x=np.zeros((len(rows),nf,rows[0][1].shape[1]),dtype='float16')
        y=np.zeros((len(rows),ny),dtype='float32');m=np.zeros_like(y,dtype=bool)
        lengths=[]
        for i,(_,xx,yy) in enumerate(rows):
            x[i,:len(xx)]=xx;y[i,:len(yy)]=yy;m[i,:len(yy)]=True;lengths.append(len(xx))
        x=torch.from_numpy(x).to('cuda',dtype=torch.float32)
        lengths=torch.tensor(lengths,device='cuda')
        fm=torch.arange(nf,device='cuda')[None,:]<lengths[:,None]
        yield rows,x,torch.from_numpy(y).to('cuda'),torch.from_numpy(m).to('cuda'),lengths,fm


def evaluate(model,data,batch,geometry,compute_eer):
    from sklearn.metrics import roc_curve
    model.eval();all_y=[];all_s=[];per_utt={}
    with torch.inference_mode():
        for rows,x,y,m,lengths,fm in batches(data,batch,list(range(len(data)))):
            logits=align_logits(model(x,fm),lengths,y.shape[1],**geometry)
            scores=torch.sigmoid(logits).cpu().numpy()
            for i,(uid,_,yy) in enumerate(rows):
                s=scores[i,:len(yy)];per_utt[uid]=s;all_y.append(yy);all_s.append(s)
    y=np.concatenate(all_y).astype(bool);s=np.concatenate(all_s)
    assert y.any() and (~y).any(),'Evaluation needs both frame classes'
    eer,bonath=compute_eer(1-s[~y],1-s[y]);threshold=1-float(bonath)
    pred=s>threshold
    tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False);fnr=1-tpr;d=fpr-fnr
    hi=int(np.flatnonzero(d>=0)[0]);lo=max(0,hi-1)
    w=-d[lo]/(d[hi]-d[lo]) if hi!=lo else 0.
    diagnostic=float(fpr[lo]+w*(fpr[hi]-fpr[lo]))
    metrics={'segment20_eer':float(eer),'tie_aware_diagnostic_eer':diagnostic,
             'dev_selected_spoof_threshold':threshold,'dev_f1_at_selected_threshold':2*tp/max(2*tp+fp+fn,1),
             'frames':len(y),'utterances':len(data),'tp':tp,'fp':fp,'fn':fn}
    return metrics,per_utt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--features',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--official-repo',required=True)
    ap.add_argument('--architectures',nargs='+',default=['linear','temporal'],choices=['linear','temporal'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--soft-targets',action='store_true',help='Train on raw 10ms duration occupancy; dev remains official hard 20ms labels')
    a=ap.parse_args();torch.set_num_threads(8)
    assert torch.cuda.is_available()
    sys.path.insert(0,a.official_repo)
    from sandbox.eval_asvspoof import compute_eer
    root=Path(a.root);features=Path(a.features);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    metas=[json.loads((features/s/'metadata.json').read_text()) for s in ['train','dev']]
    for k in ['model','model_size','source_sha256','normalize','stride_samples','receptive_field_samples','layer']:
        assert metas[0][k]==metas[1][k],k
    geometry={'center':metas[0]['first_center_seconds'],'step':metas[0]['stride_samples']/16000}
    train=load_split(root,features,'train',a.limit,a.soft_targets);dev=load_split(root,features,'dev',a.limit)
    assert not ({r[0] for r in train}&{r[0] for r in dev})
    for arch in a.architectures:
        for seed in a.seeds:
            run=out/f'{arch}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
            if (run/'complete.json').exists():print('Already complete',run,flush=True);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
            model=FrameHead(train[0][1].shape[1],arch).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0001)
            cfg=vars(a)|{'architecture':arch,'seed':seed,'geometry':geometry,
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'official_eer_sha256':hashlib.sha256((Path(a.official_repo)/'sandbox/eval_asvspoof.py').read_bytes()).hexdigest(),
                'trainable_parameters':sum(p.numel() for p in model.parameters()),'features_metadata':metas,
                'limitations':['Frozen last-layer WavLM probes, not a fine-tuned strong baseline.','Dev checkpoint/threshold selection; eval/external evaluation remains separate.']}
            (run/'config.json').write_text(json.dumps(cfg,indent=2))
            best=float('inf');history=[];t0=time.time()
            resume=run/'last.pt'
            if resume.exists():
                state=torch.load(resume,map_location='cuda',weights_only=False)
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
                history=state['history'];best=state['best']
            for epoch in range(len(history),a.epochs):
                model.train();loss_sum=0.;nframes=0
                order=np.random.default_rng(seed*1000+epoch).permutation(len(train)).tolist()
                for step,(_,x,y,m,lengths,fm) in enumerate(batches(train,a.batch,order)):
                    opt.zero_grad(set_to_none=True)
                    z=align_logits(model(x,fm),lengths,y.shape[1],**geometry)
                    loss=F.binary_cross_entropy_with_logits(z[m],y[m])
                    assert torch.isfinite(loss)
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    n=int(m.sum());loss_sum+=float(loss.detach())*n;nframes+=n
                    if step%200==0:print(f'{arch} seed={seed} epoch={epoch+1} step={step} BCE={float(loss.detach()):.4f}',flush=True)
                metrics,pred=evaluate(model,dev,a.batch,geometry,compute_eer)
                row={'epoch':epoch+1,'train_bce':loss_sum/nframes,'elapsed_s':time.time()-t0,**metrics}
                history.append(row);print(json.dumps(row),flush=True)
                if metrics['segment20_eer']<best:
                    best=metrics['segment20_eer']
                    torch.save({'model':model.state_dict(),'config':cfg,'metrics':row},run/'best.pt')
                    np.savez(run/'dev_predictions.npz',**pred)
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'best':best},run/'last.tmp')
                (run/'last.tmp').replace(resume)
                (run/'history.json').write_text(json.dumps(history,indent=2))
            (run/'complete.json').write_text(json.dumps({'best_dev_eer':best,'epochs':len(history),'elapsed_s':time.time()-t0},indent=2))


if __name__=='__main__':main()
