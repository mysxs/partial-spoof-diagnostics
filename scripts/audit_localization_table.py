"""Recompute frame and event metrics from frozen predictions; no threshold fitting.

PS dev results are development diagnostics. PS eval analysis is retrospective.
Requires original official labels (0=spoof), frozen predictions, and training logs.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import numpy as np
from event_strata import spans, match_events, duration_bin


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for key in ['run-root','data-root','official-repo','out']: ap.add_argument('--'+key,required=True)
    a=ap.parse_args(); root=Path(a.run_root); data=Path(a.data_root); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,a.official_repo)
    from sandbox.eval_asvspoof import compute_eer
    arms={'Hard frame':('frame_baselines_v2','temporal','eval_frames_v2/base'),
          'Soft frame':('frame_soft_v2','temporal','eval_frames_v2/base'),
          'Phone':('unit_models_v2','phone','eval_units_v2/full'),
          'Word':('unit_models_v2','word','eval_units_v2/full')}
    rows=[]
    for split in ['dev','eval']:
        ip=data/'database'/split/(split+'.lst');ids=ip.read_text().splitlines()
        lp10=data/'database/segment_labels'/f'{split}_seglab_0.01.npy'
        lp20=data/'database/segment_labels'/f'{split}_seglab_0.02.npy'
        labs10=np.load(lp10,allow_pickle=True).item();labs20=np.load(lp20,allow_pickle=True).item()
        assert len(ids)==len(set(ids))=={'dev':24844,'eval':71237}[split]
        # Original label arrays and their tails are retained, without clipping to waveform duration.
        targets={}; references={}
        for uid in ids:
            for labs in [labs10,labs20]:assert set(np.unique(labs[uid]).astype(str))<={'0','1'},uid
            targets[uid]=np.asarray(labs20[uid]).astype(np.int8)==0
            references[uid]=spans(np.asarray(labs10[uid]).astype(np.int8)==0,.01)
        y=np.concatenate([targets[i] for i in ids]);labelhash={'ids':sha(ip),'labels10':sha(lp10),'labels20':sha(lp20)}
        for arm,(folder,prefix,evalfolder) in arms.items():
            for seed in [1,2,3]:
                run=root/folder/f'{prefix}_seed{seed}'
                history=json.loads((run/'history.json').read_text()); best=min(history,key=lambda r:r['segment20_eer'])
                threshold=best['dev_selected_spoof_threshold'];cp=run/'best.pt'
                predpath=run/'dev_predictions.npz' if split=='dev' else root/evalfolder/folder/run.name/'predictions.npz'
                source=json.loads((run/'complete.json' if split=='dev' else predpath.parent/'complete.json').read_text())
                checkpoint_hash=sha(cp)
                if split=='eval':
                    assert source['identity']['checkpoint_sha256']==checkpoint_hash
                    assert abs(threshold-source['dev_locked_spoof_threshold'])<1e-12
                    assert source['identity']['split']=='eval' and source['utterances']==len(ids)
                ident={'checkpoint_sha256':checkpoint_hash,'predictions_sha256':sha(predpath),'history_sha256':sha(run/'history.json'),
                       'config_sha256':sha(run/'config.json'),'script_sha256':sha(__file__),'matching_code_sha256':sha(Path(__file__).with_name('event_strata.py')),
                       'official_metric_sha256':sha(Path(a.official_repo)/'sandbox/eval_asvspoof.py'),**labelhash}
                dest=out/f'{split}_{folder}_{prefix}_seed{seed}.json'
                if dest.exists():
                    row=json.loads(dest.read_text());assert row['identity']==ident
                    rows.append(row);print('REUSED',dest.name,flush=True);continue
                pieces=[];tp=fp=fn=tn=nr=npred=nm=nshort=nshortmatch=0;errstart=errend=0.;per=[]
                with np.load(io.BytesIO(predpath.read_bytes()),allow_pickle=False) as pred:
                    assert len(pred.files)==len(ids) and set(pred.files)==set(ids)
                    for uid in ids:
                        score=pred[uid];target=targets[uid]
                        assert score.shape==target.shape and np.isfinite(score).all() and ((score>=0)&(score<=1)).all(),uid
                        pieces.append(score);mask=score>threshold
                        ct=[int(np.sum(mask&target)),int(np.sum(mask&~target)),int(np.sum(~mask&target)),int(np.sum(~mask&~target))]
                        tp+=ct[0];fp+=ct[1];fn+=ct[2];tn+=ct[3]
                        ref=references[uid];proposal=spans(mask,.02);matches=match_events(ref,proposal)
                        short={i for i,(s,e) in enumerate(ref) if duration_bin(e-s)=='lt100'}
                        se=sum(abs(ref[r,0]-proposal[p,0]) for r,p,_ in matches)
                        ee=sum(abs(ref[r,1]-proposal[p,1]) for r,p,_ in matches)
                        shortmatches=sum(r in short for r,p,_ in matches)
                        nr+=len(ref);npred+=len(proposal);nm+=len(matches);nshort+=len(short);nshortmatch+=shortmatches
                        errstart+=se;errend+=ee
                        per.append([uid,len(ref),len(proposal),len(matches),len(short),shortmatches,se,ee,*ct])
                scores=np.concatenate(pieces)
                eer,_=compute_eer(1-scores[~y],1-scores[y])
                expected=best['segment20_eer'] if split=='dev' else source['segment20_eer']
                assert abs(eer-expected)<1e-10,(arm,seed,split,eer,expected)
                if split=='eval': assert all(v==source[k] for k,v in zip(['tp','fp','fn','tn'],[tp,fp,fn,tn]))
                row={'split':split,'arm':arm,'seed':seed,'layer':'last','selected_epoch':best['epoch'],'threshold':threshold,
                     'utterances':len(ids),'frames':len(y),'frame_eer_pct':100*float(eer),
                     'frame_f1_pct':100*2*tp/(2*tp+fp+fn),'frame_recall_pct':100*tp/(tp+fn),
                     'event_precision_pct':100*nm/npred,'event_recall_pct':100*nm/nr,'event_f1_pct':100*2*nm/(nr+npred),
                     'short_event_recall_pct':100*nshortmatch/nshort,'matched_boundary_mae_ms':1000*(errstart+errend)/(2*nm),
                     'reference_events':nr,'predicted_events':npred,'matched_events':nm,'short_reference_events':nshort,
                     'short_matched_events':nshortmatch,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'identity':ident}
                if split=='dev':
                    previous=json.loads((root/'event_strata_v2/full'/folder/run.name/'complete.json').read_text())
                    for metric in ['event_precision','event_recall','event_f1']:
                        assert abs(row[metric+'_pct']/100-previous[metric])<1e-12
                    row['previous_dev_event_metrics_verified']=True
                with (out/(dest.stem+'_utterances.csv')).open('w') as f:
                    writer=csv.writer(f);writer.writerow(['uttid','reference_events','predicted_events','matched_events','short_reference_events','short_matched_events','start_error_sum_s','end_error_sum_s','frame_tp','frame_fp','frame_fn','frame_tn']);writer.writerows(per)
                dest.write_text(json.dumps(row,indent=2));rows.append(row)
                print('PASS',split,arm,seed,'EER',row['frame_eer_pct'],'event F1',row['event_f1_pct'],flush=True)
                del scores,pieces,per
    metrics=['frame_eer_pct','frame_f1_pct','frame_recall_pct','event_precision_pct','event_recall_pct','event_f1_pct','short_event_recall_pct','matched_boundary_mae_ms']
    summary=[]
    for split in ['dev','eval']:
        for arm in arms:
            items=[r for r in rows if r['arm']==arm and r['split']==split];assert len(items)==3
            summary.append({'split':split,'arm':arm,'layer':'last','seeds':[1,2,3],**{m:{'mean':statistics.mean(r[m] for r in items),'sd':statistics.stdev(r[m] for r in items)} for m in metrics}})
    (out/'summary.json').write_text(json.dumps({'models':rows,'summary':summary,'scope':'Frozen prediction audit; no retraining, threshold changes or model selection. Dev diagnostics and retrospective PS eval are separate. Event IoU >= .5 on original 10ms references; scores on 20ms grid. Boundary MAE conditional on matched events only.'},indent=2))
    with (out/'seed_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['split','arm','seed','layer','selected_epoch','threshold']+metrics);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
