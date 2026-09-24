#!/usr/bin/env python3
"""Locked-threshold event diagnostics, preserving missed and false events."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from event_strata import spans,match_events,duration_bin,boundary_stratum


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for k in ['root','units','out']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--runs',nargs='+',required=True);a=ap.parse_args();root=Path(a.root);units=Path(a.units)/'dev';out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    labels=np.load(root/'database/segment_labels/dev_seglab_0.01.npy',allow_pickle=True).item()
    labels20=np.load(root/'database/segment_labels/dev_seglab_0.02.npy',allow_pickle=True).item()
    ids=(root/'database/dev/dev.lst').read_text().splitlines();assert len(ids)==24844 and set(ids)==set(labels)
    fallback={r['uttid'] for r in json.loads((units/'utterances.json').read_text()) if r['fallback']}
    metadata={}
    for uid in ids:
        with np.load(units/(uid+'.npz'),allow_pickle=False) as f:
            intervals=f['phone_intervals'];kinds=f['phone_kind'].astype(bool)
        # Only real phone starts/ends count as linguistic boundaries; gap subdivisions do not.
        boundaries=np.unique(intervals[kinds].reshape(-1))
        events=spans(np.asarray(labels[uid]).astype('int8')==0,.01)
        strata=[(duration_bin(e-s),boundary_stratum([s,e],boundaries,uid in fallback)) for s,e in events]
        metadata[uid]=(events,strata)
    runs=[p for folder in a.runs for p in sorted(Path(folder).glob('*/dev_predictions.npz'))]
    assert runs
    report={}
    for source in runs:
        run=source.parent;history=json.loads((run/'history.json').read_text());best=min(history,key=lambda r:r['segment20_eer'])
        complete=json.loads((run/'complete.json').read_text());assert abs(best['segment20_eer']-complete['best_dev_eer'])<1e-12
        threshold=best['dev_selected_spoof_threshold'];predictions=np.load(source,allow_pickle=False);assert set(predictions.files)==set(ids)
        totals=Counter();groups=defaultdict(Counter);fpbins=Counter();details=[]
        for uid in ids:
            score=predictions[uid];assert len(score)==len(labels20[uid]) and np.isfinite(score).all(),uid
            reference,strata=metadata[uid];pred=spans(score>threshold,.02);matches=match_events(reference,pred)
            matched_r={r for r,p,iou in matches};matched_p={p for r,p,iou in matches}
            totals.update(reference_events=len(reference),predicted_events=len(pred),matched_events=len(matches),
                          missed_events=len(reference)-len(matches),false_events=len(pred)-len(matches))
            if not len(reference):totals['false_events_on_genuine_utterances']+=len(pred)
            match_map={r:(p,iou) for r,p,iou in matches};per_groups={}
            for i,(duration,linguistic) in enumerate(strata):
                for group in ['duration/'+duration,'phone_boundary/'+linguistic]:
                    groups[group]['reference_events']+=1;groups[group]['matched_events']+=int(i in matched_r)
                    groups[group]['missed_events']+=int(i not in matched_r)
                    per_groups.setdefault(group,[0,0]);per_groups[group][0]+=1;per_groups[group][1]+=int(i in matched_r)
                    if i in matched_r:
                        p,_=match_map[i];groups[group]['matched_start_abs_error_seconds']+=abs(reference[i,0]-pred[p,0])
                        groups[group]['matched_end_abs_error_seconds']+=abs(reference[i,1]-pred[p,1])
            for p,(s,e) in enumerate(pred):
                if p not in matched_p:fpbins[duration_bin(e-s)]+=1
            details.append({'uttid':uid,'reference_events':len(reference),'predicted_events':len(pred),
                            'matches':matches,'missed_events':len(reference)-len(matches),'false_events':len(pred)-len(matches),
                            'groups_reference_matched':per_groups})
        tp=totals['matched_events'];precision=tp/max(totals['predicted_events'],1);recall=tp/max(totals['reference_events'],1)
        result={'counts':dict(totals),'event_precision':precision,'event_recall':recall,'event_f1':2*tp/max(totals['predicted_events']+totals['reference_events'],1),
                'groups':{k:dict(v)|{'recall':v['matched_events']/v['reference_events']} for k,v in groups.items()},
                'unmatched_prediction_duration_bins':dict(fpbins),'threshold':threshold,'selected_epoch':best['epoch'],
                'prediction_sha256':sha(source),'history_sha256':sha(run/'history.json'),
                'notes':['Unchanged raw 10ms reference events; 20ms prediction runs at saved dev threshold.',
                         'One-to-one IoU >= 0.5 matching maximizes eligible match count, then total IoU.',
                         'Reference duration bins provide recall/misses. False predictions separately binned by predicted duration.',
                         'Conditional matched-boundary errors do not describe missed events; missed counts retained.',
                         'Near linguistic boundary means within 20ms of an actual phone start/end; gap subdivision edges excluded.',
                         'Dev-selected diagnostics, not test performance or full official RangeEER.']}
        dest=out/run.parent.name/run.name;dest.mkdir(parents=True,exist_ok=True)
        (dest/'per_utterance.json').write_text(json.dumps(details));(dest/'complete.json').write_text(json.dumps(result,indent=2))
        report[str(run)]=result;predictions.close();print(str(run),'F1',result['event_f1'],flush=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
