"""Event matching for duration and linguistic-boundary diagnostics."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def spans(mask,step):
    mask=np.asarray(mask,dtype=bool);edges=np.diff(np.r_[False,mask,False].astype('int8'))
    return np.column_stack([np.flatnonzero(edges==1)*step,np.flatnonzero(edges==-1)*step])


def match_events(reference,predicted,min_iou=.5):
    reference=np.asarray(reference).reshape(-1,2);predicted=np.asarray(predicted).reshape(-1,2)
    if not len(reference) or not len(predicted):return []
    overlap=np.maximum(0,np.minimum(reference[:,1,None],predicted[None,:,1])-np.maximum(reference[:,0,None],predicted[None,:,0]))
    union=np.diff(reference)[:,0,None]+np.diff(predicted)[:,0][None,:]-overlap
    iou=overlap/union;valid=iou>=min_iou-1e-12
    # Maximize number of eligible matches first, total IoU second.
    weight=valid*(min(len(reference),len(predicted))+1+iou)
    ri,pi=linear_sum_assignment(-weight)
    return [(int(r),int(p),float(iou[r,p])) for r,p in zip(ri,pi) if valid[r,p]]


def duration_bin(seconds):
    # Integer microseconds avoid misclassifying decimal grid durations.
    value=int(round(seconds*1e6))
    return ['lt100','100to300','300to600','600to1000','gt1000'][
        0 if value<100000 else 1 if value<300000 else 2 if value<600000 else 3 if value<=1000000 else 4]


def boundary_stratum(event,boundaries,fallback=False,tolerance=.02):
    if fallback:return 'alignment_fallback'
    distance=np.min(np.abs(np.asarray(event)[:,None]-np.asarray(boundaries)[None,:]),axis=1)
    return ['neither_near','one_near','both_near'][int((distance<=tolerance+1e-12).sum())]
