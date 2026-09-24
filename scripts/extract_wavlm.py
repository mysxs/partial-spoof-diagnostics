#!/usr/bin/env python3
"""Frozen generic WavLM features, with explicit convolution time geometry."""
import argparse
import hashlib
import importlib
import json
import sys
import time
import types
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ids',required=True)
    ap.add_argument('--wav-dir',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--model',default=None)
    ap.add_argument('--source',default=None)
    a=ap.parse_args()
    if not a.model or not a.source:
        ap.error("--model and --source are required; provide local WavLM paths")
    import numpy as np
    import torch
    import torch.nn.functional as F
    import soundfile as sf
    torch.set_num_threads(8)
    torch.manual_seed(20260919)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    # Import only the published WavLM implementation, not s3prl's other models.
    pkg=types.ModuleType('icassp_wavlm');pkg.__path__=[a.source]
    sys.modules['icassp_wavlm']=pkg
    module=importlib.import_module('icassp_wavlm.WavLM')
    ckpt=torch.load(a.model,map_location='cpu',weights_only=False)
    cfg=module.WavLMConfig(ckpt['cfg'])
    model=module.WavLM(cfg);model.load_state_dict(ckpt['model'],strict=True)
    model.eval().to('cuda')
    del ckpt
    # Geometry from the actual convolution stack (sample rate verified below).
    stride,rf=1,1
    for layer in model.feature_extractor.conv_layers:
        kernel,hop=layer[0].kernel_size[0],layer[0].stride[0]
        rf+=(kernel-1)*stride;stride*=hop
    # Upstream WavLM reshapes the sample mask into equally sized groups,
    # making the valid feature count depend on the longest batch member.
    # Use the actual convolution output lengths instead.
    def exact_padding_mask(self,features,padding_mask):
        lengths=(~padding_mask).sum(-1)
        for layer in self.feature_extractor.conv_layers:
            conv=layer[0]
            lengths=torch.div(lengths-conv.kernel_size[0],conv.stride[0],rounding_mode='floor')+1
        return torch.arange(features.shape[1],device=features.device)[None,:]>=lengths[:,None]
    model.forward_padding_mask=types.MethodType(exact_padding_mask,model)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ids=Path(a.ids).read_text().splitlines()
    if a.limit:ids=ids[:a.limit]
    fingerprint={'model':a.model,'model_size':Path(a.model).stat().st_size,
       'model_mtime_ns':Path(a.model).stat().st_mtime_ns,
       'source_sha256':{f:hashlib.sha256((Path(a.source)/f).read_bytes()).hexdigest() for f in ['WavLM.py','modules.py']},
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'normalize':cfg.normalize,'stride_samples':stride,'receptive_field_samples':rf,
       'first_center_seconds':rf/2/16000,'layer':'last','dtype':'float16_storage_float32_compute',
       'padding_policy':'exact_convolution_lengths','tf32':False,'batch_size':a.batch,
       'ids_sha256':hashlib.sha256(Path(a.ids).read_bytes()).hexdigest()}
    meta=out/'metadata.json'
    if meta.exists():assert json.loads(meta.read_text())==fingerprint,'Cache provenance mismatch'
    else:meta.write_text(json.dumps(fingerprint,indent=2))
    pending=[uid for uid in ids if not (out/(uid+'.npy')).exists()]
    print(f'{len(ids)-len(pending)} cached, {len(pending)} pending; stride={stride}, RF={rf}',flush=True)
    t0=time.time();audio_seconds=0.
    for offset in range(0,len(pending),a.batch):
        batch=pending[offset:offset+a.batch];waves=[];lens=[]
        for uid in batch:
            x,sr=sf.read(Path(a.wav_dir)/(uid+'.wav'),dtype='float32')
            assert sr==16000 and x.ndim==1 and len(x)>=rf,(uid,sr,x.shape)
            w=torch.from_numpy(x).to('cuda')
            if cfg.normalize:w=F.layer_norm(w,w.shape)
            waves.append(w);lens.append(len(w));audio_seconds+=len(w)/sr
        padded=torch.nn.utils.rnn.pad_sequence(waves,batch_first=True)
        lengths=torch.tensor(lens,device='cuda')
        mask=torch.arange(padded.shape[1],device='cuda')[None,:]>=lengths[:,None]
        with torch.inference_mode():
            features,_=model.extract_features(padded,padding_mask=mask,mask=False)
        for i,uid in enumerate(batch):
            n=(lens[i]-rf)//stride+1
            assert n>0 and n<=features.shape[1]
            arr=features[i,:n].float().cpu().numpy().astype('float16')
            assert np.isfinite(arr).all(),uid
            temp=out/(uid+'.tmp')
            with temp.open('wb') as f:np.save(f,arr,allow_pickle=False)
            temp.replace(out/(uid+'.npy'))
        if offset%80==0 or offset+len(batch)==len(pending):
            elapsed=time.time()-t0
            print(f'{offset+len(batch)}/{len(pending)} elapsed={elapsed:.1f}s RTF={elapsed/max(audio_seconds,1):.4f}',flush=True)
    (out/'complete.json').write_text(json.dumps({'requested':len(ids),'elapsed_s':time.time()-t0,
        'audio_seconds_processed':audio_seconds,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}))

if __name__=='__main__':main()
