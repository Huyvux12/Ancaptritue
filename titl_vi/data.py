"""Prepare Vietnamese text/audio for the ORIGINAL 40-Hz TITL decoder clock.

Old Flow/VITS 256-hop features are NOT reused. Only 24-kHz waveform/text/splits
are reused; CTC timings, 80-Hz F0/loudness and durations are recomputed.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import tarfile
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from .model import SR,HOP,canonical,encode,CTC_ID,CTC_REVISION


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def read_rows(path):
    p=Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def write_rows(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(p)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def restore_raw(source,root):
    import maichi_handoff as hh
    source,root=Path(source),Path(root)
    bundle=hh.load_bundle(source,require_complete=True)
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        for entry in bundle['shards']:
            if entry['category']=='features':continue
            local=Path(folder)/hh.safe_path(entry['path']).name
            shutil.copyfile(source/str(hh.safe_path(entry['path'])),local)
            if file_sha(local)!=entry['sha256']:raise ValueError('Corrupt data shard')
            with tarfile.open(local) as archive:
                for member in archive:
                    if not member.isfile():raise ValueError('Only regular data files are accepted')
                    target=root/str(hh.safe_path(member.name))
                    if target.is_symlink() or any(p.is_symlink() for p in target.parents):raise ValueError('Symlink destination')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
            local.unlink()
    rows=read_rows(root/'corpus.jsonl')
    for row in rows:row['wav']=str((root/'wavs'/f"{row['id']}.wav").resolve())
    write_rows(root/'corpus.jsonl',rows)
    atomic_json(root/'raw_bundle.json',bundle)
    return rows


def frame_rms(wave):
    return np.log(np.sqrt(np.mean(wave.reshape(-1,300)**2,axis=1))+1e-5).astype(np.float32)


def pitch(wave):
    import pyworld
    f0,t=pyworld.dio(wave.astype(np.float64),SR,f0_floor=65,f0_ceil=600,frame_period=12.5)
    f0=pyworld.stonemask(wave.astype(np.float64),f0,t,SR)
    centers=(np.arange(len(wave)//300)+.5)*300/SR
    # Nearest sampling preserves unvoiced boundaries instead of interpolating voiced F0 across silence.
    index=np.minimum(np.round(centers/(300/SR)).astype(int),len(f0)-1)
    return f0[index].astype(np.float32)


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a):
        current=[i+1]
        for j,y in enumerate(b):current.append(min(current[-1]+1,previous[j+1]+1,previous[j]+(x!=y)))
        previous=current
    return previous[-1]


def prepare(raw_root,root,metadata,device='cuda',limit=0,min_probability=.10,max_ctc_cer=.35):
    from . import ctc
    raw_root,root=Path(raw_root),Path(root)
    rows=read_rows(raw_root/'corpus.jsonl')
    if not rows:raise ValueError('No text/audio corpus; run T4 data notebook or supply corpus.jsonl')
    seen=set()
    for row in rows:
        identity=row.get('id')
        if not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',identity):
            raise ValueError('Corpus ids must be safe nonempty strings: [A-Za-z0-9_-]')
        if identity in seen:raise ValueError(f'Duplicate corpus id: {identity}')
        seen.add(identity)
        if row.get('split') not in ['train','val','test']:raise ValueError(f'Invalid corpus split: {identity}')
    benchmark=read_rows(raw_root/'benchmark_manifest.jsonl')
    text_key=lambda t: ''.join(c for c in t.lower() if c.isalnum())
    blocked={text_key(r.get(field,'')) for r in benchmark for field in ['text','ref_text','text_normalized']} - {''}
    settings={'format':'titl_vi_aligned_v1','model_revision':metadata['model_revision'],'ctc_id':CTC_ID,'ctc_revision':CTC_REVISION,
              'hop':HOP,'sample_rate':SR,'vocab':metadata['vocab'],'min_probability':min_probability,'max_ctc_cer':max_ctc_cer,
              'source_sha256':{name:file_sha(Path(__file__).with_name(name)) for name in ['data.py','ctc.py','model.py']}}
    fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    marker=root/'preparation_config.json'
    if marker.exists() and json.loads(marker.read_text())!=settings:raise ValueError('Preparation configuration/source changed; use a new prepared root')
    atomic_json(marker,settings)
    model,extractor,vocab=ctc.load(device)
    reverse={v:k for k,v in vocab.items()}
    accepted=[];rejected=[]
    for i,row in enumerate(rows[:limit or None]):
        try:
            text=canonical(row.get('spoken_text',row['text']))
            if text_key(text) in blocked or text_key(row['text']) in blocked:raise ValueError('Benchmark text leaked into corpus')
            ids=encode(text,metadata['vocab']).numpy()
            labels=[vocab['|' if c==' ' else c] for c in text]
            wav=Path(row['wav'])
            if not wav.is_absolute():wav=raw_root/wav
            wave,sr=sf.read(wav,dtype='float32')
            if wave.ndim!=1:raise ValueError('Expected mono audio')
            if not np.isfinite(wave).all():raise ValueError('Nonfinite audio')
            if sr!=SR:wave=resample_poly(wave,SR,sr).astype(np.float32)
            wave=wave[:len(wave)//HOP*HOP]
            if not SR<=len(wave)<=20*SR:raise ValueError('Audio outside 1..20 seconds')
            feature=root/'features'/f"{row['id']}.npz"
            feature.parent.mkdir(parents=True,exist_ok=True)
            audio_hash=file_sha(wav)
            record={**row,'spoken_text':text,'wav':str(wav.resolve()),'feature':str(feature.resolve()),'audio_sha256':audio_hash}
            if feature.exists():
                with np.load(feature) as saved:
                    if str(saved['audio_sha256'])!=audio_hash or str(saved['fingerprint'])!=fingerprint:raise ValueError('Stale prepared audio/features')
                    if not np.array_equal(saved['ids'],ids):raise ValueError('Transcript changed for cached features; use a new prepared root')
                    confidence=float(saved['confidence']);cer=float(saved['ctc_cer'])
            else:
                audio16=resample_poly(wave,2,3).astype(np.float32)
                inputs=extractor(audio16,sampling_rate=16000,return_tensors='pt')
                with torch.inference_mode():logp=model(**{k:v.to(device) for k,v in inputs.items()}).logits[0].float().log_softmax(-1).cpu().numpy()
                greedy=[];last=None
                for v in logp.argmax(-1):
                    v=int(v)
                    if v!=last and v!=vocab['<pad>']:greedy.append(reverse.get(v,''))
                    last=v
                transcript=''.join(greedy).replace('|',' ').replace('<unk>','').strip()
                cer=edit_distance(text,transcript)/max(len(text),1)
                if cer>max_ctc_cer:raise ValueError(f'CTC transcript mismatch CER={cer:.3f}: {transcript}')
                dur,confidence=ctc.durations(logp,labels,vocab['<pad>'],len(wave)//HOP)
                if confidence<min_probability:raise ValueError(f'Low CTC alignment probability={confidence:.3f}')
                f0=pitch(wave);rms=frame_rms(wave)
                tmp=feature.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,ids=ids,duration=dur,f0=f0,log_rms=rms,confidence=confidence,ctc_cer=cer,
                                    fingerprint=fingerprint,audio_sha256=audio_hash)
                tmp.replace(feature)
            accepted.append({**record,'alignment_probability':confidence,'ctc_cer':cer})
        except (ValueError,KeyError,RuntimeError) as error:
            rejected.append({'id':row['id'],'text':row['text'],'reason':str(error)})
        if (i+1)%25==0:print(f'Prepared {i+1}: accepted={len(accepted)}, rejected={len(rejected)}',flush=True)
    write_rows(root/'manifest.jsonl',accepted);write_rows(root/'rejected.jsonl',rejected)
    train=[r for r in accepted if r['split']=='train']
    if not train:raise ValueError('No accepted training items')
    values=[np.load(r['feature'])['log_rms'] for r in train]
    energy=np.concatenate(values)
    # N is a learned Kokoro prosody signal, not literally log RMS. This is an
    # explicit moment-matched surrogate, calibrated to the pretrained range.
    atomic_json(root/'energy_statistics.json',{'log_rms_mean':float(energy.mean()),'log_rms_std':float(max(energy.std(),.1)),
                  'pretrained_N_mean':metadata['energy_calibration']['N_mean'],'pretrained_N_std':metadata['energy_calibration']['N_std'],
                  'scope':'Train-only loudness statistics; approximate N supervision, not teacher internal features'})
    atomic_json(root/'preparation_summary.json',{'accepted':len(accepted),'rejected':len(rejected),'requested':min(len(rows),limit or len(rows)),
                 'splits':{s:sum(r['split']==s for r in accepted) for s in ['train','val','test']},'quality_verified':False})
    return accepted
