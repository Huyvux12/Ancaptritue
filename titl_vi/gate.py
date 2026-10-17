"""Paired ASR intelligibility gates; never substitute for human listening/MOS.

No full training or production export is permitted solely on spectral losses.
"""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from .model import SR,HOP,encode,budget
from .train import load_checkpoint,Dataset,collate,data_fingerprint,atomic_save
from .data import atomic_json,edit_distance,file_sha,canonical

ASR_ID='vinai/PhoWhisper-small'


def diagnostics(root,checkpoint,name,split='val',count=8,device='cuda',overfit=0):
    root=Path(root);folder=root/'diagnostics'/name;folder.mkdir(parents=True,exist_ok=True)
    dataset=Dataset(root,split,overfit)
    model,ck=load_checkpoint(checkpoint,device);model.eval()
    records=[]
    with torch.inference_mode():
        for i,row in enumerate(dataset.rows[:count]):
            values=dataset[i]
            batch=[x.to(device) for x in collate([values])]
            ids,lengths,dur,aln,f0,n,wave,*_=batch
            _,pf,pn,asr,_,_=model.forced(ids,lengths,aln)
            views={}
            for kind,render in [('aligned_measured_prosody',lambda:model.decoder(asr,f0,n)),
                                ('aligned_predicted_prosody',lambda:model.decoder(asr,pf,pn)),
                                ('free_text',lambda:model.speak_ids(ids))]:
                # Fixed harmonic-noise seed per utterance, shared across views.
                torch.manual_seed(2026+i)
                views[kind]=render()
            item={'id':row['id'],'text':row['spoken_text'],'source_wav':row['wav'],'audio':{}}
            for kind,audio in views.items():
                audio=audio[0,0].float().cpu().numpy()
                if not np.isfinite(audio).all() or not .3<=len(audio)/SR<=40:raise ValueError('Invalid waveform/duration')
                path=folder/f"{i}_{kind}.wav";sf.write(path,audio,SR)
                item['audio'][kind]=str(path.resolve())
                if kind=='free_text':item['free_duration']=len(audio)/SR;item['free_peak']=float(np.max(np.abs(audio)));item['free_rms']=float(np.sqrt(np.mean(audio**2)))
            records.append(item)
    atomic_json(folder/'samples.json',{'checkpoint_sha256':file_sha(checkpoint),'noise_seed_base':2026,'records':records,'quality_verified':False})
    return records


def gate(root,checkpoint,name,split='val',count=8,device='cuda',overfit=0,max_cer=.20,max_delta=.10):
    from huggingface_hub import model_info
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    root=Path(root)
    if name=='overfit' and (split!='train' or not overfit):raise ValueError('Overfit gate must use the explicitly selected training subset')
    if name in ['pre_joint','pre_gan','final'] and split!='val':raise ValueError('Full-stage gates require validation, never public test')
    rows=diagnostics(root,checkpoint,name,split,count,device,overfit)
    if len(rows)<min(count,2):raise ValueError('Insufficient gate samples')
    revision=model_info(ASR_ID).sha
    processor=WhisperProcessor.from_pretrained(ASR_ID,revision=revision)
    asr=WhisperForConditionalGeneration.from_pretrained(ASR_ID,revision=revision).to(device).eval()
    def transcribe(path):
        wave,sr=sf.read(path,dtype='float32')
        if wave.ndim==2:wave=wave.mean(1)
        wave=resample_poly(wave,16000,sr).astype(np.float32)
        inputs=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():
            tokens=asr.generate(inputs.input_features.to(device),attention_mask=inputs.attention_mask.to(device),
                                language='vi',task='transcribe',max_new_tokens=256,do_sample=False)
        return processor.batch_decode(tokens,skip_special_tokens=True)[0]
    ref_errors=student_errors=total=0
    for row in rows:
        reference=transcribe(row['source_wav']);student=transcribe(row['audio']['free_text'])
        clean=lambda s:''.join(c for c in canonical(s) if c.isalnum())
        target=clean(row['text']);r=clean(reference);s=clean(student)
        re=edit_distance(target,r);se=edit_distance(target,s)
        row.update({'reference_asr':reference,'student_asr':student,'reference_cer':re/max(len(target),1),'student_cer':se/max(len(target),1)})
        ref_errors+=re;student_errors+=se;total+=len(target)
    ref_cer=ref_errors/max(total,1);student_cer=student_errors/max(total,1)
    signal_ok=all(r['free_rms']>.005 and r['free_peak']<2 for r in rows)
    threshold=min(max_cer,ref_cer+max_delta)
    passed=ref_cer<=max_cer and student_cer<=threshold and signal_ok
    report={'passed':passed,'name':name,'checkpoint_sha256':file_sha(checkpoint),'checkpoint':str(Path(checkpoint).resolve()),
            'data_fingerprint':data_fingerprint(root),'asr_model':ASR_ID,'asr_revision':revision,
            'teacher_cer':ref_cer,'student_cer':student_cer,'threshold':threshold,'signal_ok':signal_ok,'noise_seed_base':2026,
            'scope':'Paired automatic intelligibility gate on selected samples; not human MOS, full benchmark, or tone quality proof',
            'records':rows,'quality_verified':False}
    atomic_json(root/'gates'/f'{name}.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
    if not passed:raise RuntimeError(f'{name} gate failed. Inspect diagnostics/{name}; full-stage training/export remains blocked.')
    return report


def export(root,checkpoint):
    root=Path(root)
    report=json.loads((root/'gates/final.json').read_text())
    if not report['passed'] or report['checkpoint_sha256']!=file_sha(checkpoint) or report['data_fingerprint']!=data_fingerprint(root):
        raise ValueError('Final intelligibility gate must pass for this exact checkpoint and corpus')
    model,ck=load_checkpoint(checkpoint,'cpu');model.eval()
    path=root/'export/student_vi_8m.pt'
    atomic_save({'format':'titl_vi_deploy_v1','model':model.state_dict(),'metadata':ck['metadata'],
                'budget':budget(model),'sample_rate':SR,'intelligibility_gate':{k:v for k,v in report.items() if k!='records'},
                'quality_verified':False,'note':'Full Vietnamese quality and tone preservation require listening and held-out benchmark'},path)
    return path


def infer(path,text,output,device='cpu',speed=1.):
    model,ck=load_checkpoint(path,device);model.eval()
    waveform=speak_text(model,ck['metadata']['vocab'],text,device,speed)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    sf.write(output,waveform,SR)
    return waveform


def speak_text(model,vocab,text,device='cpu',speed=1.):
    words=canonical(text).split();chunks=[];current=''
    for word in words:
        candidate=(current+' '+word).strip()
        if len(candidate)>380:
            if not current:raise ValueError('Unsegmentable token exceeds 380 characters')
            chunks.append(current);current=word
        else:current=candidate
    if current:chunks.append(current)
    if not chunks:raise ValueError('Empty text')
    audio=[]
    for chunk in chunks:
        ids=encode(chunk,vocab)[None].to(device)
        result=model.speak_ids(ids,speed=speed)[0,0].cpu().numpy()
        if not np.isfinite(result).all():raise ValueError('Nonfinite synthesis')
        if audio:audio.append(np.zeros(int(.12*SR),np.float32))
        audio.append(result)
    return np.concatenate(audio)
