"""Warm-start supervised audio adaptation of TITL, not Kokoro head distillation.

Alignment/ground-truth prosody stabilize training; inference uses text only.
Waveform gradients update the text features and (joint/GAN) original decoder.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F

from .model import Student,budget,SR,HOP
from .data import atomic_json,read_rows,file_sha
from .upstream.student import alignment


def atomic_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)


def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False)
    model=Student(max(ck['metadata']['vocab'].values())+1).to(device)
    model.load_state_dict(ck['model'],strict=True)
    budget(model)
    return model,ck


class Dataset(torch.utils.data.Dataset):
    def __init__(self,root,split='train',overfit=0):
        self.root=Path(root)
        self.rows=[r for r in read_rows(self.root/'manifest.jsonl') if r['split']==split]
        if overfit:self.rows=sorted(self.rows,key=lambda r:r['id'])[:overfit]
        self.stats=json.loads((self.root/'energy_statistics.json').read_text())
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with np.load(self.root/'features'/f"{r['id']}.npz") as z:
            ids=torch.from_numpy(z['ids'].copy());dur=torch.from_numpy(z['duration'].copy())
            f0=torch.from_numpy(z['f0'].copy());rms=torch.from_numpy(z['log_rms'].copy())
        n=(rms-self.stats['log_rms_mean'])/self.stats['log_rms_std']*self.stats['pretrained_N_std']+self.stats['pretrained_N_mean']
        wave,sr=sf.read(r['wav'],dtype='float32')
        if sr!=SR:wave=resample_poly(wave,SR,sr).astype(np.float32)
        wave=torch.from_numpy(wave[:int(dur.sum())*HOP])
        if len(wave)!=int(dur.sum())*HOP:raise ValueError('Waveform/40-Hz duration mismatch')
        return ids,dur,f0,n,wave


def collate(rows):
    b=len(rows);tokens=max(len(r[0]) for r in rows);frames=max(int(r[1].sum()) for r in rows)
    ids=torch.zeros(b,tokens,dtype=torch.long);lengths=torch.zeros(b,dtype=torch.long)
    dur=torch.zeros(b,tokens);aln=torch.zeros(b,tokens,frames)
    f0=torch.zeros(b,2*frames);energy=torch.zeros_like(f0);wave=torch.zeros(b,frames*HOP)
    token_mask=torch.zeros(b,tokens,dtype=torch.bool);fm=torch.zeros(b,2*frames,dtype=torch.bool);fl=torch.zeros(b,dtype=torch.long)
    for i,(x,d,f,n,y) in enumerate(rows):
        t=len(x);l=int(d.sum());ids[i,:t]=x;lengths[i]=t;dur[i,:t]=d
        aln[i,:t,:l]=alignment(d);f0[i,:2*l]=f;energy[i,:2*l]=n;wave[i,:len(y)]=y
        token_mask[i,:t]=True;fm[i,:2*l]=True;fl[i]=l
    return ids,lengths,dur,aln,f0,energy,wave,token_mask,fm,fl


_MELS={}

def spectral(pred,target):
    import librosa
    key=str(pred.device)
    if key not in _MELS:
        _MELS[key]=torch.tensor(librosa.filters.mel(sr=SR,n_fft=1024,n_mels=80),device=pred.device)
    def mag(y,n):
        return torch.stft(y.float(),n,n//4,window=torch.hann_window(n,device=y.device),return_complex=True).abs().clamp_min(1e-5)
    a,b=mag(pred,1024),mag(target,1024)
    mel=F.l1_loss((_MELS[key]@a.square()).clamp_min(1e-5).log(),(_MELS[key]@b.square()).clamp_min(1e-5).log())
    mr=pred.sum()*0
    for n in [512,1024,2048]:
        a,b=mag(pred,n),mag(target,n)
        mr=mr+torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-5)+F.l1_loss(a.log(),b.log())
    return mel,mr/3


def crop_batch(x,starts,segment):
    values=[]
    for i,start in enumerate(starts):
        part=x[i,...,int(start):int(start)+segment]
        values.append(F.pad(part,(0,segment-part.shape[-1])))
    return torch.stack(values)


def objective(model,batch,segment=64,predicted_ratio=0.,anchor=None):
    ids,lengths,dur,aln,f0,energy,wave,tm,fm,fl=batch
    pd,pf,pn,asr,fi,ni=model.forced(ids,lengths,aln,f0,energy,predicted_ratio)
    ld=((torch.log1p(pd)-torch.log1p(dur)).abs()*tm).sum()/tm.sum()
    voiced=(f0>10)&fm;unvoiced=(~voiced)&fm
    lf=((pf-f0).abs()*voiced).sum()/voiced.sum().clamp_min(1)/100
    lf=lf+.2*((pf-f0).abs()*unvoiced).sum()/unvoiced.sum().clamp_min(1)/100
    ln=((pn-energy).abs()*fm).sum()/fm.sum()/5.5
    fi=fi*fm;ni=ni*fm
    starts=(torch.rand(len(ids),device=ids.device)*(fl-segment+1).clamp_min(1)).long()
    a=crop_batch(asr,starts,segment)
    f=crop_batch(fi,starts*2,segment*2);n=crop_batch(ni,starts*2,segment*2)
    target=crop_batch(wave,starts*HOP,segment*HOP)
    predicted=model.decoder(a,f,n).squeeze(1)
    if predicted.shape!=target.shape:raise ValueError(f'Decoder clock mismatch: {predicted.shape} vs {target.shape}')
    mel,mr=spectral(predicted,target)
    reg=predicted.sum()*0
    if anchor is not None:
        with torch.no_grad():reference=anchor.text(ids,lengths)[3]
        now=model.text(ids,lengths)[3]
        reg=((now-reference).square()*tm[:,None]).sum()/tm.sum()/512
    loss=5*mel+2*mr+ld+lf+ln+.01*reg
    metrics={'mel':mel,'stft':mr,'duration':ld,'f0':lf,'energy':ln,'feature_anchor':reg}
    return loss,metrics,predicted,target


def source_hashes():
    return {str(p.relative_to(Path(__file__).parent)):file_sha(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}


def data_fingerprint(root):
    root=Path(root)
    rows=[{**{k:r[k] for k in ['id','spoken_text','split','audio_sha256']},'prepared_sha256':file_sha(root/'features'/f"{r['id']}.npz")} for r in read_rows(root/'manifest.jsonl')]
    identity={'rows':rows,'energy':json.loads((root/'energy_statistics.json').read_text())}
    return hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


@torch.no_grad()
def evaluate(model,dataset,device,rows=8):
    model.eval();values=[]
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
        torch.manual_seed(1234)
        for i in range(min(rows,len(dataset))):
            batch=[v.to(device) for v in collate([dataset[i]])]
            loss,metrics,_,_=objective(model,batch,predicted_ratio=1.)
            values.append(float(loss))
    return float(np.mean(values)) if values else None


def train(root,run,stage='bridge',steps=2000,batch_size=8,overfit=0,init=None,device='cuda',workers=2,
          save_every=250,eval_every=500,backup_dir=None,backup_seconds=600,engineering_smoke=False):
    root=Path(root);folder=root/'runs'/run;folder.mkdir(parents=True,exist_ok=True)
    if stage not in ['bridge','joint','gan']:raise ValueError(stage)
    if engineering_smoke and (device!='cpu' or steps>10):raise ValueError('Engineering smoke is limited to <=10 CPU steps; it cannot bypass quality gates for GPU training')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(4)
    training=Dataset(root,'train',overfit);validation=Dataset(root,'val')
    if not len(training):raise ValueError('Empty training set')
    path=folder/'last.pt'
    origin=path if path.exists() else Path(init or root/'initial.pt')
    quality_parent=None
    if not overfit and not engineering_smoke:
        if len(training)<1000 or len(validation)<20:raise ValueError('Full adaptation requires >=1000 training and >=20 validation items')
        if not path.exists():
            required={'bridge':'overfit','joint':'pre_joint','gan':'pre_gan'}[stage]
            quality_parent=json.loads((root/'gates'/f'{required}.json').read_text())
            if not quality_parent.get('passed') or quality_parent['data_fingerprint']!=data_fingerprint(root):raise ValueError(f'{required} intelligibility gate has not passed')
            if quality_parent['checkpoint_sha256']!=file_sha(origin):raise ValueError('Quality gate belongs to a different initialization checkpoint')
    model,ck=load_checkpoint(origin,device)
    metadata=ck['metadata']
    model.requires_grad_(True)
    if stage=='bridge':model.decoder.requires_grad_(False)
    model.decoder.generator.m_source.requires_grad_(False)
    groups=[{'params':[p for p in model.text.parameters() if p.requires_grad],'lr':1e-4,'peak':1e-4}]
    if stage!='bridge':groups.append({'params':[p for p in model.decoder.parameters() if p.requires_grad],'lr':2e-5,'peak':2e-5})
    opt=torch.optim.AdamW(groups,betas=(.9,.99),weight_decay=.001)
    anchor,_=load_checkpoint(root/'initial.pt',device);anchor.eval().requires_grad_(False)
    critic=critic_opt=None
    if stage=='gan':
        from .upstream.discriminators import Discriminators
        critic=Discriminators().to(device)
        critic_opt=torch.optim.AdamW(critic.parameters(),lr=1e-4,betas=(.8,.99))
    config={'stage':stage,'batch_size':batch_size,'overfit':overfit,'data_fingerprint':data_fingerprint(root),
            'source_hashes':source_hashes(),'vocab':metadata['vocab'],'precision':'fp32','engineering_smoke':engineering_smoke,
            'training_ids':[r['id'] for r in training.rows],
            'quality_parent':ck.get('config',{}).get('quality_parent') if path.exists() else quality_parent,
            'init_sha256':ck.get('config',{}).get('init_sha256') if path.exists() else file_sha(origin)}
    step=0;best=float('inf')
    generator=torch.Generator().manual_seed(2026)
    if path.exists():
        if ck['config']!=config:raise ValueError('Resume source/config/data mismatch; choose a new run')
        opt.load_state_dict(ck['optimizer']);step=ck['step'];best=ck['best']
        torch.set_rng_state(ck['torch_rng'].cpu());random.setstate(ck['random_rng']);np.random.set_state(ck['numpy_rng'])
        generator.set_state(ck['loader_rng'].cpu())
        if device=='cuda' and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
        if critic is not None:critic.load_state_dict(ck['critic']);critic_opt.load_state_dict(ck['critic_optimizer'])
    loader=torch.utils.data.DataLoader(training,batch_size=min(batch_size,len(training)),shuffle=True,collate_fn=collate,
                num_workers=workers,persistent_workers=workers>0,multiprocessing_context='spawn' if workers else None,generator=generator)
    iterator=iter(loader)
    def save(name):
        atomic_save({'format':'titl_vi_train_v1','model':model.state_dict(),'metadata':metadata,'step':step,'best':best,'config':config,
            'optimizer':opt.state_dict(),'critic':critic.state_dict() if critic is not None else None,
            'critic_optimizer':critic_opt.state_dict() if critic_opt is not None else None,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,
            'random_rng':random.getstate(),'numpy_rng':np.random.get_state(),'loader_rng':generator.get_state(),
            'resume_note':'Optimizer and RNG restored; iterator restarts, not exact batch replay'},folder/name)
    last_backup=time.monotonic()
    atomic_json(folder/'config.json',config)
    try:
        while step<steps:
            started=time.perf_counter()
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            batch=[v.to(device) for v in batch]
            model.train()
            if stage=='bridge':model.decoder.eval()
            # Force measured prosody at first; progressively use predicted prosody.
            ratio=0. if stage=='bridge' else min(1.,step/2000)
            loss,metrics,predicted,target=objective(model,batch,predicted_ratio=ratio,anchor=anchor)
            if critic is not None:
                from .upstream.discriminators import d_loss,g_loss
                critic.requires_grad_(True);critic_opt.zero_grad(set_to_none=True)
                dl=d_loss(critic,target,predicted)
                if not torch.isfinite(dl):raise FloatingPointError('Nonfinite discriminator loss')
                dl.backward();nn.utils.clip_grad_norm_(critic.parameters(),5.);critic_opt.step()
                critic.requires_grad_(False)
                adv,fm=g_loss(critic,target,predicted)
                loss=loss+adv+2*fm;metrics.update({'discriminator':dl,'adversarial':adv,'feature_matching':fm})
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
            # Fixed-horizon-free schedule allows extending a run without changing old updates.
            for group in opt.param_groups:group['lr']=group['peak']*min(1.,(step+1)/200)*(.99995**step)
            opt.step();step+=1
            if step==1 or step%25==0:
                record={'step':step,'loss':float(loss.detach()),'gradient':float(grad),'seconds':time.perf_counter()-started,
                        'predicted_prosody_ratio':ratio,**{k:float(v.detach()) for k,v in metrics.items()}}
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(record,flush=True)
            if step%eval_every==0 or step==steps:
                score=evaluate(model,validation if len(validation) else training,device)
                with (folder/'validation.jsonl').open('a') as f:f.write(json.dumps({'step':step,'forced_validation_loss':score})+'\n')
                if score is not None and score<best:best=score;save('best.pt')
            if step%save_every==0:save('last.pt')
            if backup_dir and time.monotonic()-last_backup>=backup_seconds:
                save('last.pt')
                from .storage import backup
                backup(root,backup_dir);last_backup=time.monotonic()
    except KeyboardInterrupt:
        save('last.pt');raise
    finally:
        save('last.pt')
        if backup_dir:
            from .storage import backup
            backup(root,backup_dir)
    print('Training stage complete:',run,step,budget(model),flush=True)
    return folder/'best.pt'
