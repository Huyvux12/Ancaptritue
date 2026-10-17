"""Load actual released weights, verify English parity and calibrate N range."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from .model import pretrained,SR
from .data import atomic_json,frame_rms
from .upstream.student import alignment


def bootstrap(root,device='cpu'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    model,original,metadata=pretrained(root,device)
    examples=json.loads(Path(__file__).with_name('calibration_en.json').read_text())
    ns=[];rms=[];reports=[]
    with torch.inference_mode():
        for i,row in enumerate(examples):
            ids=torch.tensor([row['ids']],device=device)
            torch.manual_seed(100+i);a=original.speak_ids(ids)
            torch.manual_seed(100+i);b=model.speak_ids(ids)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if not torch.isfinite(b).all() or float(b.square().mean())<1e-7:raise ValueError('Pretrained English baseline is invalid')
            length=torch.tensor([ids.shape[1]],device=device)
            d=original.text(ids,length)[0][0].round().clamp(1).long()
            _,f0,n,_=original.text(ids,length,alignment(d)[None])
            ns.append(n.flatten().cpu().numpy())
            wave=b[0,0].cpu().numpy()
            rms.append(frame_rms(wave[:len(wave)//300*300]))
            path=root/'english_baseline'/f'{i}.wav';path.parent.mkdir(exist_ok=True)
            sf.write(path,wave,SR)
            reports.append({'text':row['text'],'wav':str(path),'samples':len(wave),'expanded_model_exact_parity':True})
    n=np.concatenate(ns)
    metadata['energy_calibration']={'N_mean':float(n.mean()),'N_std':float(max(n.std(),.1)),
        'method':'Pretrained English predicted N moments. Vietnamese log-RMS is moment matched to this range; approximate surrogate, not Kokoro supervision.'}
    metadata['english_baseline']={'passed':True,'records':reports,'human_listening_verified':False}
    atomic_json(root/'pretrained_metadata.json',metadata)
    torch.save({'model':model.state_dict(),'metadata':metadata,'format':'titl_vi_pretrained_v1'},root/'initial.pt')
    print(json.dumps({'pretrained_loaded':True,'english_parity':True,'budget':metadata['budget'],'N_calibration':metadata['energy_calibration']}),flush=True)
    return model,metadata
