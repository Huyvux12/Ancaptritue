"""Fixed 137-item benchmark inference; no teacher weights are loaded here."""
from pathlib import Path
import json,time
import numpy as np
import soundfile as sf
import torch
from .model import SR,budget
from .data import read_rows,write_rows,atomic_json
from .train import load_checkpoint
from .gate import speak_text


def benchmark(root,raw_root,checkpoint,device='cuda'):
    root,raw_root=Path(root),Path(raw_root)
    rows=read_rows(raw_root/'benchmark_manifest.jsonl')
    if len(rows)!=137:raise ValueError('Expected the full 137-item benchmark')
    model,ck=load_checkpoint(checkpoint,device);model.eval()
    report=ck.get('intelligibility_gate')
    if not report or not report.get('passed'):raise ValueError('Benchmark requires a gate-passed exported student')
    output=root/'benchmark_wavs/titl_vi_8m';output.mkdir(parents=True,exist_ok=True)
    speak_text(model,ck['metadata']['vocab'],'Xin chào.',device)
    timing=[];failures=[]
    for i,row in enumerate(rows):
        try:
            torch.manual_seed(10000+i)
            if device=='cuda':torch.cuda.synchronize()
            started=time.perf_counter()
            audio=speak_text(model,ck['metadata']['vocab'],row['text'],device)
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            path=output/row['output_wav'];path.parent.mkdir(parents=True,exist_ok=True)
            sf.write(path,audio,SR)
            timing.append({'id':row['id'],'subset':row['subset'],'seconds':elapsed,'duration':len(audio)/SR,
                           'rtf':elapsed/(len(audio)/SR),'device':device})
        except (RuntimeError,ValueError) as error:failures.append({'id':row['id'],'error':str(error)})
    write_rows(output/'timings.jsonl',timing);write_rows(output/'failures.jsonl',failures)
    atomic_json(output/'timing_summary.json',{'pooled_rtf':sum(r['seconds'] for r in timing)/max(sum(r['duration'] for r in timing),1e-8),
                'generated':len(timing),'failed':len(failures),'protocol':'Single-request full waveform; native 24 kHz; text frontend includes pinned Vietnamese normalization'})
    atomic_json(output/'hardware.json',{'gpu':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU','device':device})
    if failures:raise RuntimeError('Incomplete benchmark; inspect failures.jsonl')
    return output
