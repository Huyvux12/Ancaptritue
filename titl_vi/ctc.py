"""Pinned Vietnamese CTC aligner; training/preparation only, never exported.

The pinned VLSP checkpoint includes a three-layer Linear/BatchNorm/LeakyReLU
head absent from transformers' stock class. We implement that documented head
locally and reject missing/unexpected learned tensors rather than ignoring it.
Source: nguyenvulebinh/wav2vec2-base-vi-vlsp2020, model_handling.py at CTC_REVISION.
"""
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from numba import njit
import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from .model import CTC_ID, CTC_REVISION


class VietnameseCTC(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        layers = OrderedDict()
        for i in range(1, 4):
            layers[f'linear{i}'] = nn.Linear(config.hidden_size, config.hidden_size)
            layers[f'bn{i}'] = nn.BatchNorm1d(config.hidden_size)
            layers[f'activation{i}'] = nn.LeakyReLU()
            layers[f'drop{i}'] = nn.Dropout(config.final_dropout)
        self.feature_transform = nn.Sequential(layers)

    def forward(self, input_values, attention_mask=None, **kwargs):
        hidden = self.wav2vec2(input_values, attention_mask=attention_mask)[0]
        hidden = self.dropout(hidden)
        b,t,c = hidden.shape
        return SimpleNamespace(logits=self.lm_head(self.feature_transform(hidden.reshape(-1,c)).reshape(b,t,c)))


def load(device='cpu'):
    from huggingface_hub import hf_hub_download
    model, info = VietnameseCTC.from_pretrained(CTC_ID, revision=CTC_REVISION, output_loading_info=True)
    if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
        raise ValueError(f'CTC checkpoint architecture mismatch: {info}')
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(CTC_ID, revision=CTC_REVISION)
    vocab = json.loads(Path(hf_hub_download(CTC_ID, 'vocab.json', revision=CTC_REVISION)).read_text())
    if vocab['<pad>'] != model.config.pad_token_id: raise ValueError('CTC blank mismatch')
    return model.to(device).eval(), extractor, vocab


@njit(cache=True)
def viterbi(logp, labels, blank):
    tmax, length = logp.shape[0], len(labels)
    states = np.full(2*length+1, blank, np.int64)
    states[1::2] = labels
    score = np.full(len(states), -1e30, np.float32)
    score[0] = logp[0,blank]
    if length: score[1] = logp[0,labels[0]]
    back = np.zeros((tmax,len(states)), np.int8)
    for t in range(1,tmax):
        current = np.full(len(states), -1e30, np.float32)
        for s in range(len(states)):
            best, move = score[s], 0
            if s and score[s-1] > best: best,move = score[s-1],1
            if s>1 and states[s]!=blank and states[s]!=states[s-2] and score[s-2]>best: best,move=score[s-2],2
            current[s] = best + logp[t,states[s]]
            back[t,s] = move
        score=current
    s = len(states)-1 if score[-1]>=score[-2] else len(states)-2
    if score[s]<-1e20: raise ValueError('No complete CTC path for the transcript')
    path=np.zeros(tmax,np.int64)
    for t in range(tmax-1,-1,-1):
        path[t]=s
        s-=back[t,s]
    return path


def durations(logp, labels, blank, frames):
    if frames < len(labels)+2: raise ValueError('Insufficient 40-Hz frames for all graphemes and boundaries')
    path=viterbi(logp.astype(np.float32),np.array(labels,np.int64),blank)
    centers=[]; probabilities=[]
    for i,label in enumerate(labels):
        positions=np.flatnonzero(path==2*i+1)
        if not len(positions): raise ValueError('Unaligned grapheme')
        centers.append(float(positions.mean()))
        probabilities.append(float(np.exp(logp[positions,label]).mean()))
    # Reserve one boundary frame at each end; enforce positive char durations.
    boundaries=[1]
    for i in range(1,len(labels)):
        ideal=round(((centers[i-1]+centers[i])/2+0.5)/len(logp)*frames)
        boundaries.append(max(boundaries[-1]+1,min(ideal,frames-1-(len(labels)-i))))
    boundaries.append(frames-1)
    dur=np.array([1]+[boundaries[i+1]-boundaries[i] for i in range(len(labels))]+[1],np.int64)
    assert dur.min()>=1 and dur.sum()==frames
    return dur,float(np.mean(probabilities))
