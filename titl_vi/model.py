"""The original pretrained TITL 8.07M backbone, with appended Vietnamese tokens.

No inference-time posterior, CTC model, external vocoder or feature adapter.
Vietnamese adaptation uses graphemes (NFC, tones retained), not English Misaki.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import unicodedata

os.environ['ORT_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
import torch
from torch import nn

from .upstream.student import TextStudent, alignment
from .upstream.student_decoder import StudentDecoder
from .upstream.exportable import ConvSTFT, Speaker

MODEL_ID = 'MithrilMan/teacher-in-the-loop'
MODEL_REVISION = '115a6c0469632a84fbcc6c8ed0ad7b1e06154b92'
CODE_REVISION = '8d457dd80a73619fbf2e75074c5b707cbb604fbd'
CTC_ID = 'nguyenvulebinh/wav2vec2-base-vi-vlsp2020'
CTC_REVISION = '50a30dadb3ec98a0d4cdb1eb1ea315aff538f7c2'
SR, HOP = 24000, 600
LIMIT = 10_000_000


def canonical(text):
    text = unicodedata.normalize('NFC', text).lower()
    # Pinned Vietnamese text normalization is part of the frontend, not a neural teacher.
    # A corpus row may supply reviewed spoken_text to override the automatic expansion.
    if any(c.isdigit() for c in text):
        from zerotts import normalize_vi_text
        text=unicodedata.normalize('NFC',normalize_vi_text(text)).lower()
        if any(c.isdigit() for c in text):raise ValueError('Unresolved number; provide reviewed spoken_text')
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE).replace('_', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def alphabet():
    chars = set(' abcdefghijklmnopqrstuvwxyzđ')
    for vowel in 'aăâeêioôơuưy':
        for tone in ['', '\u0300', '\u0301', '\u0309', '\u0303', '\u0323']:
            chars.add(unicodedata.normalize('NFC', vowel + tone))
    return sorted(chars)


def source_phone(char):
    # Warm initialization only, not a Vietnamese pronunciation conversion.
    root = ''.join(c for c in unicodedata.normalize('NFD', char) if c not in '\u0300\u0301\u0309\u0303\u0323')
    root = unicodedata.normalize('NFC', root)
    return {'ă':'ɐ', 'â':'ɤ', 'ê':'e', 'ô':'o', 'ơ':'ɤ', 'ư':'ɯ', 'đ':'d', 'g':'ɡ'}.get(root, root)


def make_vocab(base):
    extended = dict(base)
    index = max(base.values()) + 1
    for char in alphabet():
        extended['vi:' + char] = index
        index += 1
    return extended


def encode(text, vocab):
    text = canonical(text)
    if not text: raise ValueError('Empty spoken text')
    absent = sorted(set(text) - {k[3:] for k in vocab if k.startswith('vi:')})
    if absent: raise ValueError(f'Unsupported Vietnamese characters: {absent}')
    if len(text) > 400: raise ValueError('Segment spoken text into <=400 NFC characters')
    return torch.tensor([0] + [vocab['vi:' + c] for c in text] + [0], dtype=torch.long)


def budget(model):
    parts = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ['text', 'decoder']}
    total = sum(parts.values())
    if total >= LIMIT: raise ValueError(f'Complete student {total:,} exceeds strict <10M budget')
    return {**parts, 'total': total, 'limit_exclusive': LIMIT, 'includes': 'All text, decoder and harmonic-source parameters; no external neural vocoder'}


class Student(nn.Module):
    def __init__(self, n_token=178):
        super().__init__()
        self.n_token = n_token
        self.text = TextStudent('s+mlp', n_token=n_token)
        self.decoder = StudentDecoder('A')
        self.decoder.generator.stft = ConvSTFT(20, 5)
        budget(self)

    def forced(self, ids, lengths, aln, f0=None, energy=None, predicted_ratio=0.0):
        duration, predicted_f0, predicted_energy, asr = self.text(ids, lengths, aln)
        f0_in = predicted_f0 if f0 is None else f0 * (1-predicted_ratio) + predicted_f0 * predicted_ratio
        n_in = predicted_energy if energy is None else energy * (1-predicted_ratio) + predicted_energy * predicted_ratio
        return duration, predicted_f0, predicted_energy, asr @ aln, f0_in, n_in

    @torch.no_grad()
    def speak_ids(self, ids, speed=1.0, max_frames=1600):
        if ids.ndim != 2 or len(ids) != 1 or not 0.5 <= speed <= 2:
            raise ValueError('Single-utterance inference, speed 0.5..2.0')
        lengths = torch.tensor([ids.shape[1]], device=ids.device)
        duration = self.text(ids, lengths)[0][0]
        counts = torch.round(duration / speed).clamp(1, 50).long()
        if int(counts.sum()) > max_frames: raise ValueError('Predicted duration exceeds ceiling; segment or inspect convergence')
        aln = alignment(counts)[None]
        _, f0, energy, asr = self.text(ids, lengths, aln)
        return self.decoder(asr @ aln, f0, energy).float()


def load_decoder(decoder, state):
    # ConvSTFT adds only fixed DFT/window buffers; every learned tensor is loaded strictly.
    buffers = {k: v for k, v in decoder.state_dict().items() if k.startswith('generator.stft.')}
    if any(k.startswith('generator.stft.') for k in state):
        raise ValueError('Unexpected source STFT tensors; review pinned upstream checkpoint')
    decoder.load_state_dict({**state, **buffers}, strict=True)


def pretrained(root, device='cpu'):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    get = lambda name: hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION)
    base = json.loads(Path(get('vocab.json')).read_text())
    ts = load_file(get('af_heart/text.safetensors'))
    ds = load_file(get('af_heart/decoder.safetensors'))
    original = Student(178)
    original.text.load_state_dict(ts, strict=True)
    load_decoder(original.decoder, ds)
    vocab = make_vocab(base)
    model = Student(max(vocab.values()) + 1)
    expanded = model.text.state_dict()
    embedding_keys = ['bert.embeddings.word_embeddings.weight', 'text_encoder.embedding.weight']
    # ALBERT CustomAlbert may expose its embedding module under a stable model prefix.
    actual_embedding_keys = [k for k in ts if ts[k].shape != expanded[k].shape]
    if len(actual_embedding_keys) != 2 or any('embedding' not in k for k in actual_embedding_keys):
        raise ValueError(f'Unexpected checkpoint tensor changes: {actual_embedding_keys}')
    for key, value in ts.items():
        if key not in actual_embedding_keys:
            expanded[key] = value
        else:
            expanded[key][:178] = value
            for char in alphabet():
                origin = base.get(source_phone(char), base.get(char, base['ə']))
                expanded[key][vocab['vi:' + char]] = value[origin]
    model.text.load_state_dict(expanded, strict=True)
    load_decoder(model.decoder, ds)
    metadata = {'source_model': MODEL_ID, 'model_revision': MODEL_REVISION, 'code_revision': CODE_REVISION,
                'base_vocabulary': base, 'vocab': vocab, 'budget': budget(model),
                'loaded_from_pretrained': True, 'expanded_tensors': actual_embedding_keys,
                'frontend': 'Vietnamese NFC graphemes with separate ids for all six tones; warm token initialization is not G2P'}
    (root/'pretrained_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return model.to(device).eval(), original.to(device).eval(), metadata
