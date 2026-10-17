"""Budget-constrained Mai Chi TTS: conditional flow matching + compact vocoder.

This is an experimental small DiT/OT-CFM implementation, inspired by F5-TTS
and Matcha-TTS, not their pretrained model or a claimed state-of-the-art model.
The complete inference package (text, duration, flow, pitch head and waveform
generator) must have strictly fewer than 10,000,000 parameters.

MAS and HiFi-GAN-style building blocks follow the MIT VITS implementation;
Copyright (c) 2021 Jaehyeon Kim. See LICENSE_VITS.txt. New code: MIT.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from numba import njit

import maichi_pipeline as data

LIMIT = 10_000_000
ARCHITECTURE = "maichi_dit_cfm_v2"
DEFAULT_CONFIG = {"text_width": 128, "text_layers": 3, "flow_width": 192,
                  "flow_layers": 10, "flow_heads": 4, "vocoder_channels": 256}


def sequence_mask(lengths, size=None):
    size = int(lengths.max()) if size is None else size
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


def positions(length, width, device, dtype):
    p = torch.arange(length, device=device).float()[:, None]
    w = torch.exp(-math.log(10000) * torch.arange(0, width, 2, device=device).float() / width)
    a = p * w
    return torch.stack([a.sin(), a.cos()], -1).reshape(length, width).to(dtype)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x, valid):
        b, t, h = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0)
        return self.out(out.transpose(1, 2).reshape(b, t, h)) * valid[..., None]


class TextBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(width, 4)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, valid):
        x = x + self.attn(self.norm1(x), valid)
        return (x + self.ff(self.norm2(x))) * valid[..., None]


class DiTBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = Attention(width, heads)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, time_embedding, valid):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(time_embedding).unsqueeze(1).chunk(6, -1)
        x = x + gate_a * self.attn(self.norm1(x) * (1 + scale_a) + shift_a, valid)
        x = x + gate_f * self.ff(self.norm2(x) * (1 + scale_f) + shift_f)
        return x * valid[..., None]


class Velocity(nn.Module):
    def __init__(self, width, layers, heads, text_width):
        super().__init__()
        self.width = width
        self.input = nn.Linear(2 * data.MELS + text_width + 2, width)
        self.time = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width))
        self.blocks = nn.ModuleList([DiTBlock(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, data.MELS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, prior, text, within_token, t, valid):
        inp = torch.cat([x, prior, text, within_token], 1).transpose(1, 2)
        h = self.input(inp)
        h = (h + positions(h.shape[1], self.width, h.device, h.dtype)) * valid[..., None]
        frequency = torch.exp(-math.log(10000) * torch.arange(0, self.width, 2, device=x.device).float() / self.width)
        angles = t.float()[:, None] * 1000 * frequency[None]
        temb = self.time(torch.cat([angles.sin(), angles.cos()], -1).to(h.dtype))
        for block in self.blocks:
            h = block(h, temb, valid)
        return self.output(self.norm(h)).transpose(1, 2) * valid[:, None]


class ResidualVocoderBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, kernel, dilation=d,
                                            padding=d * (kernel - 1) // 2) for d in [1, 3]])

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(F.leaky_relu(x, 0.1))
        return x


class CompactVocoder(nn.Module):
    """HiFi-GAN-style generator, trained here on Mai Chi; no hidden pretrained decoder."""
    def __init__(self, channels=256):
        super().__init__()
        self.pre = nn.Conv1d(data.MELS, channels, 7, padding=3)
        self.ups, self.residuals = nn.ModuleList(), nn.ModuleList()
        current = channels
        for rate, kernel in zip([8, 8, 2, 2], [16, 16, 4, 4]):
            target = current // 2
            self.ups.append(nn.ConvTranspose1d(current, target, kernel, rate, padding=(kernel - rate) // 2))
            self.residuals.append(nn.ModuleList([ResidualVocoderBlock(target, k) for k in [3, 7, 11]]))
            current = target
        self.post = nn.Conv1d(current, 1, 7, padding=3)

    def forward(self, mel):
        x = self.pre(mel)
        for up, blocks in zip(self.ups, self.residuals):
            x = up(F.leaky_relu(x, 0.1))
            x = sum(block(x) for block in blocks) / len(blocks)
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))


@njit(cache=True)
def _mas(scores, frames, tokens):
    paths = np.zeros(scores.shape, dtype=np.float32)
    for b in range(scores.shape[0]):
        ty, tx = frames[b], tokens[b]
        for y in range(ty):
            for x in range(max(0, tx + y - ty), min(tx, y + 1)):
                stay = -1e9 if x == y else scores[b, y - 1, x]
                previous = (0.0 if y == 0 else -1e9) if x == 0 else scores[b, y - 1, x - 1]
                scores[b, y, x] += max(previous, stay)
        index = tx - 1
        for y in range(ty - 1, -1, -1):
            paths[b, y, index] = 1
            if index and (index == y or scores[b, y - 1, index] < scores[b, y - 1, index - 1]):
                index -= 1
    return paths


class FlowStudent(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        self.config = dict(DEFAULT_CONFIG if config is None else config)
        c = self.config
        width = c["text_width"]
        self.embedding = nn.Embedding(vocab_size, width)
        self.text_blocks = nn.ModuleList([TextBlock(width) for _ in range(c["text_layers"])])
        self.prior = nn.Conv1d(width, data.MELS, 1)
        self.duration = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(width, width, 3, padding=1), nn.ReLU(), nn.Conv1d(width, 1, 1))
        self.pitch = nn.Sequential(nn.Conv1d(width + 2, 64, 3, padding=1), nn.SiLU(), nn.Conv1d(64, 2, 1))
        self.velocity = Velocity(c["flow_width"], c["flow_layers"], c["flow_heads"], width)
        self.vocoder = CompactVocoder(c["vocoder_channels"])
        self.register_buffer("mel_mean", torch.zeros(1, data.MELS, 1))
        self.register_buffer("mel_std", torch.ones(1, data.MELS, 1))
        assert_budget(self)

    def text(self, ids, lengths):
        valid = sequence_mask(lengths, ids.shape[1])
        h = self.embedding(ids)
        h = (h + positions(h.shape[1], h.shape[2], h.device, h.dtype)) * valid[..., None]
        for block in self.text_blocks:
            h = block(h, valid)
        h = h.transpose(1, 2)
        return h, self.prior(h) * valid[:, None], self.duration(h.detach()) * valid[:, None], valid

    @staticmethod
    def expanded(h, prior, alignment):
        text = torch.matmul(h, alignment.transpose(1, 2))
        mu = torch.matmul(prior, alignment.transpose(1, 2))
        duration = alignment.sum(1).unsqueeze(-1)
        starts = duration.cumsum(1) - duration
        start = torch.matmul(alignment, starts).transpose(1, 2)
        length = torch.matmul(alignment, duration).transpose(1, 2).clamp_min(1)
        frame = torch.arange(alignment.shape[1], device=h.device, dtype=h.dtype)[None, None]
        position = ((frame - start + 0.5) / length).clamp(0, 1)
        within = torch.cat([position, torch.sin(math.pi * position)], 1)
        return text, mu, within

    def objective(self, ids, lengths, raw_mel, frame_lengths, f0, vuv, step=0, pitch_weight=0.2):
        h, prior, log_duration, text_valid = self.text(ids, lengths)
        valid = sequence_mask(frame_lengths, raw_mel.shape[-1])
        mask = valid[:, None]
        target = (raw_mel - self.mel_mean) / self.mel_std
        with torch.no_grad():
            if (frame_lengths < lengths).any():
                raise ValueError("More text tokens than mel frames; shorten/resegment the text.")
            y, mu = target.float(), prior.float()
            score = -0.5 * (y.square().sum(1)[:, :, None] - 2 * y.transpose(1, 2) @ mu + mu.square().sum(1)[:, None, :])
            fy = torch.arange(y.shape[-1], device=y.device)[None, :, None] / frame_lengths[:, None, None]
            tx = torch.arange(mu.shape[-1], device=y.device)[None, None, :] / lengths[:, None, None]
            score = score - 10 * max(0, 1 - step / 10000) * (fy - tx).square()
            path = _mas(score.cpu().numpy().copy(), frame_lengths.cpu().numpy().astype(np.int32), lengths.cpu().numpy().astype(np.int32))
            alignment = torch.from_numpy(path).to(device=h.device, dtype=h.dtype)
        text, aligned_prior, within = self.expanded(h, prior, alignment)
        duration = alignment.sum(1)[:, None]
        loss_duration = ((log_duration.float() - torch.log(duration + 1e-6)).square() * text_valid[:, None]).sum() / text_valid.sum()
        loss_prior = ((target.float() - aligned_prior.float()).square() * mask).sum() / (mask.sum() * data.MELS)
        t = torch.rand(raw_mel.shape[0], device=raw_mel.device)
        noise = torch.randn_like(target)
        sigma = 1e-4
        xt = ((1 - (1 - sigma) * t[:, None, None]) * noise + t[:, None, None] * target) * mask
        velocity = self.velocity(xt, aligned_prior, text, within, t, valid)
        desired = target - (1 - sigma) * noise
        loss_flow = ((velocity.float() - desired.float()).square() * mask).sum() / (mask.sum() * data.MELS)
        pred = self.pitch(torch.cat([text, within], 1))
        voiced = vuv * mask
        loss_pitch = ((pred[:, :1].float() - f0).abs() * voiced).sum() / voiced.sum().clamp_min(1)
        loss_pitch = loss_pitch + (F.binary_cross_entropy_with_logits(pred[:, 1:].float(), vuv, reduction="none") * mask).sum() / mask.sum()
        loss = loss_flow + loss_prior + loss_duration + pitch_weight * loss_pitch
        return loss, {"flow": loss_flow, "prior": loss_prior, "duration": loss_duration, "pitch": loss_pitch}, alignment

    @torch.no_grad()
    def mel(self, ids, lengths, steps=16, temperature=0.667, max_frames=2000, durations=None):
        if steps < 1:
            raise ValueError("At least one flow step is required")
        h, prior, log_duration, valid_text = self.text(ids, lengths)
        if durations is None:
            duration = torch.ceil(log_duration[:, 0].clamp(-3, 4).exp()) * valid_text
        else:
            duration = durations * valid_text
        frame_lengths = duration.sum(-1).clamp_min(1).long()
        if int(frame_lengths.max()) > max_frames:
            raise RuntimeError("Predicted audio exceeds frame ceiling; model is unconverged or input needs segmentation")
        valid = sequence_mask(frame_lengths)
        cumulative = duration.cumsum(-1)
        frame = torch.arange(valid.shape[-1], device=ids.device)[None, :, None]
        alignment = ((frame >= (cumulative - duration)[:, None]) & (frame < cumulative[:, None])).to(h.dtype)
        text, mu, within = self.expanded(h, prior, alignment)
        z = torch.randn(ids.shape[0], data.MELS, valid.shape[-1], device=ids.device) * temperature * valid[:, None]
        for index in range(steps):
            t = torch.full((ids.shape[0],), index / steps, device=ids.device)
            z = (z + self.velocity(z, mu, text, within, t, valid) / steps) * valid[:, None]
        return z * self.mel_std + self.mel_mean, frame_lengths

    @torch.no_grad()
    def speak(self, ids, lengths, steps=16, temperature=0.667, max_frames=2000):
        if ids.shape[0] != 1:
            raise ValueError("Waveform synthesis currently requires one utterance per request")
        mel, _ = self.mel(ids, lengths, steps, temperature, max_frames)
        return self.vocoder(mel)


def counts(model):
    total = sum(p.numel() for p in model.parameters())
    vocoder = sum(p.numel() for p in model.vocoder.parameters())
    return {"architecture": ARCHITECTURE, "deployment_parameters": total,
            "acoustic_parameters": total - vocoder, "vocoder_parameters": vocoder,
            "buffer_elements": sum(b.numel() for b in model.buffers()), "limit_exclusive": LIMIT,
            "under_10m": total < LIMIT, "note": "All student inference modules included; no external neural codec/vocoder. GAN discriminator and EMA copies are training-only."}


def assert_budget(model):
    info = counts(model)
    if not info["under_10m"]:
        raise ValueError(f"Student exceeds strict budget: {info['deployment_parameters']:,} >= {LIMIT:,}")
    return info


class PeriodDiscriminator(nn.Module):
    """Training-only periodic waveform discriminator; never exported."""
    def __init__(self, period):
        super().__init__()
        self.period = period
        widths = [1, 32, 128, 256, 512]
        self.layers = nn.ModuleList([nn.utils.spectral_norm(nn.Conv2d(a, b, (5, 1), (3, 1), padding=(2, 0)))
                                     for a, b in zip(widths[:-1], widths[1:])])
        self.final = nn.utils.spectral_norm(nn.Conv2d(512, 1, (3, 1), padding=(1, 0)))

    def forward(self, waveform):
        b, c, t = waveform.shape
        remainder = t % self.period
        if remainder:
            waveform = F.pad(waveform, (0, self.period - remainder), mode="reflect")
        x = waveform.reshape(b, c, -1, self.period)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        x = self.final(x)
        return x.flatten(1), features + [x]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = nn.ModuleList([PeriodDiscriminator(p) for p in [2, 3, 5, 7, 11]])

    def forward(self, waveform):
        return [disc(waveform) for disc in self.periods]


def multi_resolution_stft(generated, target):
    loss = generated.sum() * 0
    for nfft, hop in [(512, 128), (1024, 256), (2048, 512)]:
        window = torch.hann_window(nfft, device=generated.device)
        a = torch.stft(generated[:, 0].float(), nfft, hop, nfft, window, return_complex=True).abs().clamp_min(1e-5)
        b = torch.stft(target[:, 0].float(), nfft, hop, nfft, window, return_complex=True).abs().clamp_min(1e-5)
        loss = loss + torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-5)
        loss = loss + F.l1_loss(a.log(), b.log())
    return loss / 3


def crop(x, starts, length):
    return torch.stack([x[i, :, int(start):int(start) + length] for i, start in enumerate(starts)])


@torch.no_grad()
def mel_statistics(dataset):
    if not len(dataset):
        raise ValueError("No training items for mel statistics")
    total = torch.zeros(data.MELS, dtype=torch.float64)
    squared = torch.zeros_like(total)
    n = 0
    for i in range(len(dataset)):
        spec = dataset[i][1]
        mel = data.logmel(spec[None])[0].double()
        total += mel.sum(-1)
        squared += mel.square().sum(-1)
        n += mel.shape[-1]
    mean = total / n
    std = (squared / n - mean.square()).clamp_min(1e-4).sqrt()
    return mean.float()[None, :, None], std.float()[None, :, None]


def load_weights(root, run, device="cpu", checkpoint="best"):
    root = Path(root)
    path = root / "flow_runs" / run / f"{checkpoint}.pt"
    if not path.exists() and checkpoint == "best":
        path = path.with_name("last.pt")
    if not path.exists():
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["architecture"] != ARCHITECTURE:
        raise ValueError("Checkpoint is not a v2 flow student")
    model = FlowStudent(len(ck["vocab"]), ck["model_config"]).to(device)
    model.load_state_dict(ck["ema"])
    assert_budget(model)
    return model.eval(), ck, path


@torch.no_grad()
def evaluate(model, validation, phase, device, n=8, pitch_weight=0.2):
    model.eval()
    results = []
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(12345)
        for i in range(min(n, len(validation))):
            x, xl, spec, sl, wave, f0, vuv = [a.to(device) for a in data.collate([validation[i]])]
            mel = data.logmel(spec)
            if phase == "vocoder":
                # Fixed center crop, independent of the training crop RNG.
                frames = min(64, int(sl[0]))
                first = (int(sl[0]) - frames) // 2
                predicted = model.vocoder(mel[:, :, first:first + frames])
                target = wave[:, :, first * data.HOP:(first + frames) * data.HOP]
                score = F.l1_loss(data.logmel(data.spectrogram(predicted[:, 0])), mel[:, :, first:first + frames])
            else:
                score, _, _ = model.objective(x, xl, mel, sl, f0, vuv, step=10000, pitch_weight=pitch_weight)
            results.append(float(score))
    return float(np.mean(results)) if results else None


def corpus_fingerprint(root):
    """Stable across the T4/RTX filesystem roots; data bundle checksums cover assets."""
    rows = [{**r, "wav": f"wavs/{r['id']}.wav"} for r in data.read_jsonl(Path(root) / "corpus.jsonl")]
    return data.digest(json.dumps(rows, ensure_ascii=False, sort_keys=True))


def train(root, phase="acoustic", run="flow10m", vocoder_run="vocoder", steps=100000,
          batch_size=32, workers=4, device="cuda", precision="bf16", save_every=1000,
          eval_every=1000, segment_frames=32, seed=2026, pitch_weight=0.2, allow_small=False):
    root = Path(root)
    folder = root / "flow_runs" / run
    folder.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select CPU only for explicit smoke tests")
    if phase not in {"acoustic", "vocoder"}:
        raise ValueError(phase)
    data.seed_all(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    training, validation = data.SpeechDataset(root, "train"), data.SpeechDataset(root, "val")
    if not len(training) or (not allow_small and (len(training) < 1000 or len(validation) < 20)):
        raise ValueError("Full training requires >=1000 train and >=20 validation items; use pilot/smoke mode first")
    model = FlowStudent(len(training.vocab)).to(dev)
    budget = assert_budget(model)
    corpus_sha = corpus_fingerprint(root)
    vocab_sha = data.digest(json.dumps(training.vocab, ensure_ascii=False))
    feature_metadata = root / "flow_mel_statistics.json"
    if feature_metadata.exists():
        stat = json.loads(feature_metadata.read_text())
        if stat["corpus_sha"] != corpus_sha:
            raise ValueError("Cached mel statistics belong to a different corpus")
        mean = torch.tensor(stat["mean"]).reshape(1, data.MELS, 1)
        std = torch.tensor(stat["std"]).reshape(1, data.MELS, 1)
    else:
        mean, std = mel_statistics(training)
        data.atomic_json(feature_metadata, {"corpus_sha": corpus_sha, "mean": mean.flatten().tolist(), "std": std.flatten().tolist(),
                                           "split": "train only"})
    model.mel_mean.copy_(mean.to(dev)); model.mel_std.copy_(std.to(dev))
    if phase == "acoustic" and not (folder / "last.pt").exists():
        trained_vocoder, previous, _ = load_weights(root, vocoder_run, dev)
        if previous["training_config"]["phase"] != "vocoder":
            raise ValueError("Expected a trained vocoder checkpoint")
        if previous["training_config"]["corpus_sha"] != corpus_sha or previous["training_config"]["vocab_sha"] != vocab_sha:
            raise ValueError("Vocoder and acoustic runs must use the same corpus/vocabulary")
        model.load_state_dict(trained_vocoder.state_dict())
        del trained_vocoder
    for name, parameter in model.named_parameters():
        parameter.requires_grad_((name.startswith("vocoder.")) == (phase == "vocoder"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-7, betas=(0.8, 0.99) if phase == "vocoder" else (0.9, 0.98))
    disc = Discriminator().to(dev) if phase == "vocoder" else None
    disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.8, 0.99)) if disc is not None else None
    enabled = dev.type == "cuda" and precision != "fp32"
    if enabled and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; select fp16 or fp32")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=enabled and precision == "fp16")
    config = {"phase": phase, "batch_size": batch_size, "segment_frames": segment_frames, "seed": seed,
              "pitch_weight": pitch_weight, "precision": precision, "corpus_sha": corpus_sha, "vocab_sha": vocab_sha,
              "source_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    step, best = 0, float("inf")
    rng = torch.Generator().manual_seed(seed)
    if (folder / "last.pt").exists():
        ck = torch.load(folder / "last.pt", map_location=dev, weights_only=False)
        if ck["training_config"] != config:
            raise ValueError("Resume config/source/data mismatch; use the original configuration or a new run name")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        ema = ck["ema"]
        if disc is not None:
            disc.load_state_dict(ck["discriminator"]); disc_optimizer.load_state_dict(ck["disc_optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        torch.set_rng_state(ck["torch_rng"].cpu()); rng.set_state(ck["loader_rng"].cpu())
        np.random.set_state(ck["numpy_rng"]); random.setstate(ck["python_rng"])
        if dev.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in ck["cuda_rng"]])
    data.atomic_json(folder / "parameters.json", {**budget,
                     "training_discriminator_parameters": sum(p.numel() for p in disc.parameters()) if disc is not None else 0})
    data.atomic_json(folder / "config.json", config)
    loader = torch.utils.data.DataLoader(training, batch_size=min(batch_size, len(training)), shuffle=True,
                                        collate_fn=data.collate, num_workers=workers, generator=rng,
                                        pin_memory=dev.type == "cuda", persistent_workers=workers > 0,
                                        multiprocessing_context="spawn" if workers > 0 else None)
    iterator = iter(loader)

    def save(path):
        data.save_checkpoint(path, {"architecture": ARCHITECTURE, "model_config": model.config, "vocab": training.vocab,
                "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
                "discriminator": disc.state_dict() if disc is not None else None,
                "disc_optimizer": disc_optimizer.state_dict() if disc is not None else None,
                "scaler": scaler.state_dict(), "step": step, "best": best, "training_config": config,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "loader_rng": rng.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                "resume_note": "Optimizer and RNG restored; data iterator restarts, not exact batch replay"})

    try:
        while step < steps:
            tick = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            ids, lengths, spec, frame_lengths, wave, f0, vuv = [v.to(dev, non_blocking=True) for v in batch]
            raw_mel = data.logmel(spec.float())
            optimizer.zero_grad(set_to_none=True)
            amp = lambda: torch.autocast(dev.type, dtype=amp_dtype, enabled=enabled)
            model.train()
            if phase == "vocoder":
                if int(frame_lengths.min()) < segment_frames:
                    raise ValueError("Utterance shorter than the vocoder training segment")
                starts = (torch.rand(len(ids), device=dev) * (frame_lengths - segment_frames + 1)).long()
                target_mel = crop(raw_mel, starts, segment_frames)
                target_wave = crop(wave, starts * data.HOP, segment_frames * data.HOP)
                with amp():
                    predicted = model.vocoder(target_mel)
                disc_optimizer.zero_grad(set_to_none=True)
                real_scores, fake_scores = disc(target_wave.float()), disc(predicted.detach().float())
                loss_disc = sum((1 - r[0]).square().mean() + f[0].square().mean() for r, f in zip(real_scores, fake_scores))
                if not torch.isfinite(loss_disc):
                    raise FloatingPointError("Nonfinite discriminator loss")
                scaler.scale(loss_disc).backward(); scaler.unscale_(disc_optimizer)
                nn.utils.clip_grad_norm_(disc.parameters(), 100); scaler.step(disc_optimizer)
                for p in disc.parameters():
                    p.requires_grad_(False)
                # Fix spectral-normalization power-iteration buffers for the G pass.
                disc.eval()
                real_scores, fake_scores = disc(target_wave.float()), disc(predicted.float())
                adversarial = sum((1 - f[0]).square().mean() for f in fake_scores)
                matching = sum((r.detach() - f).abs().mean() for real, fake in zip(real_scores, fake_scores)
                               for r, f in zip(real[1], fake[1])) * 2
                mel_loss = F.l1_loss(data.logmel(data.spectrogram(predicted[:, 0].float())), target_mel) * 45
                spectral = multi_resolution_stft(predicted.float(), target_wave.float())
                loss = mel_loss + spectral + matching + adversarial
                metrics = {"mel": mel_loss, "stft": spectral, "feature_matching": matching, "adversarial": adversarial,
                           "discriminator": loss_disc}
            else:
                with amp():
                    loss, metrics, _ = model.objective(ids, lengths, raw_mel, frame_lengths, f0, vuv, step, pitch_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradient = nn.utils.clip_grad_norm_(parameters, 100)
            if not torch.isfinite(gradient):
                raise FloatingPointError("Nonfinite gradients")
            scaler.step(optimizer); scaler.update()
            if disc is not None:
                for p in disc.parameters():
                    p.requires_grad_(True)
                disc.train()
            step += 1
            warmup = min(1.0, (step + 1) / 1000)
            for group in optimizer.param_groups:
                group["lr"] = 2e-4 * warmup * (0.99999 ** step)
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if name in {"mel_mean", "mel_std"}:
                        ema[name].copy_(value)
                    else:
                        ema[name].lerp_(value.detach(), 0.001)
            if step == 1 or step % 25 == 0:
                record = {"step": step, "phase": phase, "loss": float(loss.detach()), "gradient": float(gradient),
                          "seconds": time.perf_counter() - tick, **{k: float(v.detach()) for k, v in metrics.items()}}
                with (folder / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(record, flush=True)
            if step % eval_every == 0 and len(validation):
                # Evaluate EMA weights; restore live optimizer weights afterwards.
                live = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
                value = evaluate(model, validation, phase, dev, pitch_weight=pitch_weight)
                model.load_state_dict(live)
                with (folder / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, "phase": phase, "validation_loss": value}) + "\n")
                if value is not None and math.isfinite(value) and value < best:
                    best = value; save(folder / "best.pt")
            if step % save_every == 0:
                save(folder / "last.pt")
    except KeyboardInterrupt:
        save(folder / "last.pt")
        raise
    save(folder / "last.pt")
    if phase == "acoustic":
        export(root, run)
    print("Completed", phase, step, budget, flush=True)


def export(root, run):
    model, ck, source = load_weights(root, run)
    if ck["training_config"]["phase"] != "acoustic":
        raise ValueError("A vocoder-only checkpoint is not a trained TTS student")
    budget = assert_budget(model)
    path = Path(root) / "flow_runs" / run / "student_deploy.pt"
    # Count the actual exported tensors too, not only the in-memory module.
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    expected = budget["deployment_parameters"] + budget["buffer_elements"]
    if sum(v.numel() for v in state.values()) != expected or expected >= LIMIT:
        raise ValueError("Exported state exceeds budget or parameter accounting disagrees")
    data.save_checkpoint(path, {"architecture": ARCHITECTURE, "model_config": model.config, "vocab": ck["vocab"],
                     "state": state, "parameters": budget, "step": ck["step"], "source_checkpoint": source.name,
                     "sample_rate": data.SR, "quality_verified": False})
    return path


def load_deployment(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["architecture"] != ARCHITECTURE:
        raise ValueError("Wrong architecture")
    model = FlowStudent(len(ck["vocab"]), ck["model_config"]).to(device)
    model.load_state_dict(ck["state"], strict=True)
    assert_budget(model)
    if sum(v.numel() for v in ck["state"].values()) >= LIMIT:
        raise ValueError("Deployment state contains too many elements")
    return model.eval(), ck["vocab"]


def synthesize(root, run, nfe=16, device="cpu", threads=4):
    root = Path(root)
    torch.set_num_threads(threads)
    manifest = data.read_jsonl(root / "benchmark_manifest.jsonl")
    if len(manifest) != 137:
        raise ValueError("Expected the full 137-item benchmark")
    deployment = root / "flow_runs" / run / "student_deploy.pt"
    model, vocab = load_deployment(deployment, device)
    name = f"{run}_nfe{nfe}"
    output = root / "benchmark_wavs" / name
    output.mkdir(parents=True, exist_ok=True)
    ids = data.encode("Xin chào.", vocab)[None].to(device)
    model.speak(ids, torch.tensor([ids.shape[1]], device=device), steps=nfe)
    timings, failures = [], []
    for i, row in enumerate(manifest):
        data.seed_all(10000 + i)
        ids = data.encode(row["text"], vocab)[None].to(device)
        try:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            waveform = model.speak(ids, torch.tensor([ids.shape[1]], device=device), steps=nfe)[0, 0].float().cpu().numpy()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            metrics = data.sanity(waveform, data.SR)
            path = output / row["output_wav"]
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, waveform, data.SR, subtype="PCM_16")
            timings.append({"id": row["id"], "subset": row["subset"], "seconds": elapsed, "nfe": nfe,
                            "device": device, "threads": threads, "rtf": elapsed / metrics["duration"],
                            "oov_characters": int((ids == 1).sum()), "sample_rate": data.SR, **metrics})
        except (ValueError, RuntimeError) as error:
            failures.append({"id": row["id"], "error": str(error)})
    data.write_jsonl(output / "timings.jsonl", timings)
    data.write_jsonl(output / "failures.jsonl", failures)
    data.atomic_json(output / "timing_summary.json", {"generated": len(timings), "failed": len(failures),
                    "pooled_rtf": sum(r["seconds"] for r in timings) / max(sum(r["duration"] for r in timings), 1e-6),
                    "protocol": "One warmup; fixed seeds; full-utterance mel+waveform synthesis, native 24 kHz"})
    # Reuse the common comparison writer without pretending this is a VITS run.
    marker = root / "runs" / name
    marker.mkdir(parents=True, exist_ok=True)
    data.atomic_json(marker / "parameters.json", counts(model))
    data.atomic_json(marker / "deployment_file.json", {"path": str(deployment.resolve()), "size_bytes": deployment.stat().st_size})
    if failures:
        raise RuntimeError(f"Incomplete benchmark: {len(failures)} failed items; see failures.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["budget", "train", "export", "synthesize"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--phase", default="acoustic", choices=["vocoder", "acoustic"])
    parser.add_argument("--run", default="flow10m")
    parser.add_argument("--vocoder-run", default="vocoder")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--segment-frames", type=int, default=32)
    parser.add_argument("--pitch-weight", type=float, default=0.2)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "budget":
        print(json.dumps(counts(FlowStudent(args.vocab_size)), indent=2))
    elif args.stage == "train":
        train(args.root, args.phase, args.run, args.vocoder_run, args.steps, args.batch_size, args.workers,
              args.device, args.precision, args.save_every, args.eval_every, args.segment_frames,
              pitch_weight=args.pitch_weight, allow_small=args.allow_small)
    elif args.stage == "export":
        print(export(args.root, args.run))
    elif args.stage == "synthesize":
        synthesize(args.root, args.run, args.nfe, args.device, args.threads)


if __name__ == "__main__":
    main()
