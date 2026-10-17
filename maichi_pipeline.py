"""Mai Chi single-speaker distillation research pipeline, v0.1.

Student: compact VITS with an optional frame-level F0/voicing predictor and
pitch conditioning. This is NOT a reproduction of Teacher in the Loop.
VITS forward/inference and monotonic alignment are adapted from jaywalnut310/vits
(MIT), commit 2e561ba58618d021b5b8323d3765880f7e0ecfdb.
Copyright (c) 2021 Jaehyeon Kim. See the upstream LICENSE, retained by bootstrap.
New pipeline code is MIT. Dataset and voice rights are separate.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import types
import unicodedata

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

TEACHER_REV = "c2bfbd67dc648cac455077333f7cf5c18a2e3bb4"
SPEECH_REV = "40d6c083c420f5b620a2f9cd8f5cee9045fed715"
BENCH_REV = "bba02b598c84b41ceb67b141bfbd97c6de149103"
VITS_REV = "2e561ba58618d021b5b8323d3765880f7e0ecfdb"
SR, HOP, FFT, MELS = 24000, 256, 1024, 80


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def text_key(text):
    return "".join(c for c in normalized(text).casefold() if c.isalnum())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap(vits_dir):
    """Pin upstream source; replace only its optional Cython MAS extension."""
    vits_dir = Path(vits_dir).resolve()
    if not (vits_dir / "models.py").exists():
        vits_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/jaywalnut310/vits.git", str(vits_dir)], check=True)
        subprocess.run(["git", "-C", str(vits_dir), "checkout", VITS_REV], check=True)
    revision = subprocess.check_output(["git", "-C", str(vits_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != VITS_REV:
        raise RuntimeError(f"Unexpected VITS revision: {revision}. Use a separate pinned checkout.")
    if str(vits_dir) not in sys.path:
        sys.path.insert(0, str(vits_dir))
    # Same dynamic program as upstream core.pyx; Numba avoids compiler/NumPy ABI
    # failures on changing Colab runtimes. Alignment runs in float32 on CPU.
    from numba import njit

    @njit(cache=False)
    def mas(values, ys, xs):
        paths = np.zeros(values.shape, dtype=np.float32)
        for b in range(values.shape[0]):
            ty, tx = ys[b], xs[b]
            for y in range(ty):
                for x in range(max(0, tx + y - ty), min(tx, y + 1)):
                    stay = -1e9 if x == y else values[b, y - 1, x]
                    advance = (0.0 if y == 0 else -1e9) if x == 0 else values[b, y - 1, x - 1]
                    values[b, y, x] += max(stay, advance)
            index = tx - 1
            for y in range(ty - 1, -1, -1):
                paths[b, y, index] = 1.0
                if index != 0 and (index == y or values[b, y - 1, index] < values[b, y - 1, index - 1]):
                    index -= 1
        return paths

    def maximum_path(values, mask):
        ys = mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32)
        xs = mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32)
        if np.any(ys < xs):
            raise ValueError("MAS requires at least one acoustic frame per text token.")
        a = values.detach().float().cpu().numpy().copy()
        return torch.from_numpy(mas(a, ys, xs)).to(device=values.device, dtype=values.dtype)

    sys.modules["monotonic_align"] = types.SimpleNamespace(maximum_path=maximum_path)
    import models
    import commons
    import losses
    return models, commons, losses


def make_teacher(root, device="cpu", threads=4):
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from zerotts import ZeroTTS
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA ONNX provider unavailable. Install onnxruntime-gpu or explicitly select CPU.")
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    local = snapshot_download("zeroweight-ai/ZeroTTS", revision=TEACHER_REV,
                              allow_patterns=["config.json", "tokenizer.json", "null_voice_emb.npy", "silence_frame.npy",
                                              "onnx/*", "voices/index.json", "voices/maichi/*"], max_workers=4)
    tts = ZeroTTS(local, providers=providers, intra_op_num_threads=threads)
    sessions = [tts.text_encoder_sess, tts.prefix_step_sess, tts.local_frame_decode_sess]
    actual = [s.get_providers() for s in sessions]
    if device == "cuda" and any(p[0] != "CUDAExecutionProvider" for p in actual):
        raise RuntimeError(f"ONNX sessions fell back to CPU: {actual}")
    atomic_json(Path(root) / "teacher_runtime.json", {"revision": TEACHER_REV, "requested": device, "session_providers": actual,
                                                    "note": "Provider registration does not prove every graph node executes on GPU."})
    tts.load_voice("maichi")
    return tts


SMOKE_TEXTS = [
    "Xin chào, mình là Mai Chi. Hôm nay chúng ta thử giọng nói tiếng Việt.",
    "Ma, má, mà, mả, mã, mạ. Bé đi học và mẹ ở nhà.",
    "Sáng nay tôi dùng Python để kiểm tra mô hình trí tuệ nhân tạo.",
]


def sanity(audio, sr):
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("Empty or nonfinite waveform")
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y**2)))
    if duration < 0.5 or rms < 0.001:
        raise ValueError(f"Short or silent waveform: duration={duration:.3f}, rms={rms:.6f}")
    return {"duration": duration, "rms": rms, "peak": float(np.max(np.abs(y))),
            "clipped_fraction": float(np.mean(np.abs(y) >= 0.999))}


def smoke(root, device="cpu", threads=4):
    root = Path(root)
    tts = make_teacher(root, device, threads)
    records = []
    for i, text in enumerate(SMOKE_TEXTS):
        np.random.seed(100 + i)
        start = time.perf_counter()
        audio = tts.synthesize(text, voice="maichi", max_frames=300)
        elapsed = time.perf_counter() - start
        metrics = sanity(audio, tts.sample_rate)
        name = root / "smoke" / f"{i}.wav"
        name.parent.mkdir(parents=True, exist_ok=True)
        tts.save_audio(audio, str(name))
        record = {"text": text, "wav": str(name), "seconds": elapsed, "rtf": elapsed / metrics["duration"], **metrics}
        records.append(record)
        print(record, flush=True)
    atomic_json(root / "smoke.json", {"passed_waveform_sanity": True, "quality_verified": False,
                                    "teacher_revision": TEACHER_REV, "samples": records})


def prepare_benchmark(root):
    from huggingface_hub import snapshot_download
    root = Path(root)
    bench = snapshot_download("zeroweight-ai/ZeroBench-TTS", repo_type="dataset", revision=BENCH_REV,
                              local_dir=root / "ZeroBench-TTS", max_workers=4)
    manifest = root / "benchmark_manifest.jsonl"
    subprocess.run([sys.executable, "-m", "zerobench_eval", "manifest", "--out", str(manifest.resolve())], cwd=bench, check=True)
    rows = read_jsonl(manifest)
    if len(rows) != 137:
        raise ValueError(f"Pinned benchmark expected 137 rows, found {len(rows)}")
    return rows


def zero_speech_texts(seed):
    """Read viewer JSON pages, never fetch audio or use a Parquet callback.

    The viewer API serves cached content. Each page's asset revision must match
    the pinned dataset revision before its text can enter the corpus.
    Raw text and IDs are persisted separately, so exact selection is auditable.
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    base = "https://datasets-server.huggingface.co/rows?"
    rng = random.Random(seed)

    def page(offset):
        params = urllib.parse.urlencode({"dataset": "zeroweight-ai/ZeroSpeech", "config": "default", "split": "train",
                                         "offset": offset, "length": 100})
        for attempt in range(4):
            try:
                req = urllib.request.Request(base + params, headers={"User-Agent": "maichi-research/0.1"})
                with urllib.request.urlopen(req, timeout=45) as response:
                    data = json.load(response)
                for wrapped in data["rows"]:
                    assets = wrapped["row"].get("audio", [])
                    if not isinstance(assets, list) or not assets:
                        raise RuntimeError("Viewer response omitted its dataset revision; cannot verify source.")
                    if any(f"/--/{SPEECH_REV}/--/" not in a.get("src", "") for a in assets):
                        raise RuntimeError("Viewer cache revision differs from pinned ZeroSpeech. Do not mix dataset versions.")
                return data
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    first = page(0)
    total = first["num_rows_total"]
    pages = list(range((total + 99) // 100))
    rng.shuffle(pages)
    print(f"ZeroSpeech: {total} rows; sampling revision-checked transcript pages", flush=True)
    for n, index in enumerate(pages):
        data = first if index == 0 else page(index * 100)
        rows = data["rows"]
        rng.shuffle(rows)
        for wrapped in rows:
            row = wrapped["row"]
            yield {k: row[k] for k in ["id", "text", "kind", "source"]}
        if (n + 1) % 10 == 0:
            print(f"Read {n+1} transcript pages", flush=True)


def collect_texts(root, count=20000, seed=2026, source_jsonl=None):
    """Read revision-checked ZeroSpeech text pages, never materialize audio."""
    root = Path(root)
    bench = read_jsonl(root / "benchmark_manifest.jsonl")
    if len(bench) != 137:
        raise RuntimeError("Run benchmark preparation before selecting training text.")
    excluded = {text_key(r["text"]) for r in bench}
    for r in bench:
        excluded.add(text_key(r.get("ref_text", "")))
        excluded.add(text_key(r.get("text_normalized", "")))
    if source_jsonl:
        source = iter(read_jsonl(source_jsonl))
        source_name = str(Path(source_jsonl).resolve())
    else:
        source = zero_speech_texts(seed)
        source_name = "zeroweight-ai/ZeroSpeech"
    rows, seen = [], set()
    for row in source:
        if row.get("kind", "vi") not in {"vi", "mixed"}:
            continue
        text = normalized(row["text"])
        key = text_key(text)
        if not 20 <= len(text) <= 220 or not key or key in seen:
            continue
        # Conservative substring exclusion removes longer near-copies too.
        if any(k and (key == k or (min(len(k), len(key)) >= 30 and (key in k or k in key))) for k in excluded):
            continue
        if sum(c.isalpha() for c in text) < len(text) * 0.55:
            continue
        seen.add(key)
        uid = digest(key)[:20]
        bucket = int(digest("split:" + key)[:8], 16) % 100
        split = "test" if bucket < 5 else "val" if bucket < 10 else "train"
        rows.append({"id": uid, "source_id": str(row.get("id", "")), "source": row.get("source", source_name),
                     "text": text, "key": key, "split": split})
        if len(rows) >= count:
            break
    if len(rows) != count:
        raise ValueError(f"Requested {count} unique texts but found {len(rows)}")
    if hasattr(source, "close"):
        source.close()
    del source
    write_jsonl(root / "texts.jsonl", rows)
    atomic_json(root / "data_provenance.json", {"source": source_name, "source_revision": SPEECH_REV if not source_jsonl else None,
                                               "benchmark_revision": BENCH_REV, "seed": seed,
                                               "count": count, "texts_sha256": digest(json.dumps(rows, ensure_ascii=False, sort_keys=True)),
                                               "split_counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]},
                                               "license": "ZeroSpeech / ZeroBench CC-BY-NC-4.0; source corpora and voice rights also apply",
                                               "retrieval": "Hugging Face viewer JSON pages, cached asset revision verified; audio URLs are never fetched",
                                               "leakage_limit": "Exact and substring screening; paraphrase/source-document overlap is not ruled out."})


def generate(root, device="cpu", threads=4):
    from scipy.signal import resample_poly
    root = Path(root)
    rows = read_jsonl(root / "texts.jsonl")
    if not rows:
        raise RuntimeError("No training texts")
    passed = json.loads((root / "smoke.json").read_text())
    if not passed["passed_waveform_sanity"]:
        raise RuntimeError("Teacher smoke test must pass first")
    tts = make_teacher(root, device, threads)
    outputs, rejected = [], []
    (root / "wavs").mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        path = root / "wavs" / (row["id"] + ".wav")
        record = root / "wavs" / (row["id"] + ".json")
        if path.exists() and record.exists():
            saved = json.loads(record.read_text())
            if saved["text"] != row["text"] or saved["teacher_revision"] != TEACHER_REV:
                raise RuntimeError("Stale corpus cache; use a new run directory")
            outputs.append(saved)
            continue
        np.random.seed(int(row["id"][:8], 16))
        start = time.perf_counter()
        try:
            # 20 s ceiling, longer than selected short text should require.
            audio = tts.synthesize(row["text"], voice="maichi", max_frames=250)
            stats = sanity(audio, tts.sample_rate)
            rate = len(row["text"]) / stats["duration"]
            if not 1.0 <= stats["duration"] <= 18.0 or not 3.0 <= rate <= 45.0 or stats["clipped_fraction"] > 0.01:
                raise ValueError(f"Duration/rate/clipping filter: {stats}, chars/s={rate:.2f}")
            y = resample_poly(audio.reshape(-1), SR, tts.sample_rate).astype(np.float32)
            sf.write(str(path) + ".tmp.wav", y, SR, subtype="PCM_16")
            Path(str(path) + ".tmp.wav").replace(path)
            saved = {**row, "wav": str(path.resolve()), "teacher_revision": TEACHER_REV,
                     "sample_rate": SR, "generation_seconds": time.perf_counter() - start, **stats}
            atomic_json(record, saved)
            outputs.append(saved)
        except (ValueError, FloatingPointError) as e:
            rejected.append({"id": row["id"], "error": str(e)})
        if (i + 1) % 25 == 0:
            write_jsonl(root / "corpus.jsonl", outputs)
            print(f"Generated {i+1}/{len(rows)}, accepted={len(outputs)}, rejected={len(rejected)}", flush=True)
    write_jsonl(root / "corpus.jsonl", outputs)
    write_jsonl(root / "rejected.jsonl", rejected)
    atomic_json(root / "corpus_summary.json", {"accepted": len(outputs), "rejected": len(rejected),
                                              "hours": sum(r["duration"] for r in outputs) / 3600})


def spectrogram(y):
    y = F.pad(y.unsqueeze(1), ((FFT - HOP) // 2, (FFT - HOP) // 2), mode="reflect").squeeze(1)
    window = torch.hann_window(FFT, device=y.device, dtype=y.dtype)
    return torch.stft(y, FFT, HOP, FFT, window, center=False, return_complex=True).abs().clamp_min(1e-5)


_mel_cache = {}


def logmel(spec):
    import librosa
    key = (str(spec.device), spec.dtype)
    if key not in _mel_cache:
        basis = librosa.filters.mel(sr=SR, n_fft=FFT, n_mels=MELS, fmin=0, fmax=SR // 2)
        _mel_cache[key] = torch.tensor(basis, device=spec.device, dtype=spec.dtype)
    return torch.matmul(_mel_cache[key], spec).clamp_min(1e-5).log()


def extract_one(row, root):
    import pyworld
    target = Path(root) / "features" / (row["id"] + ".npz")
    if target.exists():
        return
    y, sr = sf.read(row["wav"], dtype="float32")
    if sr != SR or y.ndim != 1:
        raise ValueError("Expected 24 kHz mono corpus")
    y = y[:len(y) // HOP * HOP]
    spec = spectrogram(torch.from_numpy(y)[None])[0].numpy()
    raw_f0, t = pyworld.dio(y.astype(np.float64), SR, f0_floor=65, f0_ceil=600, frame_period=1000 * HOP / SR)
    raw_f0 = pyworld.stonemask(y.astype(np.float64), raw_f0, t, SR)
    # WORLD timestamps start at zero; STFT frame centers are hop/2 + k*hop
    centers = (np.arange(spec.shape[1]) * HOP + HOP / 2) / SR
    nearest = np.minimum(np.floor(centers / (HOP / SR) + 0.5).astype(int), len(raw_f0) - 1)
    f0 = raw_f0[nearest].astype(np.float32)
    vuv = (f0 > 0).astype(np.float32)
    f0n = np.where(f0 > 0, (np.log(np.maximum(f0, 1)) - 5.3) / 0.5, 0).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, spec=spec.astype(np.float32), f0=f0, f0n=f0n, vuv=vuv)
    tmp.replace(target)


def features(root, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    rows = read_jsonl(Path(root) / "corpus.jsonl")
    if not rows:
        raise RuntimeError("Corpus is empty")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(lambda row: extract_one(row, root), rows)):
            if (i + 1) % 100 == 0:
                print(f"Features {i+1}/{len(rows)}", flush=True)
    # Build vocabulary from TRAIN text only. Blank=0 and unknown=1.
    chars = sorted(set("".join(normalized(r["text"]).lower() for r in rows if r["split"] == "train")))
    vocab = ["<blank>", "<unk>"] + chars
    atomic_json(Path(root) / "vocab.json", vocab)
    all_test = [r for r in rows if r["split"] != "train"] + read_jsonl(Path(root) / "benchmark_manifest.jsonl")
    unknown = sorted(set("".join(normalized(r["text"]).lower() for r in all_test)) - set(chars))
    atomic_json(Path(root) / "vocab_audit.json", {"vocab_size": len(vocab), "unseen_characters": unknown,
                                              "note": "No accents are removed; OOVs are mapped to <unk> and counted at evaluation."})


def encode(text, vocab):
    lookup = {c: i for i, c in enumerate(vocab)}
    chars = [lookup.get(c, 1) for c in normalized(text).lower()]
    ids = [0] * (2 * len(chars) + 1)
    ids[1::2] = chars
    return torch.tensor(ids, dtype=torch.long)


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.vocab = json.loads((self.root / "vocab.json").read_text())
        self.rows = [r for r in read_jsonl(self.root / "corpus.jsonl") if r["split"] == split]
        self.rows = [r for r in self.rows if (self.root / "features" / (r["id"] + ".npz")).exists()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with np.load(self.root / "features" / (row["id"] + ".npz")) as f:
            spec, f0n, vuv = [torch.from_numpy(f[k].copy()) for k in ["spec", "f0n", "vuv"]]
        y, _ = sf.read(row["wav"], dtype="float32")
        y = torch.from_numpy(y[:spec.shape[-1] * HOP])[None]
        ids = encode(row["text"], self.vocab)
        if spec.shape[-1] < len(ids):
            raise ValueError(f"Insufficient acoustic frames for MAS: {row['id']}")
        return ids, spec, y, f0n, vuv


def collate(batch):
    batch = sorted(batch, key=lambda item: item[1].shape[-1], reverse=True)
    b, tx, ty = len(batch), max(len(x[0]) for x in batch), max(x[1].shape[-1] for x in batch)
    x, xl = torch.zeros(b, tx, dtype=torch.long), torch.zeros(b, dtype=torch.long)
    spec, sl = torch.zeros(b, FFT // 2 + 1, ty), torch.zeros(b, dtype=torch.long)
    wave, f0, vuv = torch.zeros(b, 1, ty * HOP), torch.zeros(b, 1, ty), torch.zeros(b, 1, ty)
    for i, (ids, s, y, f, v) in enumerate(batch):
        xl[i], sl[i] = len(ids), s.shape[-1]
        x[i, :len(ids)] = ids
        spec[i, :, :s.shape[-1]] = s
        wave[i, :, :y.shape[-1]] = y
        f0[i, 0, :len(f)] = f
        vuv[i, 0, :len(v)] = v
    return x, xl, spec, sl, wave, f0, vuv


def model_config(size="small"):
    # Budgets are measured, never assumed. Both checkpoint and deploy counts
    # are reported; enc_q is a training-only posterior encoder.
    h, layers, u = {"tiny": (64, 3, 128), "small": (128, 4, 256), "medium": (192, 6, 384)}[size]
    return dict(inter_channels=h, hidden_channels=h, filter_channels=h * 4, n_heads=2, n_layers=layers,
                kernel_size=3, p_dropout=0.1, resblock="2", resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3], [1, 3], [1, 3]], upsample_rates=[8, 8, 2, 2],
                upsample_initial_channel=u, upsample_kernel_sizes=[16, 16, 4, 4], use_sdp=False)


def make_student(vits_dir, vocab_size, size="small", pitch=True, segment_frames=32):
    models, commons, _ = bootstrap(vits_dir)

    class PitchStudent(models.SynthesizerTrn):
        def __init__(self):
            super().__init__(vocab_size, FFT // 2 + 1, segment_frames, **model_config(size))
            self.with_pitch = pitch
            if pitch:
                self.pitch_head = nn.Sequential(nn.Conv1d(self.inter_channels + 2, 64, 3, padding=1), nn.ReLU(),
                                                nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 2, 1))
                self.pitch_projection = nn.Conv1d(2, self.inter_channels, 1)
                nn.init.zeros_(self.pitch_projection.weight)
                nn.init.zeros_(self.pitch_projection.bias)

        def pitch_features(self, prior, mask, attn):
            # Within-token position lets the pitch head represent a rising or
            # dipping contour even when expanded text states are constant.
            a = attn.squeeze(1)
            duration = a.sum(1).unsqueeze(-1)
            start = duration.cumsum(1) - duration
            frame_start = torch.matmul(a, start).transpose(1, 2)
            frame_duration = torch.matmul(a, duration).transpose(1, 2).clamp_min(1)
            frame = torch.arange(prior.shape[-1], device=prior.device, dtype=prior.dtype)[None, None]
            position = ((frame - frame_start + 0.5) / frame_duration).clamp(0, 1)
            inp = torch.cat([prior, position, torch.sin(math.pi * position)], 1) * mask
            pred = self.pitch_head(inp) * mask
            return pred[:, :1], pred[:, 1:]

        def forward(self, ids, lengths, spec, spec_lengths, target_f0=None, target_vuv=None, predicted_mix=1.0):
            x, mp, lp, xm = self.enc_p(ids, lengths)
            z, mq, lq, ym = self.enc_q(spec, spec_lengths)
            zp = self.flow(z, ym)
            with torch.no_grad():
                # Float32 MAS scores avoid half-precision underflow/overflow.
                a, b, c = zp.float(), mp.float(), lp.float().clamp(-7, 7)
                inv = torch.exp(-2 * c)
                scores = (-0.5 * math.log(2 * math.pi) - c).sum(1, keepdim=True)
                scores = scores + torch.matmul(-0.5 * a.square().transpose(1, 2), inv)
                scores = scores + torch.matmul(a.transpose(1, 2), b * inv) - 0.5 * (b.square() * inv).sum(1, keepdim=True)
                mask = xm.unsqueeze(2) * ym.unsqueeze(-1)
                import monotonic_align
                attn = monotonic_align.maximum_path(scores, mask.squeeze(1)).unsqueeze(1).detach()
            durations = attn.sum(2)
            logw = self.dp(x, xm)
            duration_loss = (((logw - torch.log(durations + 1e-6) * xm).square()) * xm).sum() / xm.sum()
            mp = torch.matmul(attn.squeeze(1), mp.transpose(1, 2)).transpose(1, 2)
            lp = torch.matmul(attn.squeeze(1), lp.transpose(1, 2)).transpose(1, 2)
            pf, pv = None, None
            conditioned = z
            if self.with_pitch:
                pf, pv = self.pitch_features(mp, ym, attn)
                probability = torch.sigmoid(pv)
                predicted = torch.cat([pf * probability, probability], 1)
                if target_f0 is not None:
                    observed = torch.cat([target_f0 * target_vuv, target_vuv], 1)
                    condition = (1 - predicted_mix) * observed + predicted_mix * predicted
                else:
                    condition = predicted
                conditioned = z + self.pitch_projection(condition) * ym
            zs, starts = commons.rand_slice_segments(conditioned, spec_lengths, self.segment_size)
            wave = self.dec(zs)
            return wave, duration_loss, attn, starts, ym, (zp, lp, mp, lq), (pf, pv)

        @torch.no_grad()
        def speak(self, ids, lengths, noise_scale=0.667, length_scale=1.0, max_frames=2000):
            x, mp, lp, xm = self.enc_p(ids, lengths)
            logw = self.dp(x, xm).clamp(-3, 4)
            w = torch.ceil(torch.exp(logw) * xm * length_scale)
            yl = torch.clamp_min(w.sum([1, 2]), 1).long()
            if int(yl.max()) > max_frames:
                raise RuntimeError(f"Predicted {int(yl.max())} frames; model likely not converged or input too long.")
            ym = commons.sequence_mask(yl).unsqueeze(1).to(xm.dtype)
            attn = commons.generate_path(w, xm.unsqueeze(2) * ym.unsqueeze(-1))
            mp = torch.matmul(attn.squeeze(1), mp.transpose(1, 2)).transpose(1, 2)
            lp = torch.matmul(attn.squeeze(1), lp.transpose(1, 2)).transpose(1, 2)
            zp = mp + torch.randn_like(mp) * torch.exp(lp.clamp(-7, 7)) * noise_scale
            z = self.flow(zp, ym, reverse=True)
            if self.with_pitch:
                pf, pv = self.pitch_features(mp, ym, attn)
                p = torch.sigmoid(pv)
                z = z + self.pitch_projection(torch.cat([pf * p, p], 1)) * ym
            return self.dec(z * ym)

    return PitchStudent()


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    posterior = sum(p.numel() for p in model.enc_q.parameters())
    return {"checkpoint_generator_parameters": total, "deployment_parameters": total - posterior,
            "training_only_posterior_parameters": posterior,
            "note": "Deployment includes text encoder, duration, flow, waveform generator and pitch modules; excludes discriminator and enc_q."}


def audit_corpus(root, count=32, device="cuda"):
    """Small, separate ASR audit. This is not the official dual-ASR benchmark.

    Use rows with plain alphabetic text to avoid treating number expansion as
    an ASR defect. Retain both text and hypothesis for human review. Do not
    tune the student or remove benchmark items based on benchmark results.
    """
    import librosa
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from huggingface_hub import model_info
    from jiwer import wer, cer
    root = Path(root)
    candidates = [r for r in read_jsonl(root / "corpus.jsonl") if not any(c.isdigit() for c in r["text"])]
    rng = random.Random(2026)
    selected = rng.sample(candidates, min(count, len(candidates)))
    model_id = "vinai/PhoWhisper-small"
    revision = model_info(model_id).sha
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, revision=revision).to(device).eval()
    rows = []
    for row in selected:
        y, sr = sf.read(row["wav"], dtype="float32")
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        inp = processor(y, sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to(device)
        with torch.inference_mode():
            ids = asr.generate(**inp, language="vi", task="transcribe", max_new_tokens=256)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        normalize = lambda s: " ".join(re.sub(r"[^\w\s]", " ", normalized(s).lower()).split())
        target, pred = normalize(row["text"]), normalize(hyp)
        rows.append({"id": row["id"], "split": row["split"], "text": row["text"], "hypothesis": hyp,
                     "wer": wer(target, pred), "cer": cer(target, pred), "wav": row["wav"]})
    write_jsonl(root / "corpus_asr_audit.jsonl", rows)
    atomic_json(root / "corpus_asr_audit_summary.json", {"n": len(rows), "asr": model_id, "revision": revision,
                "mean_utterance_wer": float(np.mean([r["wer"] for r in rows])) if rows else None,
                "mean_utterance_cer": float(np.mean([r["cer"] for r in rows])) if rows else None,
                "note": "Small single-ASR audit; retains Vietnamese accents. Not a tone classifier or human listening test."})


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(root, vits_dir, name="small_pitch", size="small", pitch=True, steps=100000,
          batch_size=32, workers=4, lr=2e-4, save_every=1000, validate_every=1000,
          seed=2026, precision="bf16", segment_frames=32, device="cuda", allow_small=False):
    root, out = Path(root), Path(root) / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use CPU only for the explicit smoke test.")
    dev = torch.device(device)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ds, validation = SpeechDataset(root, "train"), SpeechDataset(root, "val")
    if not allow_small and (len(ds) < 1000 or len(validation) < 20):
        raise ValueError("Full training needs >=1000 train and >=20 validation items; run the pilot first.")
    if len(ds) < 1:
        raise ValueError("No training samples")
    for i in range(min(8, len(ds))):
        if ds[i][1].shape[-1] < segment_frames:
            raise ValueError("Training segment exceeds utterance length")
    g = make_student(vits_dir, len(ds.vocab), size, pitch, segment_frames).to(dev)
    models, commons, losses = bootstrap(vits_dir)
    d = models.MultiPeriodDiscriminator().to(dev)
    og = torch.optim.AdamW(g.parameters(), lr=lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(d.parameters(), lr=lr, betas=(0.8, 0.99))
    enabled = dev.type == "cuda" and precision != "fp32"
    if enabled and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; select fp16 or fp32")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=enabled and precision == "fp16")
    config = {"size": size, "pitch": pitch, "segment_frames": segment_frames, "seed": seed,
              "batch_size": batch_size, "precision": precision, "lr": lr,
              "vocab_sha256": digest(json.dumps(ds.vocab, ensure_ascii=False)),
              "corpus_sha256": hashlib.sha256((root / "corpus.jsonl").read_bytes()).hexdigest(),
              "vits_revision": VITS_REV, "teacher_revision": TEACHER_REV}
    latest = out / "last.pt"
    start, best = 0, float("inf")
    sampler_generator = torch.Generator().manual_seed(seed)
    if latest.exists():
        ck = torch.load(latest, map_location=dev, weights_only=False)
        if ck["config"] != config:
            raise ValueError("Resume configuration/corpus mismatch. Keep the config or choose another run name.")
        g.load_state_dict(ck["g"]); d.load_state_dict(ck["d"])
        og.load_state_dict(ck["og"]); od.load_state_dict(ck["od"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["step"], ck["best"]
        torch.set_rng_state(ck["torch_rng"].cpu()); np.random.set_state(ck["numpy_rng"]); random.setstate(ck["python_rng"])
        if dev.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        sampler_generator.set_state(ck["sampler_rng"].cpu())
    atomic_json(out / "config.json", config)
    atomic_json(out / "parameters.json", parameter_counts(g))
    env = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
           "packages": {p: importlib.metadata.version(p) for p in ["numpy", "librosa", "pyworld", "zerotts"]}}
    atomic_json(out / "environment.json", env)
    loader = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True,
                                        num_workers=workers, collate_fn=collate, pin_memory=dev.type == "cuda",
                                        generator=sampler_generator, persistent_workers=workers > 0)
    iterator = iter(loader)
    history = out / "train.jsonl"

    def checkpoint(step):
        save_checkpoint(latest, {"g": g.state_dict(), "d": d.state_dict(), "og": og.state_dict(), "od": od.state_dict(),
                                "scaler": scaler.state_dict(), "step": step, "best": best, "config": config,
                                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                                "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                                "sampler_rng": sampler_generator.get_state(),
                                "resume_note": "Optimizer/RNG resume; data iterator restarts at checkpoint, not bit-exact batch replay."})

    step = start
    try:
        g.train(); d.train()
        while step < steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            x, xl, spec, sl, y, f0, vuv = [v.to(dev, non_blocking=True) for v in batch]
            tick = time.perf_counter()
            mix = min(1.0, step / 20000)
            amp = lambda: torch.autocast(device_type=dev.type, dtype=dtype, enabled=enabled)
            og.zero_grad(set_to_none=True)
            with amp():
                yh, ldur, attn, starts, ym, latent, pitch_pred = g(x, xl, spec, sl, f0, vuv, mix)
            yt = commons.slice_segments(y, starts * HOP, segment_frames * HOP)
            # Loss STFTs and KL are float32 even under BF16/FP16.
            refmel = commons.slice_segments(logmel(spec.float()), starts, segment_frames)
            genmel = logmel(spectrogram(yh.float().squeeze(1)))
            lmel = F.l1_loss(genmel, refmel) * 45.0
            zp, lp, mp, lq = latent
            lkl = losses.kl_loss(zp.float(), lq.float(), mp.float(), lp.float(), ym.float())
            lpitch = yh.sum() * 0
            if pitch:
                pf, pv = pitch_pred
                voiced = vuv * ym
                lpitch = ((pf.float() - f0).abs() * voiced).sum() / voiced.sum().clamp_min(1)
                lpitch = lpitch + (F.binary_cross_entropy_with_logits(pv.float(), vuv, reduction="none") * ym).sum() / ym.sum()
            od.zero_grad(set_to_none=True)
            with amp():
                dr, dg, _, _ = d(yt, yh.detach())
            ld = losses.discriminator_loss(dr, dg)[0]
            if not torch.isfinite(ld):
                raise FloatingPointError("Nonfinite discriminator loss")
            scaler.scale(ld).backward()
            scaler.unscale_(od); nn.utils.clip_grad_norm_(d.parameters(), 100)
            scaler.step(od)
            # Freeze discriminator parameters for G but retain gradient to audio.
            for p in d.parameters():
                p.requires_grad_(False)
            with amp():
                dr, dg, fr, fg = d(yt, yh)
            ladv, lfm = losses.generator_loss(dg)[0], losses.feature_loss(fr, fg)
            lg = lmel + lkl + ldur.float() + ladv + lfm + 2.0 * lpitch
            if not torch.isfinite(lg):
                raise FloatingPointError("Nonfinite generator loss")
            scaler.scale(lg).backward()
            scaler.unscale_(og); grad = nn.utils.clip_grad_norm_(g.parameters(), 100)
            scaler.step(og); scaler.update()
            for p in d.parameters():
                p.requires_grad_(True)
            step += 1
            # Smooth step-based exponential decay; independent of corpus size.
            for opt in [og, od]:
                for group in opt.param_groups:
                    group["lr"] = lr * (0.99999 ** step)
            if step == 1 or step % 25 == 0:
                row = {"step": step, "generator": float(lg.detach()), "discriminator": float(ld.detach()),
                       "mel": float(lmel.detach()), "kl": float(lkl.detach()), "duration": float(ldur.detach()),
                       "pitch": float(lpitch.detach()), "grad_norm": float(grad), "predicted_pitch_mix": mix,
                       "seconds": time.perf_counter() - tick}
                with history.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(row, flush=True)
            if step % validate_every == 0 and len(validation):
                metric = validate(g, validation, dev, out, step, commons)
                if metric < best:
                    best = metric
                    save_checkpoint(out / "best.pt", {"g": g.state_dict(), "config": config, "step": step, "val_mel": best})
                g.train()
            if step % save_every == 0:
                checkpoint(step)
    except KeyboardInterrupt:
        checkpoint(step)
        print("Interrupted; checkpoint saved", flush=True)
        raise
    checkpoint(step)
    export_student(root, vits_dir, name, dev)
    print("Finished", step, parameter_counts(g), flush=True)


@torch.no_grad()
def validate(g, ds, dev, out, step, commons):
    """Teacher-forced reconstruction for model selection, never headline TTS quality."""
    g.eval()
    scores = []
    # Isolate evaluation RNG from training so checkpoint cadence doesn't alter
    # training noise. Validation does use random posterior samples/segments.
    with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else []):
        torch.manual_seed(12345)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(12345)
        for i in range(min(8, len(ds))):
            x, xl, s, sl, y, f0, vuv = [t.to(dev) for t in collate([ds[i]])]
            yh, _, _, starts, _, _, _ = g(x, xl, s, sl, f0, vuv, 1.0)
            target = commons.slice_segments(logmel(s), starts, g.segment_size)
            value = F.l1_loss(logmel(spectrogram(yh.squeeze(1))), target)
            scores.append(float(value))
            if i == 0:
                try:
                    wav = g.speak(x, xl, max_frames=2000)[0, 0].float().cpu().numpy()
                    sf.write(out / f"validation_{step}.wav", wav, SR)
                except RuntimeError as error:
                    with (out / "validation_inference_failures.jsonl").open("a") as f:
                        f.write(json.dumps({"step": step, "error": str(error)}) + "\n")
    value = float(np.mean(scores))
    with (out / "validation.jsonl").open("a") as f:
        f.write(json.dumps({"step": step, "teacher_forced_mel_l1": value, "n": len(scores)}) + "\n")
    print("Validation", step, value, flush=True)
    return value


def load_student(root, vits_dir, name, device="cpu", checkpoint="best"):
    root = Path(root)
    vocab = json.loads((root / "vocab.json").read_text())
    folder = root / "runs" / name
    path = folder / f"{checkpoint}.pt"
    if not path.exists():
        if checkpoint == "best":
            path = folder / "last.pt"
        if not path.exists():
            raise FileNotFoundError(path)
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    g = make_student(vits_dir, len(vocab), cfg["size"], cfg["pitch"], cfg["segment_frames"]).to(device)
    g.load_state_dict(ck["g"])
    return g.eval(), vocab, ck, path


def export_student(root, vits_dir, name, device="cpu"):
    g, vocab, ck, source = load_student(root, vits_dir, name, device)
    # Strip posterior encoder from deployed weights; discriminator was never
    # part of this generator. Loading deployment is handled explicitly below.
    state = {k: v.cpu() for k, v in g.state_dict().items() if not k.startswith("enc_q.")}
    path = Path(root) / "runs" / name / "student_deploy.pt"
    save_checkpoint(path, {"state": state, "config": ck["config"], "vocab": vocab, "sample_rate": SR,
                           "source_checkpoint": source.name, "step": ck["step"], "parameters": parameter_counts(g)})
    return path


def load_deployment(path, vits_dir, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    g = make_student(vits_dir, len(ck["vocab"]), cfg["size"], cfg["pitch"], cfg["segment_frames"])
    del g.enc_q
    g.load_state_dict(ck["state"], strict=True)
    return g.eval().to(device), ck["vocab"]


def synthesize_benchmark(root, vits_dir, name, model="student", device="cpu", threads=4):
    root = Path(root)
    rows = read_jsonl(root / "benchmark_manifest.jsonl")
    if len(rows) != 137:
        raise ValueError("Benchmark manifest must contain all 137 rows")
    torch.set_num_threads(threads)
    tts, student = None, None
    if model == "teacher":
        tts = make_teacher(root, device, threads)
        sr, run = tts.sample_rate, "teacher_maichi"
        tts.synthesize("Xin chào.", voice="maichi", max_frames=80)
    else:
        student, vocab, ck, checkpoint = load_student(root, vits_dir, name, device)
        sr, run = SR, name
        ids = encode("Xin chào.", vocab)[None].to(device)
        student.speak(ids, torch.tensor([ids.shape[1]], device=device), max_frames=2000)
    out = root / "benchmark_wavs" / run
    records, failures = [], []
    for i, row in enumerate(rows):
        raw = row["text"]  # Headline track uses raw text for BOTH systems.
        path = out / row["output_wav"]
        path.parent.mkdir(parents=True, exist_ok=True)
        seed_all(10000 + i)
        try:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if tts is not None:
                audio = tts.synthesize(raw, voice="maichi", max_frames=500).reshape(-1)
            else:
                ids = encode(raw, vocab)[None].to(device)
                audio = student.speak(ids, torch.tensor([ids.shape[1]], device=device))[0, 0].float().cpu().numpy()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stats = sanity(audio, sr)
            sf.write(path, audio, sr, subtype="PCM_16")
            unknown = 0 if tts is not None else int((ids == 1).sum())
            records.append({"id": row["id"], "subset": row["subset"], "seconds": elapsed,
                            "rtf": elapsed / stats["duration"], "oov_characters": unknown,
                            "device": device, "threads": threads, "sample_rate": sr, **stats})
        except (RuntimeError, ValueError) as e:
            failures.append({"id": row["id"], "error": str(e)})
    write_jsonl(out / "timings.jsonl", records)
    write_jsonl(out / "failures.jsonl", failures)
    atomic_json(out / "timing_summary.json", {"generated": len(records), "failed": len(failures),
                                              "pooled_rtf": sum(r["seconds"] for r in records) / max(sum(r["duration"] for r in records), 1e-6),
                                              "protocol": "One warmup, one timed synthesis/item; CUDA sync; raw text; RTF includes waveform decoding."})
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark failures; see failures.jsonl. No complete score is claimed.")


def official_score(root, name, device="cuda", skip_utmos=True):
    root = Path(root).resolve()
    bench = root / "ZeroBench-TTS"
    wav = root / "benchmark_wavs" / name
    if not (wav / "timings.jsonl").exists() or len(read_jsonl(wav / "timings.jsonl")) != 137:
        raise ValueError("Official full score requires all 137 synthesized files")
    command = [sys.executable, "-m", "zerobench_eval", "score", "--wav_dir", str(wav), "--name", name,
               "--device", device, "--out_dir", str(root / "scores" / name)]
    if skip_utmos:
        command += ["--skip_utmos"]
    subprocess.run(command, cwd=bench, check=True)
    # The official speaker-similarity field compares to original reference
    # speakers and is NOT valid for this fixed Mai Chi protocol. Retain raw
    # outputs for audit; the summary tool below never presents it as Mai Chi SIM.


def paired_similarity(root, name, device="cuda"):
    """WavLM verification cosine: student versus Mai Chi teacher, same text.

    This is an adapted single-voice protocol, not the original cloning task.
    Both files are resampled to 16 kHz for the same external verifier.
    """
    import librosa
    from transformers import AutoFeatureExtractor, WavLMForXVector
    root = Path(root)
    model_id = "microsoft/wavlm-base-plus-sv"
    from huggingface_hub import model_info
    revision = model_info(model_id).sha
    extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
    verifier = WavLMForXVector.from_pretrained(model_id, revision=revision).eval().to(device)
    records = []
    for row in read_jsonl(root / "benchmark_manifest.jsonl"):
        embeddings = []
        for run in ["teacher_maichi", name]:
            path = root / "benchmark_wavs" / run / row["output_wav"]
            y, sr = sf.read(path, dtype="float32")
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            inp = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                embeddings.append(verifier(**inp).embeddings.float())
        score = F.cosine_similarity(*embeddings).item()
        records.append({"id": row["id"], "subset": row["subset"], "maichi_teacher_student_cosine": score})
    write_jsonl(root / "benchmark_wavs" / name / "maichi_similarity.jsonl", records)
    atomic_json(root / "benchmark_wavs" / name / "similarity_protocol.json", {"verifier": model_id,
                "revision": revision, "reference": "same-text ZeroTTS Mai Chi waveform", "sample_rate": 16000,
                "mean": float(np.mean([r["maichi_teacher_student_cosine"] for r in records])),
                "not_comparable_to_official_zero_shot_similarity": True})


def report(root):
    import pandas as pd
    root = Path(root)
    rows = []
    for run_dir in sorted((root / "benchmark_wavs").glob("*")):
        timing = read_jsonl(run_dir / "timings.jsonl")
        if not timing:
            continue
        summary = json.loads((run_dir / "timing_summary.json").read_text())
        row = {"run": run_dir.name, "n_generated": len(timing), "pooled_rtf": summary["pooled_rtf"],
               "device": timing[0]["device"], "mean_duration": np.mean([r["duration"] for r in timing]),
               "mean_full_utterance_latency_seconds": np.mean([r["seconds"] for r in timing])}
        sim = run_dir / "similarity_protocol.json"
        if sim.exists():
            row["paired_maichi_cosine"] = json.loads(sim.read_text())["mean"]
        params = root / "runs" / run_dir.name / "parameters.json"
        if params.exists():
            row.update(json.loads(params.read_text()))
            weights = root / "runs" / run_dir.name / "student_deploy.pt"
            if weights.exists():
                row["deploy_mib"] = weights.stat().st_size / 2**20
        quality = root / "scores" / run_dir.name / "summary.json"
        if quality.exists():
            q = json.loads(quality.read_text())
            row["quality_complete"] = q["complete"]
            row["wer_robust_mean"] = q["overall"].get("wer_robust_mean")
            row["utmos_mean"] = q["overall"].get("utmos_mean")
            row["excess_silence_mean"] = q["overall"].get("excess_silence_mean")
            # Original reference-speaker SSIM is intentionally excluded.
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(root / "comparison.csv", index=False)
    print(df.to_string(index=False))
    print("Official ASR/naturalness/excess-silence reports remain under scores/. Original reference-speaker SIM is invalid here.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "benchmark", "texts", "generate", "audit", "features", "train", "synthesize", "score", "similarity", "report"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--vits-dir", default="vits")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--source-jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="small_pitch")
    parser.add_argument("--size", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--no-pitch", action="store_true")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--model", default="student", choices=["teacher", "student"])
    parser.add_argument("--with-utmos", action="store_true")
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke": smoke(args.root, args.device, args.threads)
    elif args.stage == "benchmark": prepare_benchmark(args.root)
    elif args.stage == "texts": collect_texts(args.root, args.count, source_jsonl=args.source_jsonl)
    elif args.stage == "generate": generate(args.root, args.device, args.threads)
    elif args.stage == "audit": audit_corpus(args.root, args.count, args.device)
    elif args.stage == "features": features(args.root, args.workers)
    elif args.stage == "train": train(args.root, args.vits_dir, args.name, args.size, not args.no_pitch,
                                       args.steps, args.batch_size, args.workers, save_every=args.save_every,
                                       validate_every=args.validate_every, precision=args.precision, device=args.device,
                                       allow_small=args.allow_small)
    elif args.stage == "synthesize": synthesize_benchmark(args.root, args.vits_dir, args.name, args.model, args.device, args.threads)
    elif args.stage == "score": official_score(args.root, args.name, args.device, not args.with_utmos)
    elif args.stage == "similarity": paired_similarity(args.root, args.name, args.device)
    elif args.stage == "report": report(args.root)


if __name__ == "__main__":
    main()
