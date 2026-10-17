"""Multi-utterance ZeroTTS generation with the pinned public ONNX export.

Text encoding/global transformer/codec are batched. The public local sampler
has batch-one outputs (its batch-two hidden is CFG), so it is called per lane.
No claim of fully vectorized sampling or guaranteed GPU speedup is made.
Runtime adapter follows ZeroTTS (MIT); see LICENSE_ZEROTTS.txt for attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import maichi_pipeline as data

SAMPLING = {"text_temperature": 1.0, "text_topk": 50, "audio_temperature": 0.8,
            "audio_topk": 25, "audio_topp": 0.95, "audio_repetition_penalty": 1.2,
            "min_frames": 4, "max_frames": 250, "eoa_extra_frames": 1, "cfg_scale": 1.0}


def local_frame(tts, hidden, seen, rng, forbid_eoa):
    """The upstream sampler contract, with an independent RNG per utterance."""
    k = tts.num_codebooks
    ended, codes = tts.local_frame_decode_sess.run(None, {
        "global_hidden": np.ascontiguousarray(hidden, dtype=np.float32),
        "forbid_eoa": np.array([forbid_eoa], dtype=bool),
        "text_temperature": np.array([SAMPLING["text_temperature"]], np.float32),
        "text_topk": np.array([SAMPLING["text_topk"]], np.int64),
        "audio_temperature": np.array([SAMPLING["audio_temperature"]], np.float32),
        "audio_topk": np.array([SAMPLING["audio_topk"]], np.int64),
        "audio_topp": np.array([SAMPLING["audio_topp"]], np.float32),
        "audio_repetition_penalty": np.array([SAMPLING["audio_repetition_penalty"]], np.float32),
        "seen_mask": seen,
        "ctrl_random_u": rng.random_sample(1).astype(np.float32),
        "audio_random_u": rng.random_sample((1, k)).astype(np.float32),
        "cfg_scale": np.array([1.0], np.float32)})
    codes = codes.astype(np.int64)
    seen[0, np.arange(k), codes[0]] = True
    return bool(np.asarray(ended).reshape(-1)[0]), codes[0]


def synthesize_batch(tts, texts, seeds, max_frames=250):
    """Return one waveform and termination record per text. No CFG branching.

    Keeps finished lanes masked; texts and seeds keep their original order.
    This intentionally does not use ZeroTTS.codec.decode(), whose wrapper
    supplies a single-element length vector even for a multi-item batch.
    """
    if not texts or len(texts) != len(seeds) or max_frames < 1:
        raise ValueError("Require nonempty equal-length texts/seeds and positive max_frames")
    b = len(texts)
    tokens = [tts.tokenizer(text).astype(np.int64) for text in texts]
    lengths = np.array([len(t) for t in tokens], dtype=np.int64)
    ids = np.zeros((b, int(lengths.max())), dtype=np.int64)
    for i, row in enumerate(tokens): ids[i, :len(row)] = row
    voice = np.repeat(tts.resolve_voice("maichi"), b, axis=0)
    h, kv, valid, cross, text_valid = tts._prefix_step_init(ids, lengths, voice)
    rngs = [np.random.RandomState(int(seed)) for seed in seeds]
    seen = [np.zeros((1, tts.num_codebooks, tts.codebook_size), dtype=bool) for _ in texts]
    frames = [[] for _ in texts]
    active = np.ones(b, dtype=bool)
    ended = np.zeros(b, dtype=bool)
    for step in range(max_frames):
        next_codes = np.zeros((b, 1, tts.num_codebooks), dtype=np.int64)
        for i in np.flatnonzero(active):
            stop, codes = local_frame(tts, h[i:i+1], seen[i], rngs[i], step < SAMPLING["min_frames"])
            # Keep one EOA trailing frame, identical to the upstream default.
            frames[i].append(codes.copy())
            next_codes[i, 0] = codes
            if stop:
                active[i] = False
                ended[i] = True
        if not active.any() or step + 1 == max_frames:
            break
        h_all, kv, valid = tts.prefix_step_sess.run(None, {
            "external_embed": np.zeros((b, 1, tts.d_model), dtype=np.float32),
            "use_external_embed": np.zeros((b, 1), dtype=bool),
            "frame_codes": next_codes,
            "new_pos": np.full((b, 1), voice.shape[1] + 1 + step, dtype=np.int64),
            "new_valid": active[:, None].copy(), "packed_kv": kv,
            "past_valid": valid, "cross_kv": cross,
            "new_bidirectional": np.zeros((b, 1), dtype=bool), "text_valid": text_valid})
        h = h_all[:, -1]
    code_lengths = np.array([len(x) for x in frames], dtype=np.int32)
    codes = np.zeros((b, int(code_lengths.max()), tts.num_codebooks), dtype=np.int32)
    for i, lane in enumerate(frames): codes[i, :len(lane)] = np.stack(lane)
    audio, audio_lengths = tts.codec._decode_full_sess.run(None, {
        "audio_codes": codes, "audio_code_lengths": code_lengths})
    outputs = []
    for i, n in enumerate(np.asarray(audio_lengths).reshape(-1)):
        wave = audio[i, :, :int(n)].mean(0).astype(np.float32)
        outputs.append({"audio": wave, "frames": int(code_lengths[i]), "ended": bool(ended[i]),
                        "hit_frame_limit": not bool(ended[i]), "codes": codes[i, :code_lengths[i]].copy()})
    if len(outputs) != b:
        raise RuntimeError("Codec did not preserve the requested batch")
    return outputs


def is_oom(error):
    message = str(error).lower()
    return any(t in message for t in ["out of memory", "cudaerrormemoryallocation", "failed to allocate memory", "cudnn_status_alloc_failed"])


def run_adaptive(tts, rows, batch_size, max_frames=250):
    """Shrink on memory errors only; never silently change devices or sampler."""
    offset, size = 0, batch_size
    while offset < len(rows):
        part = rows[offset:offset + size]
        seeds = [int(row["id"][:8], 16) for row in part]
        tick = time.perf_counter()
        try:
            results = synthesize_batch(tts, [r["text"] for r in part], seeds, max_frames)
        except Exception as error:
            if not is_oom(error) or size <= 1:
                raise
            size = max(1, size // 2)
            print(f"Teacher OOM: retry batch_size={size}", flush=True)
            continue
        yield part, results, time.perf_counter() - tick, size
        offset += len(part)


def generate(root, batch_size=2, device="cuda", threads=2, sync_dir=None, sync_every=128):
    if batch_size < 1 or sync_every < 1:
        raise ValueError("Batch size and sync interval must be positive")
    root = Path(root).resolve()
    rows = data.read_jsonl(root / "texts.jsonl")
    if not rows or len(data.read_jsonl(root / "benchmark_manifest.jsonl")) != 137:
        raise ValueError("Prepare benchmark and texts before generation")
    smoke = json.loads((root / "smoke.json").read_text())
    if not smoke["passed_waveform_sanity"]:
        raise ValueError("Teacher smoke check is required")
    config = {"teacher_revision": data.TEACHER_REV, "voice": "maichi", "sampling": SAMPLING,
              "text_sha256": hashlib.sha256((root / "texts.jsonl").read_bytes()).hexdigest(),
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_path = root / "batch_generation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Teacher source/text/sampling changed; use the original source or a new data root")
    data.atomic_json(config_path, config)
    accepted = {r["id"]: r for r in data.read_jsonl(root / "corpus.jsonl")}
    rejected = {r["id"]: r for r in data.read_jsonl(root / "rejected.jsonl")}
    for row in rows:
        uid = row["id"]
        sidecar = root / "wavs" / f"{uid}.json"
        path = sidecar.with_suffix(".wav")
        if path.exists() and sidecar.exists():
            saved = json.loads(sidecar.read_text())
            if saved["text"] != row["text"] or saved["generation_config"] != config:
                raise ValueError(f"Stale teacher cache: {uid}")
            saved["wav"] = str(path)
            accepted[uid] = saved
        elif uid in accepted:
            del accepted[uid]
    tts = data.make_teacher(root, device, threads)
    root.joinpath("wavs").mkdir(exist_ok=True)
    progress = root / "generation_batches.jsonl"

    def checkpoint():
        data.write_jsonl(root / "corpus.jsonl", [accepted[r["id"]] for r in rows if r["id"] in accepted])
        data.write_jsonl(root / "rejected.jsonl", list(rejected.values()))
        data.atomic_json(root / "corpus_summary.json", {"accepted": len(accepted), "rejected": len(rejected),
                         "requested": len(rows), "hours": sum(r["duration"] for r in accepted.values()) / 3600,
                         "generation_complete": len(accepted) + len(rejected) == len(rows)})

    # Bucket only within a checkpoint block, keeping completed archive blocks stable.
    try:
        for first in range(0, len(rows), sync_every):
            pending = [r for r in rows[first:first+sync_every] if r["id"] not in accepted and r["id"] not in rejected]
            pending.sort(key=lambda r: len(r["text"]))
            for part, outputs, elapsed, actual_batch in run_adaptive(tts, pending, batch_size):
                for row, output in zip(part, outputs):
                    try:
                        if output["hit_frame_limit"]:
                            raise ValueError("Teacher did not emit EOA before the frame ceiling")
                        wave = output["audio"]
                        metrics = data.sanity(wave, tts.sample_rate)
                        rate = len(row["text"]) / metrics["duration"]
                        if not 1 <= metrics["duration"] <= 18 or not 3 <= rate <= 45 or metrics["clipped_fraction"] > 0.01:
                            raise ValueError(f"Duration/rate/clipping filter: {metrics}")
                        path = root / "wavs" / f"{row['id']}.wav"
                        y = resample_poly(wave, data.SR, tts.sample_rate).astype(np.float32)
                        sf.write(str(path) + ".tmp.wav", y, data.SR, subtype="PCM_16")
                        Path(str(path) + ".tmp.wav").replace(path)
                        saved = {**row, **metrics, "wav": str(path), "sample_rate": data.SR,
                                 "teacher_revision": data.TEACHER_REV, "generation_config": config,
                                 "seed": int(row["id"][:8], 16), "batch_size": len(part)}
                        data.atomic_json(path.with_suffix(".json"), saved)
                        accepted[row["id"]] = saved
                    except (ValueError, FloatingPointError) as error:
                        rejected[row["id"]] = {"id": row["id"], "error": str(error)}
                with progress.open("a") as handle:
                    handle.write(json.dumps({"items": len(part), "seconds": elapsed, "items_per_second": len(part)/elapsed,
                                             "requested_batch": batch_size, "actual_batch": len(part), "capacity": actual_batch}) + "\n")
                if (len(accepted) + len(rejected)) % 32 < len(part):
                    print(f"Teacher accepted={len(accepted)} rejected={len(rejected)}/{len(rows)} batch={len(part)} seconds={elapsed:.2f}", flush=True)
            checkpoint()
            if sync_dir:
                from maichi_handoff import publish
                publish(root, sync_dir, complete=False)
    except KeyboardInterrupt:
        checkpoint()
        if sync_dir:
            from maichi_handoff import publish
            publish(root, sync_dir, complete=False)
        raise
    checkpoint()
    print("Corpus generation complete; extract features and publish a COMPLETE bundle before training.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sync-dir")
    parser.add_argument("--sync-every", type=int, default=128)
    args = parser.parse_args()
    generate(args.root, args.batch_size, args.device, args.threads, args.sync_dir, args.sync_every)


if __name__ == "__main__":
    main()
