"""Batch state/RNG/OOM tests; --real also loads the pinned teacher.

Real checks are execution/equivalence checks, not a perceptual quality score.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

import maichi_pipeline as data
import maichi_teacher_batch as batch


class TextSession:
    def run(self, _, inputs):
        return []


class FakePrefix:
    def __init__(self): self.calls = []
    def run(self, _, x):
        self.calls.append(x)
        b = len(x["new_valid"])
        return np.zeros((b, 1, 4), np.float32), x["packed_kv"], x["new_valid"]


class FakeCodec:
    def run(self, _, x):
        lengths = x["audio_code_lengths"] * 8
        assert len(lengths) == x["audio_codes"].shape[0]
        return np.ones((len(lengths), 2, int(lengths.max())), np.float32), lengths


class FakeTeacher:
    num_codebooks, codebook_size, d_model = 2, 8, 4
    def __init__(self):
        self.prefix_step_sess = FakePrefix()
        self.codec = SimpleNamespace(_decode_full_sess=FakeCodec())
    def tokenizer(self, text): return np.arange(len(text) + 1)
    def resolve_voice(self, _): return np.ones((1, 2, 4), np.float32)
    def _prefix_step_init(self, ids, lengths, voice):
        assert list(lengths) == [2, 5]
        assert ids.shape == (2, 5) and voice.shape[0] == 2
        return np.zeros((2, 4)), np.zeros((1, 2, 2, 1, 0, 4)), np.zeros((2, 0)), None, None


def checks():
    t = FakeTeacher()
    calls = [0, 0]
    seen_ids = {}
    def local(tts, h, seen, rng, forbid):
        lane = seen_ids.setdefault(id(seen), len(seen_ids))
        calls[lane] += 1
        return calls[lane] >= [5, 7][lane], np.array([1, 2])
    with patch.object(batch, "local_frame", local):
        result = batch.synthesize_batch(t, ["a", "abcd"], [1, 2])
    assert calls == [5, 7]
    assert [r["frames"] for r in result] == [5, 7]
    assert all(r["ended"] for r in result)
    assert any(not call["new_valid"][0, 0] and call["new_valid"][1, 0] for call in t.prefix_step_sess.calls)
    rows = [{"id": f"{i:020x}", "text": str(i)} for i in range(4)]
    sizes = []
    def synth(tts, texts, seeds, maximum):
        sizes.append(len(texts))
        if len(texts) > 2: raise RuntimeError("CUDA out of memory")
        return [{"ended": True} for _ in texts]
    with patch.object(batch, "synthesize_batch", synth), contextlib.redirect_stdout(io.StringIO()):
        output = list(batch.run_adaptive(None, rows, 4))
    assert sizes == [4, 2, 2] and sum(len(x[0]) for x in output) == 4
    assert not batch.is_oom(RuntimeError("Unexpected input shape"))
    print("Passed: batch shapes, lane-specific EOA/lengths, finished-lane masks and memory-only adaptive retry")


def real(device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tts = data.make_teacher(output_dir, device, threads=2)
    texts, seeds = data.SMOKE_TEXTS[:2], [100, 101]
    np.random.seed(seeds[0])
    reference = tts.synthesize(texts[0], voice="maichi", max_frames=250)[0]
    one = batch.synthesize_batch(tts, texts[:1], seeds[:1])[0]
    np.testing.assert_allclose(one["audio"], reference, atol=2e-5, rtol=2e-5)
    outputs = batch.synthesize_batch(tts, texts, seeds)
    records = []
    for i, out in enumerate(outputs):
        assert out["ended"] and not out["hit_frame_limit"]
        # The same codes decoded alone must match the padded-batch codec result.
        single_decode = tts.codec.decode(out["codes"].T[None])[0]
        np.testing.assert_allclose(out["audio"], single_decode, atol=2e-4, rtol=2e-4)
        info = data.sanity(out["audio"], tts.sample_rate)
        filename = f"batch_{i}.wav"
        sf.write(output_dir / filename, out["audio"], tts.sample_rate)
        records.append({"text": texts[i], "file": filename, "frames": out["frames"], "seed": seeds[i], **info})
    del tts  # Release the smoke-test GPU sessions before loading the corpus writer.
    # Exercise the actual resumable corpus writer, not only the tensor adapter.
    corpus_root = output_dir / "corpus_fixture"
    corpus_root.mkdir(exist_ok=True)
    fixture_rows = [{"id": data.digest(text)[:20], "text": text, "split": "train"} for text in texts]
    data.write_jsonl(corpus_root / "texts.jsonl", fixture_rows)
    data.write_jsonl(corpus_root / "benchmark_manifest.jsonl", [{"id": str(i), "text": "count-only test fixture"} for i in range(137)])
    data.atomic_json(corpus_root / "smoke.json", {"passed_waveform_sanity": True})
    batch.generate(corpus_root, batch_size=2, device=device, threads=2)
    before = (corpus_root / "generation_batches.jsonl").read_bytes()
    batch.generate(corpus_root, batch_size=2, device=device, threads=2)
    assert before == (corpus_root / "generation_batches.jsonl").read_bytes(), "Cached rows must not be regenerated"
    assert len(data.read_jsonl(corpus_root / "corpus.jsonl")) == 2
    data.write_jsonl(output_dir / "results.jsonl", records)
    data.atomic_json(output_dir / "verification.json", {"real_teacher_batch": 2, "device": device,
        "batch_one_upstream_equivalence": True, "padded_codec_matches_individual_decode": True,
        "all_utterances_ended": True, "batched_corpus_resume": True, "quality_claim": False, "records": records})
    print(json.dumps({"status": "passed", "real_teacher_batch": 2, "device": device, "quality_claim": False}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    checks()
    if args.real:
        if args.output_dir: real(args.device, args.output_dir)
        else:
            with tempfile.TemporaryDirectory() as directory: real(args.device, directory)


if __name__ == "__main__": main()
