"""Short CPU integration check on the bundled real ZeroTTS WAVs.

This tests training/checkpoint plumbing, not perceptual speech quality.
Run: python test_maichi_flow_real_audio.py [--root /new/test/directory]
"""
from pathlib import Path
import argparse
import json
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

import maichi_pipeline as data
import maichi_flow as flow


def check(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "corpus.jsonl").exists() or (root / "flow_runs").exists():
        raise ValueError("Use an empty test directory; existing corpora/runs are protected")
    rows = []
    for i, text in enumerate(data.SMOKE_TEXTS):
        source = Path(__file__).resolve().parent / "samples" / f"MaiChi_sample_{i + 1}.wav"
        wave, sample_rate = sf.read(source, dtype="float32")
        wave = resample_poly(wave, data.SR, sample_rate).astype(np.float32)
        path = root / f"{i}.wav"
        sf.write(path, wave, data.SR)
        rows.append({"id": str(i), "text": text, "wav": str(path), "split": "train" if i < 2 else "val"})
    data.write_jsonl(root / "corpus.jsonl", rows)
    data.features(root, workers=1)
    common = dict(batch_size=1, workers=0, device="cpu", precision="fp32", save_every=1,
                  eval_every=1, allow_small=True)
    flow.train(root, phase="vocoder", run="vocoder", steps=2, **common)
    flow.train(root, phase="acoustic", run="flow10m", steps=2, **common)
    flow.train(root, phase="acoustic", run="flow10m", steps=3, **common)
    model, vocab = flow.load_deployment(root / "flow_runs/flow10m/student_deploy.pt")
    ids = data.encode(data.SMOKE_TEXTS[0], vocab)[None]
    audio = model.speak(ids, torch.tensor([ids.shape[1]]), steps=4)
    assert torch.isfinite(audio).all()
    checkpoint = torch.load(root / "flow_runs/flow10m/last.pt", weights_only=False)
    assert checkpoint["step"] == 3
    assert checkpoint["optimizer"]["state"]
    result = {"vocoder_real_audio_training_steps": 2, "acoustic_real_audio_training_steps": 3,
              "resume_passed": True, "deployment_reload_passed": True,
              "parameters": flow.counts(model), "waveform_shape": list(audio.shape),
              "quality_claim": False, "device": "CPU"}
    data.atomic_json(root / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.root:
        check(args.root)
    else:
        with tempfile.TemporaryDirectory(prefix="maichi-flow-real-") as folder:
            check(folder)


if __name__ == "__main__":
    main()
