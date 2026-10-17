"""CPU regression checks for the custom student and deployment format.

Usage: python test_maichi_pipeline.py --vits-dir /path/to/pinned/vits
This checks implementation behavior, not speech quality.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import maichi_pipeline as p


def main(vits_dir):
    torch.set_num_threads(2)
    p.seed_all(42)
    vocab = ["<blank>", "<unk>"] + list(" abcdefghijklmnopqrstuvwxyzáàảãạđ")
    # NFC and tone-bearing characters must survive normalization/tokenization.
    assert p.text_key("má") != p.text_key("ma")
    assert p.normalized("ma\u0301") == "má"
    assert int((p.encode("má", vocab) == 1).sum()) == 0
    model = p.make_student(vits_dir, len(vocab), "tiny", True, 16)
    x = p.encode("ma má mà mả mã mạ", vocab)[None]
    xl = torch.tensor([x.shape[1]])
    frames = max(96, x.shape[1] + 20)
    spec = torch.rand(1, 513, frames)
    sl = torch.tensor([frames])
    f0 = torch.ones(1, 1, frames)
    vuv = torch.ones_like(f0)
    y, duration, attn, starts, mask, latent, pitch = model(x, xl, spec, sl, f0, vuv, 0.5)
    assert y.shape == (1, 1, 4096)
    assert torch.equal(attn[0, 0].sum(-1), torch.ones(frames))
    path = attn[0, 0].argmax(-1)
    assert path[0] == 0 and path[-1] == x.shape[1] - 1
    assert ((path[1:] - path[:-1]) >= 0).all() and ((path[1:] - path[:-1]) <= 1).all()
    loss = y.square().mean() + duration + pitch[0].square().mean() + pitch[1].square().mean()
    loss.backward()
    assert model.pitch_head[0].weight.grad is not None
    assert model.dec.conv_pre.weight.grad is not None
    assert torch.isfinite(model.pitch_head[0].weight.grad).all()
    counts = p.parameter_counts(model)
    assert counts["deployment_parameters"] + counts["training_only_posterior_parameters"] == counts["checkpoint_generator_parameters"]
    cfg = {"size": "tiny", "pitch": True, "segment_frames": 16}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "runs/test").mkdir(parents=True)
        p.atomic_json(tmp / "vocab.json", vocab)
        p.save_checkpoint(tmp / "runs/test/best.pt", {"g": model.state_dict(), "config": cfg, "step": 0})
        exported = p.export_student(tmp, vits_dir, "test")
        loaded, actual_vocab = p.load_deployment(exported, vits_dir)
        assert not hasattr(loaded, "enc_q") and actual_vocab == vocab
        with torch.no_grad():
            p.seed_all(200)
            reference = model.eval().speak(x, xl)
            p.seed_all(200)
            actual = loaded.speak(x, xl)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    baseline = p.make_student(vits_dir, len(vocab), "tiny", False, 16)
    out = baseline(x, xl, spec, sl, f0, vuv)
    assert out[-1] == (None, None)
    assert not any(k.startswith("pitch_") for k in baseline.state_dict())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, xl, spec, sl, f0, vuv)
        assert torch.isfinite(out[0]).all()
    print(json.dumps({"status": "passed", "checks": ["Vietnamese tone preservation", "monotonic alignment coverage",
        "gradient flow", "parameter accounting", "deployment exact reload", "pitch ablation", "BF16 forward"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vits-dir", required=True)
    main(parser.parse_args().vits_dir)
