"""CPU checks for the strict <10M flow student; no quality claims.

Run: python test_maichi_flow.py
"""
from pathlib import Path
import json
import tempfile

import torch

import maichi_flow as flow
import maichi_pipeline as data


def main():
    torch.set_num_threads(2)
    data.seed_all(42)
    budget = {}
    for size in [160, 256, 512]:
        model = flow.FlowStudent(size)
        info = flow.assert_budget(model)
        assert info["deployment_parameters"] == sum(p.numel() for p in model.parameters())
        assert info["acoustic_parameters"] + info["vocoder_parameters"] == info["deployment_parameters"]
        assert sum(t.numel() for t in model.state_dict().values()) < 10_000_000
        budget[size] = info
    rejected = False
    try:
        flow.FlowStudent(10000)
    except ValueError:
        rejected = True
    assert rejected, "An over-budget model must be rejected, not silently exported"
    vocab = ["<blank>", "<unk>"] + list(" abcdefghijklmnopqrstuvwxyzáàảãạđ")
    model = flow.FlowStudent(len(vocab))
    ids = data.encode("ma má mà mả mã mạ", vocab)[None]
    lengths = torch.tensor([ids.shape[1]])
    frames = max(96, ids.shape[1] + 32)
    target = torch.randn(1, data.MELS, frames)
    frame_lengths = torch.tensor([frames])
    f0, vuv = torch.zeros(1, 1, frames), torch.ones(1, 1, frames)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        loss, metrics, alignment = model.objective(ids, lengths, target, frame_lengths, f0, vuv)
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
    assert model.velocity.output.weight.grad is not None
    assert model.velocity.blocks[0].modulation[-1].weight.grad is not None
    assert torch.isfinite(model.velocity.blocks[0].modulation[-1].weight.grad).all()
    path = alignment[0].argmax(-1)
    assert torch.equal(alignment[0].sum(-1), torch.ones(frames))
    assert path[0] == 0 and path[-1] == ids.shape[1] - 1
    assert (path[1:] >= path[:-1]).all() and ((path[1:] - path[:-1]) <= 1).all()
    assert data.text_key("má") != data.text_key("mà")
    durations = torch.full_like(ids, 3)
    model.eval()
    data.seed_all(100)
    mel, mel_lengths = model.mel(ids, lengths, steps=4, durations=durations)
    assert mel.shape[-1] == int(durations.sum())
    generated = model.vocoder(mel)
    assert generated.shape[-1] == mel.shape[-1] * data.HOP
    assert torch.isfinite(generated).all()
    # Padding must not change the valid portion of the text states.
    with torch.no_grad():
        padded = torch.cat([ids, torch.zeros(1, 7, dtype=torch.long)], 1)
        a = model.text(ids, lengths)[0]
        b = model.text(padded, lengths)[0][:, :, :ids.shape[1]]
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value, _, _ = model.objective(ids, lengths, target, frame_lengths, f0, vuv)
        assert torch.isfinite(value)
    # Export contains BOTH acoustic model and vocoder; exact seeded reload.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = root / "flow_runs/test"
        folder.mkdir(parents=True)
        data.save_checkpoint(folder / "best.pt", {"architecture": flow.ARCHITECTURE, "vocab": vocab,
            "model_config": model.config, "ema": model.state_dict(), "training_config": {"phase": "acoustic"}, "step": 0})
        path = flow.export(root, "test")
        reloaded, restored_vocab = flow.load_deployment(path)
        assert restored_vocab == vocab
        data.seed_all(300)
        expected = model.speak(ids, lengths, steps=4)
        data.seed_all(300)
        actual = reloaded.speak(ids, lengths, steps=4)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert any(k.startswith("vocoder.") for k in torch.load(path, weights_only=False)["state"])
    print(json.dumps({"status": "passed", "checks": ["strict full-package parameter budget", "over-budget rejection",
        "MAS coverage", "tone-bearing text", "DiT gradient flow", "padding invariance", "CPU BF16 objective",
        "flow integration", "waveform length", "exact deployment reload including vocoder"], "budgets": budget}))


if __name__ == "__main__":
    main()
