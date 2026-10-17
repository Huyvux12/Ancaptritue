"""Portable data/backup regression checks; --real-training also optimizes on WAVs.

Benchmark rows here are count-only TEST FIXTURES, not quality measurements.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

import maichi_pipeline as data
import maichi_handoff as transfer
import maichi_flow as flow
from maichi_train_runner import run_training


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(data.SMOKE_TEXTS):
        uid = f"{i:020x}"
        y, sr = sf.read(Path(__file__).parent / "samples" / f"MaiChi_sample_{i+1}.wav", dtype="float32")
        path = root / "wavs" / f"{uid}.wav"
        path.parent.mkdir(exist_ok=True)
        sf.write(path, resample_poly(y, data.SR, sr), data.SR)
        row = {"id": uid, "text": text, "wav": str(path.resolve()), "split": "train" if i < 2 else "val"}
        data.atomic_json(path.with_suffix(".json"), row)
        rows.append(row)
    data.write_jsonl(root / "texts.jsonl", rows)
    data.write_jsonl(root / "corpus.jsonl", rows)
    data.write_jsonl(root / "benchmark_manifest.jsonl", [{"id": f"fixture-{i}", "text": "count-only fixture"} for i in range(137)])
    data.atomic_json(root / "corpus_summary.json", {"generation_complete": True})
    data.atomic_json(root / "source_revision.json", {"commit": "test-fixture", "files": {}})
    data.features(root, workers=1)


def checks(base, real_training=False):
    original, drive, restored = base / "T4", base / "Drive/dataset", base / "RTX_A"
    fixture(original)
    transfer.publish(original, drive, complete=False, shard_items=2)
    try: transfer.restore(drive, restored)
    except ValueError: pass
    else: raise AssertionError("Training must reject partial data")
    published = transfer.publish(original, drive, complete=True, shard_items=2)
    transfer.restore(drive, restored)
    untracked = base / "untracked_training"
    (untracked / "flow_runs").mkdir(parents=True)
    try: transfer.restore(drive, untracked)
    except ValueError: pass
    else: raise AssertionError("Untracked training directory must be preserved")
    assert flow.corpus_fingerprint(original) == flow.corpus_fingerprint(restored)
    for r in data.read_jsonl(restored / "corpus.jsonl"):
        assert Path(r["wav"]).parent == restored / "wavs"
        relative = Path("wavs") / f"{r['id']}.wav"
        assert transfer.sha256(original / relative) == transfer.sha256(restored / relative)
    assert transfer.publish(original, drive, complete=True, shard_items=2)["bundle_id"] == published["bundle_id"]
    assert (drive / "bundle.json").exists()
    # Audit/log updates must not change training identity or invalidate backups.
    data.atomic_json(original / "audit_note.json", {"note": "new audit, unchanged training corpus"})
    updated = transfer.publish(original, drive, complete=True, shard_items=2)
    assert updated["dataset_id"] == published["dataset_id"]
    published = updated
    # Corruption must fail before the archive is extracted.
    obj = drive / published["shards"][0]["path"]
    saved = obj.read_bytes(); obj.write_bytes(b"X" + saved[1:])
    try: transfer.restore(drive, base / "bad")
    except ValueError: pass
    else: raise AssertionError("Corrupt archive accepted")
    obj.write_bytes(saved)
    try: transfer.safe_path("../../outside")
    except ValueError: pass
    else: raise AssertionError("Traversal accepted")
    # Backup retention and recovery use independent objects, not live source files.
    dummy = base / "backup_fixture"
    dummy.mkdir(); data.atomic_json(dummy / ".data_bundle.json", published)
    backup = base / "Drive/dummy_training"
    checkpoint = dummy / "flow_runs/check/last.pt"
    checkpoint.parent.mkdir(parents=True)
    for i in range(3):
        data.save_checkpoint(checkpoint, {"step": i})
        transfer.sync_training(dummy, backup)
    assert len(list((backup / "objects").glob("*.pt"))) == 2
    recovery = base / "recovery"
    recovery.mkdir(); data.atomic_json(recovery / ".data_bundle.json", published)
    assert transfer.restore_training(backup, recovery)
    assert torch.load(recovery / "flow_runs/check/last.pt", weights_only=False)["step"] == 2
    status = {"portable_data_checksums_and_paths": True, "canonical_corpus_hash": True,
              "partial_bundle_rejected": True, "corrupt_shard_rejected": True,
              "idempotent_publish": True, "two_backup_snapshots_retained": True,
              "checkpoint_restore": True, "quality_claim": False}
    if real_training:
        training_backup = base / "Drive/real_training"
        common = dict(batch_size=1, workers=0, device="cpu", precision="fp32",
                      save_every=1, eval_every=1, allow_small=True)
        run_training(restored, training_backup, phase="vocoder", run="vocoder", steps=2, **common)
        run_training(restored, training_backup, phase="acoustic", run="flow10m", steps=2, **common)
        other_machine = base / "RTX_B_different_path"
        transfer.restore(drive, other_machine)
        assert transfer.restore_training(training_backup, other_machine)
        run_training(other_machine, training_backup, phase="acoustic", run="flow10m", steps=3, **common)
        ck = torch.load(other_machine / "flow_runs/flow10m/last.pt", weights_only=False)
        assert ck["step"] == 3 and ck["optimizer"]["state"]
        model, vocab = flow.load_deployment(other_machine / "flow_runs/flow10m/student_deploy.pt")
        ids = data.encode(data.SMOKE_TEXTS[0], vocab)[None]
        wave = model.speak(ids, torch.tensor([ids.shape[1]]), steps=4)
        assert torch.isfinite(wave).all()
        status.update({"real_vocoder_steps": 2, "real_acoustic_steps_before_transfer": 2,
                       "real_acoustic_steps_after_transfer_resume": 3, "optimizer_resume_across_different_roots": True,
                       "deployment_parameters": flow.counts(model)["deployment_parameters"]})
    data.atomic_json(base / "verification.json", status)
    print(json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-training", action="store_true")
    p.add_argument("--root", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.root:
        if args.root.exists(): raise ValueError("Use a new test directory")
        args.root.mkdir(parents=True); checks(args.root.resolve(), args.real_training)
    else:
        with tempfile.TemporaryDirectory(prefix="maichi-handoff-") as temp:
            checks(Path(temp), args.real_training)


if __name__ == "__main__": main()
