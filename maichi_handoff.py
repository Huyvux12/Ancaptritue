"""Checksummed local-SSD <-> Google Drive data/checkpoint handoff.

The Drive path is a normal mounted directory. No cloud credentials are handled.
Data tar shards are immutable; bundle.json is committed last. Training backups
retain objects referenced by the latest TWO successful backup manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time

import maichi_pipeline as data

FORMAT = "maichi_data_shards_v3"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_copy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".uploading")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


def safe_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"Unsafe bundle path: {name}")
    return p


def normalized_member(item):
    item.mtime = 0
    item.uid = item.gid = 0
    item.uname = item.gname = ""
    item.mode = 0o644
    item.pax_headers = {}
    return item


def dataset_key(bundle):
    return bundle.get("dataset_id", bundle["bundle_id"])


def add_bytes(tar, name, content):
    item = tarfile.TarInfo(name)
    item.size = len(content)
    tar.addfile(item, io.BytesIO(content))


def cached_hash(path, root, cache):
    key = str(path.relative_to(root))
    stat = path.stat()
    stamp = [stat.st_size, stat.st_mtime_ns]
    if key not in cache or cache[key]["stamp"] != stamp:
        cache[key] = {"stamp": stamp, "sha256": sha256(path)}
    return cache[key]["sha256"]


def publish(root, destination, complete=False, shard_items=128):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or root in destination.parents:
        raise ValueError("Keep the Drive bundle outside the local corpus tree")
    rows = data.read_jsonl(root / "corpus.jsonl")
    if complete:
        summary = json.loads((root / "corpus_summary.json").read_text())
        if not summary.get("generation_complete"):
            raise ValueError("Generation is incomplete")
        if not rows or not (root / "vocab.json").exists():
            raise ValueError("Extract features/vocabulary before final publication")
        if any(not (root / "features" / f"{r['id']}.npz").exists() for r in rows):
            raise ValueError("Missing features; cannot publish a complete training bundle")
        if len(data.read_jsonl(root / "benchmark_manifest.jsonl")) != 137:
            raise ValueError("Missing full benchmark manifest")
    cache_file = root / ".handoff_hash_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    objects = destination / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    shards, asset_files = [], 0
    row_lookup = {r["id"]: r for r in rows}
    plan = data.read_jsonl(root / "texts.jsonl") or rows
    with tempfile.TemporaryDirectory(prefix="maichi-publish-", dir=root) as tmp:
        tmp = Path(tmp)
        for category, suffixes in [("wavs", [".wav", ".json"]), ("features", [".npz"])]:
            for start in range(0, len(plan), shard_items):
                files = []
                for planned in plan[start:start + shard_items]:
                    if planned["id"] not in row_lookup: continue
                    row = row_lookup[planned["id"]]
                    for suffix in suffixes:
                        path = root / category / (row["id"] + suffix)
                        if path.exists(): files.append(path)
                if not files: continue
                signatures = [(str(p.relative_to(root)), cached_hash(p, root, cache)) for p in files]
                content_id = data.digest(json.dumps(signatures, sort_keys=True))
                name = f"{category}-{content_id}.tar"
                target = objects / name
                receipt = objects / (name + ".json")
                if not target.exists() or not receipt.exists():
                    local = tmp / name
                    with tarfile.open(local, "w") as tar:
                        for file in files: tar.add(file, arcname=str(file.relative_to(root)), recursive=False, filter=normalized_member)
                    info = {"path": "objects/" + name, "sha256": sha256(local), "bytes": local.stat().st_size,
                            "category": category, "files": len(files)}
                    atomic_copy(local, target)
                    if sha256(target) != info["sha256"]: raise IOError("Drive shard copy failed checksum")
                    data.atomic_json(receipt, info)
                info = json.loads(receipt.read_text())
                shards.append(info); asset_files += len(files)
        # Small mutable files and evaluation artifacts get their own immutable snapshot.
        metadata = tmp / "metadata.tar"
        with tarfile.open(metadata, "w") as tar:
            for file in sorted(root.iterdir()):
                if file.is_file() and not file.name.startswith(".") and file.suffix in {".json", ".jsonl", ".txt", ".py"} and file.name != "corpus.jsonl":
                    tar.add(file, arcname=file.name, recursive=False, filter=normalized_member)
            portable = [{**r, "wav": f"wavs/{r['id']}.wav"} for r in rows]
            add_bytes(tar, "corpus.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in portable).encode())
            for folder in ["smoke", "batch_smoke", "benchmark_wavs"]:
                if (root / folder).exists():
                    for file in sorted((root / folder).rglob("*")):
                        if file.is_file(): tar.add(file, arcname=str(file.relative_to(root)), recursive=False, filter=normalized_member)
        digest = sha256(metadata)
        name = f"metadata-{digest}.tar"
        target = objects / name
        if not target.exists(): atomic_copy(metadata, target)
        if sha256(target) != digest: raise IOError("Metadata checksum mismatch")
        shards.append({"path": "objects/" + name, "sha256": digest, "bytes": metadata.stat().st_size, "category": "metadata"})
    revision = json.loads((root / "source_revision.json").read_text()) if (root / "source_revision.json").exists() else {}
    bundle = {"format": FORMAT, "complete": complete, "accepted": len(rows), "asset_files": asset_files,
              "source_commit": revision.get("commit"), "source_hashes": revision.get("files", {}),
              "teacher_revision": data.TEACHER_REV, "benchmark_revision": data.BENCH_REV,
              "sample_rate": data.SR, "shards": shards}
    # Stable training identity ignores audit timestamps and archive metadata.
    assets = [(str(p.relative_to(root)), cached_hash(p, root, cache))
              for category, suffix in [("wavs", ".wav"), ("features", ".npz")]
              for p in sorted((root / category).glob("*" + suffix))
              if p.stem in row_lookup]
    identity = {"rows": [{**r, "wav": f"wavs/{r['id']}.wav"} for r in rows], "assets": assets,
                "vocab": json.loads((root / "vocab.json").read_text()) if (root / "vocab.json").exists() else None,
                "source_hashes": bundle["source_hashes"], "teacher_revision": data.TEACHER_REV,
                "benchmark_manifest": sha256(root / "benchmark_manifest.jsonl") if (root / "benchmark_manifest.jsonl").exists() else None}
    bundle["dataset_id"] = data.digest(json.dumps(identity, sort_keys=True, ensure_ascii=False))
    bundle["bundle_id"] = data.digest(json.dumps(bundle, sort_keys=True))
    data.atomic_json(destination / "bundle.json", bundle)  # Commit only after ALL objects are durable.
    data.atomic_json(cache_file, cache)
    print(f"Published {'COMPLETE' if complete else 'partial'} bundle: {len(rows)} items, {len(shards)} shards -> {destination}", flush=True)
    return bundle


def load_bundle(source, require_complete=True):
    source = Path(source)
    bundle = json.loads((source / "bundle.json").read_text())
    if bundle["format"] != FORMAT or (require_complete and not bundle["complete"]):
        raise ValueError("Expected a COMPLETE v3 data bundle; finish notebook 01 first")
    expected = bundle["bundle_id"]
    unsigned = {k: v for k, v in bundle.items() if k != "bundle_id"}
    if data.digest(json.dumps(unsigned, sort_keys=True)) != expected:
        raise ValueError("Bundle manifest hash mismatch")
    return bundle


def restore(source, root, require_complete=True):
    source, root = Path(source).resolve(), Path(root).resolve()
    bundle = load_bundle(source, require_complete)
    marker = root / ".data_bundle.json"
    if (root / "flow_runs").exists() and not marker.exists():
        raise ValueError("Untracked training state exists; import data into a new directory")
    if (root / "flow_runs").exists() and marker.exists() and dataset_key(json.loads(marker.read_text())) != dataset_key(bundle):
        raise ValueError("Data bundle changed after training began; choose a new training directory")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="maichi-restore-", dir=root) as tmp:
        for entry in bundle["shards"]:
            relative = safe_path(entry["path"])
            local = Path(tmp) / relative.name
            shutil.copyfile(source / str(relative), local)
            if local.stat().st_size != entry["bytes"] or sha256(local) != entry["sha256"]:
                raise ValueError(f"Corrupt/incomplete Drive shard: {relative}")
            with tarfile.open(local) as tar:
                for member in tar:
                    name = safe_path(member.name)
                    if not member.isfile(): raise ValueError("Only regular files are allowed in a data bundle")
                    target = root / str(name)
                    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root.parent):
                        raise ValueError("Symlink extraction target is not allowed")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as inp, target.open("wb") as out: shutil.copyfileobj(inp, out)
            local.unlink()
    rows = data.read_jsonl(root / "corpus.jsonl")
    if len(rows) != bundle["accepted"]: raise ValueError("Corpus count mismatch")
    for row in rows:
        row["wav"] = str(root / "wavs" / f"{row['id']}.wav")
        if not Path(row["wav"]).exists(): raise FileNotFoundError(row["wav"])
        sidecar = root / "wavs" / f"{row['id']}.json"
        if sidecar.exists():
            saved = json.loads(sidecar.read_text()); saved["wav"] = row["wav"]
            data.atomic_json(sidecar, saved)
        if require_complete and not (root / "features" / f"{row['id']}.npz").exists():
            raise FileNotFoundError(f"Missing features: {row['id']}")
    data.write_jsonl(root / "corpus.jsonl", rows)
    data.atomic_json(marker, bundle)
    print(f"Restored {len(rows)} items to local SSD: {root}", flush=True)
    return bundle


def sync_training(root, destination):
    """Snapshot only atomically finished checkpoints; retain two backup versions."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    marker = json.loads((root / ".data_bundle.json").read_text())
    manifest_path = destination / "training_backup.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["bundle_id"] != dataset_key(marker):
        raise ValueError("Backup directory belongs to a different data bundle")
    files = list((root / "flow_runs").rglob("*")) if (root / "flow_runs").exists() else []
    files += [root / n for n in ["flow_mel_statistics.json", "training_source_revision.json", "student_parameter_budget.json"]]
    entries = []
    old = {x["path"]: x for x in previous["files"]} if previous else {}
    for file in sorted(files):
        if not file.is_file() or file.suffix not in {".pt", ".json", ".jsonl"}: continue
        relative = str(file.relative_to(root))
        stat = file.stat()
        stamp = [stat.st_size, stat.st_mtime_ns]
        if relative in old and old[relative].get("stamp") == stamp:
            entries.append(old[relative]); continue
        # Copy locally first; an atomic checkpoint replace during copy is safe,
        # but a changing JSONL is skipped and retried at the next backup.
        with tempfile.TemporaryDirectory(prefix="maichi-backup-", dir=root) as folder:
            snapshot = Path(folder) / file.name
            shutil.copyfile(file, snapshot)
            if [file.stat().st_size, file.stat().st_mtime_ns] != stamp:
                if relative in old: entries.append(old[relative])
                continue
            digest = sha256(snapshot)
            object_name = "objects/" + digest + file.suffix
            target = destination / object_name
            if not target.exists(): atomic_copy(snapshot, target)
            if sha256(target) != digest: raise IOError("Checkpoint backup checksum failed")
        entries.append({"path": relative, "object": object_name, "sha256": digest, "stamp": stamp})
    manifest = {"format": "maichi_training_backup_v3", "bundle_id": dataset_key(marker), "files": entries}
    if previous: data.atomic_json(destination / "training_backup_previous.json", previous)
    data.atomic_json(manifest_path, manifest)
    # Only delete unreferenced generated backup objects, never source/data files.
    keep = {x["object"] for x in entries} | {x["object"] for x in (previous or {}).get("files", [])}
    if (destination / "objects").exists():
        for file in (destination / "objects").iterdir():
            if "objects/" + file.name not in keep and not file.name.endswith(".uploading"): file.unlink()
    return len(entries)


def restore_training(source, root):
    source, root = Path(source), Path(root)
    path = source / "training_backup.json"
    if not path.exists(): return False
    manifest = json.loads(path.read_text())
    marker = json.loads((root / ".data_bundle.json").read_text())
    if manifest["bundle_id"] != dataset_key(marker): raise ValueError("Checkpoint/data bundle mismatch")
    if (root / "flow_runs").exists():
        print("Local training state already exists; preserving it. Use a new SSD directory to restore Drive state.")
        return False
    for entry in manifest["files"]:
        target = root / str(safe_path(entry["path"]))
        source_file = source / str(safe_path(entry["object"]))
        if sha256(source_file) != entry["sha256"]: raise ValueError("Corrupt checkpoint backup")
        atomic_copy(source_file, target)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["publish", "restore", "backup", "resume"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--drive-dir", required=True)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.stage == "publish": publish(args.root, args.drive_dir, args.complete)
    elif args.stage == "restore": restore(args.drive_dir, args.root, not args.allow_partial)
    elif args.stage == "backup": print(sync_training(args.root, args.drive_dir))
    elif args.stage == "resume": print(restore_training(args.drive_dir, args.root))


if __name__ == "__main__": main()
