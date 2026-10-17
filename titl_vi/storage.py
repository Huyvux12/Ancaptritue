"""Atomic v4 prepared-data/training snapshots to a mounted Drive directory.

Two successful snapshots retained. Raw corpus/teacher weights are not duplicated.
"""
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from .data import atomic_json,file_sha


def backup(root,destination):
    root,destination=Path(root).resolve(),Path(destination).resolve()
    if root==destination or root in destination.parents:raise ValueError('Keep Drive backups outside the training root')
    destination.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'snapshot.tar'
        with tarfile.open(path,'w') as archive:
            for file in sorted(root.rglob('*')):
                if not file.is_file() or file.is_symlink() or file.name.endswith(('.tmp.pt','.tmp.npz','.tmp')):continue
                archive.add(file,arcname=str(file.relative_to(root)),recursive=False)
        digest=file_sha(path);name='snapshot-'+digest+'.tar'
        target=destination/name
        if not target.exists():
            pending=destination/(name+'.uploading');shutil.copyfile(path,pending)
            if file_sha(pending)!=digest:raise IOError('Backup checksum failed')
            pending.replace(target)
    latest=destination/'latest.json';previous=json.loads(latest.read_text()) if latest.exists() else None
    if previous:atomic_json(destination/'previous.json',previous)
    revision=json.loads((root/'source_revision.json').read_text()) if (root/'source_revision.json').exists() else {}
    now={'format':'titl_vi_snapshot_v1','file':name,'sha256':digest,'project_commit':revision.get('commit')}
    atomic_json(latest,now)
    keep={name,(previous or {}).get('file')}
    for old in destination.glob('snapshot-*.tar'):
        if old.name not in keep:old.unlink()
    print('Drive snapshot:',target,flush=True)


def restore(source,root):
    import maichi_handoff as hh
    source,root=Path(source),Path(root)
    if not (source/'latest.json').exists():return False
    if (root/'runs').exists():
        print('Preserving existing local runs; restore into a new root to recover Drive state')
        return False
    info=json.loads((source/'latest.json').read_text());path=source/str(hh.safe_path(info['file']))
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local=Path(tmp)/'snapshot.tar';shutil.copyfile(path,local)
        if file_sha(local)!=info['sha256']:raise ValueError('Corrupt Drive snapshot')
        with tarfile.open(local) as archive:
            for member in archive:
                if not member.isfile():raise ValueError('Snapshot contains a non-regular file')
                target=root/str(hh.safe_path(member.name))
                if target.is_symlink() or any(p.is_symlink() for p in target.parents):raise ValueError('Symlink destination')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
    return True
