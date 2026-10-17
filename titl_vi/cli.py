"""Vietnamese TITL 8M workflow CLI. Full training is quality-gated."""
from pathlib import Path
import argparse,json
from .data import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['bootstrap','import','prepare','train','diagnostics','gate','export','speak','backup','restore','benchmark'])
    p.add_argument('--root',required=True)
    p.add_argument('--raw-root')
    p.add_argument('--bundle-dir')
    p.add_argument('--backup-dir')
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--run',default='smoke_bridge')
    p.add_argument('--phase',default='bridge',choices=['bridge','joint','gan'])
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--overfit',type=int,default=0)
    p.add_argument('--init')
    p.add_argument('--checkpoint')
    p.add_argument('--name',default='overfit')
    p.add_argument('--split',default='val',choices=['train','val','test'])
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--text',default='Xin chào, mình là Mai Chi.')
    p.add_argument('--output',default='sample.wav')
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    if a.stage=='bootstrap':
        from .bootstrap import bootstrap
        bootstrap(root,a.device)
    elif a.stage=='import':
        from .data import restore_raw
        restore_raw(a.bundle_dir,a.raw_root)
    elif a.stage=='prepare':
        from .data import prepare
        metadata=json.loads((root/'pretrained_metadata.json').read_text())
        prepare(a.raw_root,root,metadata,a.device)
    elif a.stage=='train':
        from .train import train
        train(root,a.run,a.phase,a.steps,a.batch_size,a.overfit,a.init,a.device,a.workers,backup_dir=a.backup_dir)
    elif a.stage in ['diagnostics','gate']:
        from .gate import diagnostics,gate
        func=gate if a.stage=='gate' else diagnostics
        func(root,a.checkpoint,a.name,a.split,a.count,a.device,a.overfit)
    elif a.stage=='export':
        from .gate import export
        print(export(root,a.checkpoint))
    elif a.stage=='speak':
        from .gate import infer
        infer(a.checkpoint,a.text,a.output,a.device)
    elif a.stage=='benchmark':
        from .benchmark import benchmark
        benchmark(root,a.raw_root,a.checkpoint,a.device)
    elif a.stage=='backup':
        from .storage import backup
        backup(root,a.backup_dir)
    elif a.stage=='restore':
        from .storage import restore
        restore(a.backup_dir,root)


if __name__=='__main__':main()
