"""Contract tests and optional real-weight/WAV integration for TITL Vietnamese.

Quick: python test_titl_vi.py
Real:  python test_titl_vi.py --real
Real integration is engineering evidence, not a quality benchmark.
"""
from pathlib import Path
import argparse
import json
import tempfile
import unicodedata
import unittest

import numpy as np
import torch

from titl_vi.model import make_vocab,encode,canonical,budget,pretrained
from titl_vi.ctc import durations,viterbi
from titl_vi.data import atomic_json,file_sha,write_rows
from titl_vi.storage import backup,restore


class Contracts(unittest.TestCase):
    def test_tones_numbers_and_unknowns(self):
        vocab=make_vocab({'ə':0,'a':1,' ':2})
        tone_ids=[int(encode(v,vocab)[1]) for v in ['a','à','á','ả','ã','ạ']]
        self.assertEqual(len(set(tone_ids)),6)
        self.assertTrue(torch.equal(encode('Tiếng Việt',vocab),encode(unicodedata.normalize('NFD','Tiếng Việt'),vocab)))
        self.assertFalse(any(c.isdigit() for c in canonical('năm 2026')))
        with self.assertRaises(ValueError):encode('Xin chào 中',vocab)
        with self.assertRaises(ValueError):encode('a'*401,vocab)

    def test_ctc_repeated_char_blank_and_clock(self):
        # Adjacent identical labels require a blank-separated CTC path.
        logits=np.full((6,96),-20.,np.float32)
        path=[95,1,95,1,2,95]
        logits[np.arange(6),path]=0
        labels=[1,1,2]
        states=viterbi(logits,np.array(labels,np.int64),95)
        self.assertEqual([path[i] for i in range(6) if states[i]%2],[1,1,2])
        d,p=durations(logits,labels,95,24)
        self.assertEqual(int(d.sum()),24);self.assertGreaterEqual(int(d.min()),1)
        self.assertEqual(len(d),len(labels)+2);self.assertAlmostEqual(p,1.)
        with self.assertRaises(ValueError):durations(logits,labels,95,4)
        # Two identical labels cannot both fit in two emissions.
        with self.assertRaises(ValueError):viterbi(logits[:2],np.array([1,1]),95)

    def test_snapshot_corruption_retention_and_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'root';root.mkdir();dest=base/'drive'
            atomic_json(root/'source_revision.json',{'commit':'test-fixture'})
            for i in range(3):
                (root/'value.txt').write_text(str(i));backup(root,dest)
            self.assertEqual(len(list(dest.glob('snapshot-*.tar'))),2)
            self.assertTrue(restore(dest,base/'restored'))
            self.assertEqual((base/'restored/value.txt').read_text(),'2')
            info=json.loads((dest/'latest.json').read_text())
            self.assertEqual(info['project_commit'],'test-fixture')
            (dest/info['file']).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):restore(dest,base/'bad')

    def test_failed_gate_blocks_export(self):
        from titl_vi.gate import export
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ck=root/'untrusted.pt';ck.write_bytes(b'fixture')
            atomic_json(root/'gates/final.json',{'passed':False})
            with self.assertRaises(ValueError):export(root,ck)
            # A passed report for a different checkpoint must still be rejected.
            atomic_json(root/'gates/final.json',{'passed':True,'checkpoint_sha256':'wrong'})
            with self.assertRaises(ValueError):export(root,ck)


def export_serialization_fixture(root,checkpoint):
    """Positive serialization contract with a labelled fake gate, never quality evidence."""
    import shutil
    from titl_vi.gate import export
    from titl_vi.train import data_fingerprint,load_checkpoint
    with tempfile.TemporaryDirectory() as temp:
        fixture=Path(temp)
        shutil.copytree(Path(root)/'features',fixture/'features')
        for name in ['manifest.jsonl','energy_statistics.json']:shutil.copyfile(Path(root)/name,fixture/name)
        atomic_json(fixture/'gates/final.json',{'passed':True,'checkpoint_sha256':file_sha(checkpoint),
                    'data_fingerprint':data_fingerprint(fixture),'fixture_only':True})
        path=export(fixture,checkpoint)
        model,ck=load_checkpoint(path)
        assert 'optimizer' not in ck and 'critic' not in ck and not ck['quality_verified']
        assert ck['budget']['total']==8_110_486 and budget(model)['total']==8_110_486


def real_integration(root):
    """Download actual weights; use three bundled teacher samples, not synthetic tones."""
    import shutil
    from titl_vi.bootstrap import bootstrap
    from titl_vi.data import prepare
    from titl_vi.train import train,load_checkpoint,Dataset,data_fingerprint
    from titl_vi.gate import infer
    from titl_vi.upstream.student_decoder import StudentDecoder
    from titl_vi.model import load_decoder,MODEL_ID,MODEL_REVISION
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    model,metadata=bootstrap(root,'cpu')
    assert budget(model)['total']==8_110_486
    expanded,original,_=pretrained(root,'cpu')
    a,b=original.text.state_dict(),expanded.text.state_dict()
    for key in a:
        torch.testing.assert_close(a[key],b[key][:178] if a[key].shape!=b[key].shape else b[key],rtol=0,atol=0)
    del a,b,expanded
    # Compare native Kokoro TorchSTFT with the upstream fixed export STFT.
    decoder=StudentDecoder('A').eval()
    decoder.load_state_dict(load_file(hf_hub_download(MODEL_ID,'af_heart/decoder.safetensors',revision=MODEL_REVISION)),strict=True)
    ids=torch.tensor([json.loads(Path('titl_vi/calibration_en.json').read_text())[0]['ids']])
    from titl_vi.upstream.student import alignment
    with torch.no_grad():
        lengths=torch.tensor([ids.shape[1]])
        d=original.text(ids,lengths)[0][0].round().clamp(1).long()
        _,f,n,features=original.text(ids,lengths,alignment(d)[None])
        expanded_features=features@alignment(d)[None]
        torch.manual_seed(123);native=decoder(expanded_features,f,n)
        torch.manual_seed(123);exported=original.decoder(expanded_features,f,n)
        relative=float((native-exported).norm()/native.norm())
        assert relative<.01,relative
    raw=root.parent/(root.name+'_raw');raw.mkdir(exist_ok=True)
    texts=['Xin chào, mình là Mai Chi. Hôm nay chúng ta thử giọng nói tiếng Việt.',
           'Ma, má, mà, mả, mã, mạ. Bé đi học và mẹ ở nhà.',
           'Sáng nay tôi dùng Python để kiểm tra mô hình trí tuệ nhân tạo.']
    rows=[{'id':str(i),'text':text,'wav':str(Path(f'samples/MaiChi_sample_{i+1}.wav').resolve()),'split':'train' if i<2 else 'val'} for i,text in enumerate(texts)]
    assert all(Path(r['wav']).exists() for r in rows)
    write_rows(raw/'corpus.jsonl',rows)
    prepared=prepare(raw,root,metadata,'cpu')
    assert len(prepared)==3
    before=data_fingerprint(root)
    p=train(root,'bridge_test','bridge',steps=2,batch_size=1,overfit=2,device='cpu',workers=0,save_every=1,eval_every=1)
    q=train(root,'joint_test','joint',steps=2,batch_size=1,overfit=2,init=p,device='cpu',workers=0,save_every=1,eval_every=1)
    q=train(root,'joint_test','joint',steps=3,batch_size=1,overfit=2,device='cpu',workers=0,save_every=1,eval_every=1)
    ck=torch.load(root/'runs/joint_test/last.pt',map_location='cpu',weights_only=False)
    assert ck['step']==3 and ck['optimizer']['state']
    assert all(int(s['step'])==3 for s in ck['optimizer']['state'].values())
    audio=infer(q,'Xin chào, mình là Mai Chi.',root/'unverified.wav','cpu')
    assert np.isfinite(audio).all()
    export_serialization_fixture(root,q)
    # Relocate prepared state and raw paths. Fingerprint must be portable.
    portable=root.parent/(root.name+'_portable');shutil.copytree(root,portable,dirs_exist_ok=True)
    relocated=[]
    for r in prepared:
        r={**r,'wav':str(Path(r['wav']).resolve()),'feature':str(portable/'features'/f"{r['id']}.npz")}
        relocated.append(r)
    write_rows(portable/'manifest.jsonl',relocated)
    assert data_fingerprint(portable)==before
    assert len(Dataset(portable,'train'))==2
    train(portable,'joint_test','joint',steps=4,batch_size=1,overfit=2,device='cpu',workers=0,save_every=1,eval_every=1)
    moved=torch.load(portable/'runs/joint_test/last.pt',map_location='cpu',weights_only=False)
    assert moved['step']==4
    try:
        train(portable,'joint_test','joint',steps=5,batch_size=2,overfit=2,device='cpu',workers=0)
    except ValueError as error:assert 'mismatch' in str(error)
    else:raise AssertionError('Changed resume configuration was accepted')
    train(root,'gan_test','gan',steps=1,batch_size=1,overfit=2,init=q,device='cpu',workers=0,save_every=1,eval_every=1)
    gan=torch.load(root/'runs/gan_test/last.pt',map_location='cpu',weights_only=False)
    assert gan['critic'] and gan['critic_optimizer']['state']
    # Cached alignment must not silently retain an earlier transcript.
    changed=[dict(r) for r in rows];changed[0]['spoken_text']='ma má mà mả mã mạ'
    write_rows(raw/'corpus.jsonl',changed)
    kept=prepare(raw,root,metadata,'cpu')
    assert '0' not in {r['id'] for r in kept}
    rejects=[json.loads(line) for line in (root/'rejected.jsonl').read_text().splitlines()]
    assert any('Transcript changed' in r['reason'] for r in rejects)
    print(json.dumps({'engineering_checks':'passed','budget':budget(model),'native_export_stft_relative_l2':relative,
        'prepared':3,'bridge_steps':2,'joint_steps_then_resume':[2,3,4],'gan_steps':1,
        'resume_mismatch_rejected':True,'changed_cached_transcript_rejected':True,'quality_verified':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--real',action='store_true');p.add_argument('--root')
    a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():raise SystemExit(1)
    if a.real:
        if a.root:real_integration(a.root)
        else:
            with tempfile.TemporaryDirectory() as temp:real_integration(Path(temp)/'prepared')
