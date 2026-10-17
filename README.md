# Mai Chi distillation: T4 data → Drive → RTX 6000 training

Two separate clone-and-run notebooks. The complete student remains **strictly
below 10,000,000 parameters, including its waveform vocoder**.

| Order | Notebook | Runtime | Work |
| --- | --- | --- | --- |
| 1 | [01 — Data on T4](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/01_MaiChi_Data_T4.ipynb) | T4 | Batch teacher generation, audit/features, resumable Drive shards |
| 2 | [02 — Train on RTX 6000](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/02_MaiChi_Train_RTX6000.ipynb) | RTX PRO 6000 96 GB | Import to SSD, vocoder/acoustic training, export/evaluation |

Run **01 first**, wait for `READY FOR NOTEBOOK 02`, then run **02** with the same
`MODE` and `DRIVE_PARENT`. The training notebook does not regenerate the corpus
or load ZeroTTS teacher weights. It automatically clones the data producer's
recorded commit. Existing old notebook links now display these two links.

## Settings and storage

- Start with `MODE="pilot"`: 512 texts, then 2,000 vocoder and 4,000 acoustic steps.
  This is an execution check, not a quality demonstration.
- Full mode: 20,000 texts, 100,000 vocoder steps and 200,000 acoustic steps.
  These are starting budgets, not guaranteed convergence.
- T4 teacher batch starts at **2**. Try 4 after measuring memory/throughput.
  Only memory errors trigger smaller batches; graph/data errors stop execution.
- RTX training starts at batch **32**, BF16 if supported, otherwise FP16.
  Use `TRAIN_WORKERS=0` if multiprocessing is unavailable.
- Mounted Drive default: `/content/drive/MyDrive/MaiChi_Distillation_v03`.
  External runtimes can set `USE_DRIVE=False` and point `DRIVE_PARENT` at an
  already mounted/synced Drive folder. Mounting is not provided automatically
  outside supported Colab runtimes.
- Data/features are read from **local SSD**, not directly from Drive. Allow
  approximately 60 GB or more of local and Drive space depending on audio,
  features, cached data snapshots and checkpoints. Check actual quotas.

## What teacher batching means

The pinned public ZeroTTS ONNX export supports independent batched text encoding,
global-transformer steps and codec decoding. Its **local-frame sampling graph
has batch-one outputs**; its batch-two hidden state is for CFG, not two requests.
Our adapter calls that sampler separately per lane, preserving its sampling
knobs and repetition history, while batching the other stages. This is partial
model batching, not a fully vectorized sampler or a measured T4 speedup.

Text lengths are bucketed within each 128-text checkpoint block. Independent
RNG streams are seeded per text; floating-point differences across batch shapes
or hardware may still change sampled outputs. The T4 notebook performs a real
CUDA batch smoke check before collecting data and logs actual batch throughput.
Teacher graphs retain their original FP32 precision. ONNX telemetry is disabled
before initialization.

## Data transfer and resume

Notebook 01 writes WAVs and atomic sidecars to SSD, then publishes immutable tar
shards and a checksummed manifest to Drive after each 128-text block. It reserves
all 137 benchmark texts before training-text selection, records rejections and
extracts train-only vocabulary/WORLD features. After all features are present,
it publishes a **COMPLETE** bundle. An interrupted VM can lose the current
unsynced block; re-extracting features may be needed if their final publication
was interrupted. Completed Drive-saved WAVs are reused.

Notebook 02 verifies checksums, imports shards to SSD and rebinds paths to its
own filesystem. A portable corpus fingerprint allows checkpoint resume across
different runtime root paths. Partial/corrupt data or a different bundle for an
existing training run is rejected.

Training checkpoints are copied to Drive every 10 minutes and at each stage's
end. Backup objects retain the latest **two successful backup snapshots**,
including referenced best/last checkpoints. Checkpoint/optimizer/RNG states
resume; the data iterator restarts, so exact batch replay is not promised.
New experiments require a new local root and `RUN_LABEL`. Do not republish a
changed dataset into a running experiment. The old monolithic v2 run directories
are separate from this v3 workflow.

Teacher benchmark WAVs can also be generated on T4 and transferred. Student
benchmark inference then runs on RTX without teacher weights. Quality evaluation
uses identical text, but **T4 teacher RTF and RTX student RTF are different-hardware
measurements**; the comparison CSV labels this explicitly. Benchmark generation
is single-request, separate from batched corpus generation.

## Student architecture and strict budget

The student is a compact DiT trained with OT conditional flow matching in mel
space, learned durations/MAS and auxiliary pitch supervision, followed by a
compact HiFi-GAN-style vocoder. Train the vocoder first, then freeze it for
acoustic training. At inference no teacher audio/F0/durations are needed.
This is sequence-level audio distillation, not Teacher in the Loop online
correction or an official pretrained F5-TTS/Matcha-TTS model.

| Vocabulary | Acoustic modules | Waveform vocoder | Complete student |
| --- | ---: | ---: | ---: |
| 160 | 7,781,347 | 1,724,929 | **9,506,276** |
| 256 | 7,793,635 | 1,724,929 | **9,518,564** |
| 512 | 7,826,403 | 1,724,929 | **9,551,332** |

Actual vocabulary is checked before training. Initialization/export/reload fail
at >=10M. Counts include the auxiliary pitch head; 160 normalization buffer
elements are also checked in the exported tensor budget. GAN discriminator,
optimizer and duplicate EMA weights are training-only. There is no hidden
external neural codec/vocoder in the student. Teacher is 48 kHz; student 24 kHz.

## Validation and files

CPU checks completed: real teacher batch 2 with independent lengths, batch-one
upstream equivalence, padded codec versus individual decode, corpus-cache resume;
portable/checksummed shards; partial/corrupt data rejection; backup retention;
real-WAV vocoder/acoustic optimization and optimizer resume across different
root paths; full-student parameter/export regressions.

**Not verified:** actual T4/RTX GPU throughput/memory, full training, the 137-item
trained-student quality benchmark or a listening study. A new architecture and
short CPU smoke tests cannot establish state-of-the-art speech quality.

| File | Purpose |
| --- | --- |
| [Vietnamese guide](docs/FLOW_UNDER10M.md) | Two-runtime setup and limitations |
| [Batch teacher adapter](maichi_teacher_batch.py) | Multi-utterance ONNX orchestration, RNG/EOA/OOM/resume |
| [Data and checkpoint transfer](maichi_handoff.py) | Checksummed shards, portable paths, atomic backups |
| [Training supervisor](maichi_train_runner.py) | Local training with periodic/final Drive backups |
| [Flow student](maichi_flow.py) | Strict <10M model, training/export/inference |
| [Shared data/evaluation pipeline](maichi_pipeline.py) | Teacher assets, text selection, features and scoring |
| [v3 evidence](validation/MaiChi_split_v3_evidence.json) | Measurements and validation scope |
| [Pinned dependencies](requirements-colab.txt) | Keeps runtime CUDA PyTorch |

Local checks after dependency installation:

```bash
python test_maichi_teacher_batch.py
python test_maichi_teacher_batch.py --real --device cpu
python test_maichi_handoff.py --real-training
python test_maichi_flow.py
```

Benchmark count-only fixtures in integration tests are not benchmark results.
Fixed Mai Chi evaluation is not zero-shot cloning of the original reference
speakers. Paired teacher/student WavLM similarity is kept separate from official
reference-speaker SSIM. Select hyperparameters on validation, not the public test
set. Optional UTMOS is not human MOS.

Legacy [standalone VITS](notebooks/MaiChi_Standalone_Colab.ipynb),
[v1 guide](docs/README_MaiChi_vi.md) and v1/v2 evidence are retained for history;
the legacy medium model exceeds 10M and is not used in this workflow.

## Sources and rights

[ZeroTTS](https://github.com/zeroweight-ai/ZeroTTS),
[ZeroSpeech](https://huggingface.co/datasets/zeroweight-ai/ZeroSpeech),
[ZeroBench-TTS](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS),
[F5-TTS](https://github.com/SWivid/F5-TTS),
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS),
[VITS](https://github.com/jaywalnut310/vits).

New code is MIT. Attribution is retained in [LICENSE_ZEROTTS.txt](LICENSE_ZEROTTS.txt)
and [LICENSE_VITS.txt](LICENSE_VITS.txt). Dataset/source/voice rights are separate:
ZeroSpeech and ZeroBench are CC-BY-NC-4.0. This research pipeline does not establish
commercial rights to a distilled voice model.
