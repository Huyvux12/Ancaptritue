# Mai Chi Flow: complete TTS student below 10M parameters

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/MaiChi_Flow_Under10M_Colab.ipynb)

Train a compact Vietnamese single-speaker student from ZeroTTS audio with the
fixed `maichi` voice. The recommended v2 notebook uses a **compact DiT with
optimal-transport conditional flow matching (OT-CFM), learned durations and a
compact waveform vocoder**. The entire student must stay strictly below
10,000,000 parameters, including the vocoder. It is trained from scratch.

This is an experimental budget-constrained implementation inspired by modern
flow-matching TTS methods. It is not an official F5-TTS/Matcha-TTS checkpoint,
a demonstrated state-of-the-art model, or a reproduction of Teacher in the Loop.
Quality must be measured after full training.

## Run on Colab

1. Open [MaiChi_Flow_Under10M_Colab.ipynb](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/MaiChi_Flow_Under10M_Colab.ipynb).
2. Connect the GPU and run from the top. The notebook clones this repo, installs
   dependencies, records source hashes and checks actual CUDA operations.
3. Start with `MODE="pilot"`: 512 sentences, 2,000 vocoder steps and 4,000 acoustic
   steps. This checks the workflow; it does not establish natural speech.
4. Use `MODE="full"`: 20,000 sentences, 100,000 vocoder steps and 200,000 acoustic
   steps. Inspect validation curves and listening samples; these are starting
   budgets, not a convergence guarantee. Re-run unchanged settings to resume.

For an external RTX PRO 6000 96 GB runtime, set `USE_DRIVE=False` if mounting
Drive is unavailable, and use a persistent `PERSISTENT_BASE`. Start at batch 32;
measure memory before increasing it. Full runs should have approximately 60 GB
free; actual usage varies with utterance length and feature compression. Set
`TRAIN_WORKERS=0` if the runtime cannot use multiprocessing.

Keep the runtime's CUDA PyTorch. The teacher uses ONNX Runtime; the student uses
PyTorch. Select `TEACHER_DEVICE="cpu"` explicitly if ORT CUDA cannot run.
The original [MaiChi_Clone_Colab.ipynb](notebooks/MaiChi_Clone_Colab.ipynb) link
also opens the new flow workflow.

## Strict deployment budget

Measured directly from the default model, not estimated from file size:

| Vocabulary | Acoustic modules | Waveform vocoder | Complete student |
| --- | ---: | ---: | ---: |
| 160 entries | 7,781,347 | 1,724,929 | **9,506,276** |
| 256 entries | 7,793,635 | 1,724,929 | **9,518,564** |
| 512 entries | 7,826,403 | 1,724,929 | **9,551,332** |

The actual corpus vocabulary is checked before training. Initialization, export
and deployment reload reject an over-budget student. All student modules,
including the auxiliary pitch head, are counted. The 160 normalization buffer
elements are reported separately and also included in the exported-state guard.
No external neural codec or pretrained waveform decoder is needed at inference.
The GAN discriminator, optimizer and duplicate EMA state are training-only.
The teacher and evaluation ASR models are not part of the deployed student.

## Workflow

- Download pinned ZeroTTS assets and listen to real Mai Chi synthesis.
- Reserve all 137 ZeroBench items before selecting training text.
- Fetch revision-checked ZeroSpeech transcripts and regenerate them with Mai Chi.
- Audit audio/transcripts, extract mel/F0/voicing and compute train-only statistics.
- Train the compact vocoder with mel/STFT/GAN/feature-matching losses.
- Freeze the vocoder and train a 10-block, width-192 DiT with OT-CFM, monotonic
  alignment, duration/prior objectives and auxiliary pitch supervision.
- Save optimizer/RNG checkpoints, select EMA weights using validation and export
  one complete acoustic/vocoder state.
- Compare 4/8/16 flow evaluations on the fixed 137-item benchmark: official
  dual-ASR, optional UTMOSv2, RTF/size, separate Mai Chi similarity and listening.

Teacher output is 48 kHz; student output is 24 kHz. The fixed-Mai-Chi protocol
is not zero-shot cloning of the benchmark's original speakers. The notebook
keeps the adapted teacher/student WavLM score separate from official reference
speaker similarity. Select hyperparameters on validation, not public test items.

## Files and validation

| File | Purpose |
| --- | --- |
| [Flow Colab notebook](notebooks/MaiChi_Flow_Under10M_Colab.ipynb) | Recommended end-to-end clone-and-run workflow |
| [Flow implementation](maichi_flow.py) | Model, two-stage training, export and benchmark inference |
| [Shared pipeline](maichi_pipeline.py) | Teacher, corpus, features and official evaluation; legacy VITS support |
| [Vietnamese v2 guide](docs/FLOW_UNDER10M.md) | Architecture choice, setup and limits |
| [Flow regression checks](test_maichi_flow.py) | Budget, alignment, gradients, BF16 and exact deployment reload |
| [Real-WAV smoke test](test_maichi_flow_real_audio.py) | Short vocoder/acoustic training, checkpoint resume and synthesis |
| [v2 validation evidence](validation/MaiChi_flow_v2_evidence.json) | Exact counts and checks actually completed |
| [Pinned dependencies](requirements-colab.txt) | Retains runtime Torch |
| [Teacher samples](samples/) | Real ZeroTTS Mai Chi WAVs |

Completed on CPU: regression checks; two vocoder steps on real teacher WAVs;
two acoustic steps plus resume to step three; finite losses; export/reload and
finite text-only waveform generation. A multiprocessing smoke test was blocked
by this environment's local-socket restriction; the single-process path passed.

**Not completed:** full GPU training, RTX PRO 6000 execution, the 137-item quality
benchmark, or a listening study of a trained student. No student quality claim
is made from the short smoke run. GPU checks fail early in the notebook if the
runtime lacks CUDA support.

The [standalone notebook](notebooks/MaiChi_Standalone_Colab.ipynb),
[old Vietnamese guide](docs/README_MaiChi_vi.md) and
[v1 evidence](validation/MaiChi_validation_evidence.json) describe the **legacy
VITS baseline**. Its optional medium model exceeds 10M and is not used by v2.

## Local checks

After installing the notebook's dependencies:

```bash
python test_maichi_flow.py
python test_maichi_flow_real_audio.py
python maichi_flow.py budget --vocab-size 256
```

See the notebook for data collection, the two required training stages and full
evaluation. Source is readable Python; notebooks contain no encoded payloads.

## Sources and rights

[ZeroTTS](https://github.com/zeroweight-ai/ZeroTTS),
[ZeroSpeech](https://huggingface.co/datasets/zeroweight-ai/ZeroSpeech),
[ZeroBench-TTS](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS),
[F5-TTS](https://github.com/SWivid/F5-TTS),
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS),
[ZipVoice](https://github.com/k2-fsa/ZipVoice),
[VITS](https://github.com/jaywalnut310/vits).

New code is MIT. Attribution for VITS-derived MAS and HiFi-GAN-style components
is retained in [LICENSE_VITS.txt](LICENSE_VITS.txt). Dataset/source/voice rights
are separate: ZeroSpeech and ZeroBench are CC-BY-NC-4.0. This research workflow
does not establish commercial rights to a distilled voice model.
