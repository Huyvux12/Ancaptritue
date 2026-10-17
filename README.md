# Mai Chi Vietnamese TTS: pretrained TITL 8M adaptation

Use the **actual pretrained Paradee / Teacher-in-the-Loop architecture** for the
new experiment. The original text backbone and waveform decoder are retained;
Vietnamese tone-bearing character embeddings are appended. Complete inference
student: **8,110,486 parameters**, including waveform generation, strictly <10M.

**Vietnamese quality is experimental.** The upstream English checkpoint works as
a pretrained baseline; that does not establish Vietnamese quality. ZeroTTS WAV
fine-tuning differs from the paper's Kokoro internal-feature recipe. Small-set
listening and independent ASR gates must pass before full training/export.
Read the [design and failure diagnosis](docs/TITL8M_VI_DESIGN.md) and
[measured validation scope](validation/TITL8M_VI_evidence.json).

No quality-verified Vietnamese checkpoint is published with this code. Quality
measurements and corpus-derived records are omitted from this public report.

| Order | Notebook | Runtime | Work |
| --- | --- | --- | --- |
| 1 | [01 — Data on T4](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/01_MaiChi_Data_T4.ipynb) | T4 | Batched Mai Chi teacher WAVs, checksummed Drive shards |
| 2 | [02 — Actual pretrained TITL 8M](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/02_MaiChi_Train_RTX6000.ipynb) | RTX 6000 / CUDA | Import raw WAVs, Vietnamese alignment, small-set gates, adaptation, export |

Existing **COMPLETE v3 data bundles can be reused**. Notebook 02 skips the old
Flow/VITS features and recomputes the original model's 40-Hz durations and 80-Hz
pitch/loudness. It does not regenerate teacher audio or load teacher weights.
`zerotts` is used only for its pinned number-normalization utility during training
and inference. Notebook 01 still prepares legacy features for bundle compatibility.

The previous custom Flow student was reported to produce buzzing after training.
Its [notebook](notebooks/legacy/02_MaiChi_Flow_Train_RTX6000.ipynb) and
[old description](docs/LEGACY_FLOW_README.md) are retained as legacy experiments;
their short engineering checks did not demonstrate speech quality.

## Run the two notebooks

- Keep `MODE` and `DRIVE_PARENT` consistent. Default Drive parent:
  `/content/drive/MyDrive/MaiChi_Distillation_v03`.
- Start with `pilot` for execution checks. Full adaptation requires `full` data
  from the start, with >=1,000 accepted train and >=20 validation items.
- Finish notebook 01 until `READY FOR NOTEBOOK 02`. On RTX, notebook 02 imports
  raw audio to SSD, loads the pinned pretrained weights, verifies English parity,
  aligns Vietnamese, then runs 32-utterance bridge/joint trials.
- **Stop if the gate fails.** Inspect teacher audio and three student views:
  aligned/measured prosody, aligned/predicted prosody, and free text. Do not run
  a full job on the assumption that more steps will fix decoder mismatch.
- Full adaptation, optional GAN, validation gate and deployment export follow
  only after the required gates. These are starting budgets, not measured
  convergence guarantees. FP32 and batch 8 are conservative defaults; measure
  actual step time/VRAM before increasing batch.
- External GPU runtimes can set `USE_DRIVE=False` and supply an already mounted
  or synced Drive directory. Keep data/training on local SSD.

Training snapshots are saved every ten minutes and after stages, with the latest
two successful snapshots retained. TITL backups use a separate `training_titl8m`
directory. Resume checks source, data, vocabulary and configuration fingerprints;
optimizer and RNG are restored, but the data iterator restarts. A new experiment
requires a new `RUN_LABEL` and local root. Training source is pinned independently
from the old raw-data producer commit.

## Implementation

| File | Purpose |
| --- | --- |
| [Vietnamese design](docs/TITL8M_VI_DESIGN.md) | Architecture, clocks, corpus schema, stages, gates, limitations |
| [Model and pretrained loading](titl_vi/model.py) | Original architecture, appended Vietnamese IDs, strict <10M budget |
| [Preparation](titl_vi/data.py), [CTC](titl_vi/ctc.py) | Text/WAV -> 40-Hz durations, 80-Hz F0/loudness |
| [Training](titl_vi/train.py) | Frozen-decoder bridge, joint/GAN, strict checkpoint resume |
| [Quality gates/export](titl_vi/gate.py) | Independent paired PhoWhisper checks, text-only deployed inference |
| [Storage](titl_vi/storage.py) | Checksummed SSD/Drive snapshots |
| [Benchmark](titl_vi/benchmark.py) | Fixed 137-item held-out text-to-WAV evaluation |
| [CLI](titl_vi/cli.py) | Readable end-to-end stage commands |
| [Dependencies](requirements-titl-vi.txt) | Retains the runtime's CUDA Torch |
| [Tests](test_titl_vi.py), [evidence](validation/TITL8M_VI_evidence.json) | Real checks and explicit quality limits |

No base64 code payloads. Notebooks clone this repository and invoke readable
Python modules. `export/student_vi_8m.pt` contains one inference model; no CTC,
PhoWhisper, discriminator, optimizer or external neural vocoder is deployed.
The current inference target is PyTorch, not an untested mobile/ONNX export.

## Validation and quality scope

Actual released weights were loaded strictly. Expanded-vocabulary English output
matches the original token path exactly with the same RNG. Original versus
exportable STFT was compared; real Vietnamese CTC/WAV preparation, finite gradient
updates and optimizer resume were exercised on CPU. The public evidence file records implementation checks only; it does not
contain corpus-derived ASR results or audio metadata.

**Not established:** successful full-corpus Vietnamese adaptation, actual T4/RTX
training speed/memory, six-tone accuracy, human MOS or similarity to new large
models. ASR gates on selected utterances are necessary screening, not sufficient
proof of natural speech. Full export remains marked `quality_verified=false`
until human/held-out evaluation is performed. No quality-verified Vietnamese
checkpoint is distributed here.

The optional 137-item evaluation preserves public test separation. Mai Chi
teacher/student WavLM similarity is an adapted fixed-voice measure, separate from
original reference-speaker similarity. T4 teacher and RTX student RTF are
measurements on different hardware. Optional predicted MOS is not human MOS.

## Sources and rights

[Teacher in the Loop](https://github.com/MithrilMan/teacher-in-the-loop),
[released weights](https://huggingface.co/MithrilMan/teacher-in-the-loop),
[Paradee](https://github.com/sahilmahendrakar/paradee),
[Vietnamese CTC](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi-vlsp2020),
[PhoWhisper](https://huggingface.co/vinai/PhoWhisper-small),
[ZeroTTS](https://github.com/zeroweight-ai/ZeroTTS),
[ZeroSpeech](https://huggingface.co/datasets/zeroweight-ai/ZeroSpeech),
[ZeroBench-TTS](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS).

New adaptation code is MIT. Upstream code/weights are Apache-2.0, with retained
[licenses and attribution](titl_vi/ATTRIBUTION.md); the selected English calibration
text is CC-BY-SA-4.0. ZeroSpeech/ZeroBench are CC-BY-NC-4.0; data and voice rights
are separate from code. This pipeline does not establish commercial voice rights.
