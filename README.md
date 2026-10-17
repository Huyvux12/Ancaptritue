# Mai Chi single-speaker TTS distillation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/MaiChi_Clone_Colab.ipynb)

Train and evaluate a compact Vietnamese TTS student using audio synthesized by
ZeroTTS with the fixed `maichi` voice. The student is a compact VITS variant with
optional F0/voicing supervision and conditioning. This is sequence-level
distillation from generated audio, **not a reproduction of Teacher in the Loop**.

## Run on Colab

1. Open [MaiChi_Clone_Colab.ipynb](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/MaiChi_Clone_Colab.ipynb).
2. Connect the intended GPU runtime and run the notebook from the top. It clones
   this public repository, installs dependencies and records the source commit.
3. Start with `MODE = "pilot"`: 512 sentences and 2,000 training steps. This checks
   the workflow; it does not promise natural speech.
4. Switch to `MODE = "full"` for 20,000 sentences, 100,000 steps and the 137-item
   evaluation. Repeat with the same configuration to resume saved checkpoints.

For an external RTX PRO 6000 96 GB runtime, set `USE_DRIVE = False` if Drive
mounting is unavailable and set `PERSISTENT_BASE` to a persistent disk. The
notebook checks actual CUDA execution. Start at batch size 32; increase only
after measuring memory. Full runs should have approximately 60 GB free, with
actual requirements depending on utterance length and feature compression.

The GPU PyTorch installation is retained. The teacher uses ONNX Runtime while
student training uses PyTorch. If ONNX CUDA cannot run, explicitly select
`TEACHER_DEVICE = "cpu"`; student training can still use CUDA.

## Workflow

- Download pinned ZeroTTS inference assets and test Mai Chi synthesis.
- Reserve all ZeroBench test items before collecting training text.
- Sample revision-checked ZeroSpeech transcript pages without downloading its audio.
- Regenerate accepted text using Mai Chi, with resumable WAV/provenance storage.
- Extract WORLD F0 and voicing, audit sample transcripts, and train the student.
- Save optimizer/RNG checkpoints and export speech inference weights.
- Evaluate teacher/student with the official dual-ASR scorer, optional UTMOSv2,
  latency/RTF, size, and a separately labeled Mai Chi speaker-similarity protocol.
- Optionally run a pitch ablation, a larger student and a listening worksheet.

Approximate deployment sizes are 7.2M parameters (`small`) and 17.9M (`medium`),
including waveform generation. Exact counts depend on vocabulary and are
reported by the pipeline. Training-only posterior/discriminator weights are
excluded from deployment counts. Teacher output is 48 kHz; student output is
24 kHz, which must be disclosed in comparisons.

## Files

| File | Purpose |
| --- | --- |
| [Clone notebook](notebooks/MaiChi_Clone_Colab.ipynb) | Recommended: clone this repo and run its source |
| [Standalone notebook](notebooks/MaiChi_Standalone_Colab.ipynb) | Complete source embedded in readable Python cells |
| [maichi_pipeline.py](maichi_pipeline.py) | Download, text selection, corpus, training, export, scoring |
| [requirements-colab.txt](requirements-colab.txt) | Shared pinned Python dependencies; keeps runtime Torch |
| [Vietnamese guide](docs/README_MaiChi_vi.md) | Detailed setup, protocol and limitations |
| [Validation evidence](validation/MaiChi_validation_evidence.json) | Measurements and checks actually completed |
| [CPU regression checks](test_maichi_pipeline.py) | Alignment, gradients, deployment equivalence, ablation |
| [Teacher samples](samples/) | Three real ZeroTTS Mai Chi WAVs |

## Validation status

Completed: real teacher download and CPU synthesis; four-item corpus generation
and resume; feature extraction; three real-WAV training steps including GAN,
pitch, duration and KL losses; checkpoint resume; exact deployment reload;
alignment/gradient checks and CPU BF16 forward. The benchmark manifest contains
all 137 items, and real ZeroSpeech transcript selection was exercised.

**Not completed:** full GPU training, execution on an RTX PRO 6000 96 GB,
official 137-item quality scoring, or a listening study of a trained student.
The small ASR audit is not a quality guarantee or a leaderboard result.

This fixed-Mai-Chi setup does not evaluate zero-shot cloning of the original
reference speakers. Do not report the official reference-speaker SSIM field as
Mai Chi similarity. Use the separately labeled teacher/student WavLM comparison.
Hyperparameters must be selected on validation data, not the public test set.

## Local source use

Install dependencies using the notebook's installation steps. Then, for example:

```bash
git clone https://github.com/Huyvux12/Ancaptritue.git
cd Ancaptritue
python maichi_pipeline.py smoke --root /persistent/maichi/pilot --device cpu
python maichi_pipeline.py benchmark --root /persistent/maichi/pilot
python maichi_pipeline.py texts --root /persistent/maichi/pilot --count 512
python maichi_pipeline.py generate --root /persistent/maichi/pilot --device cuda
python maichi_pipeline.py features --root /persistent/maichi/pilot --workers 4
python maichi_pipeline.py train --root /persistent/maichi/pilot \
  --vits-dir ./vits --name small_pitch --size small --steps 2000 \
  --batch-size 32 --device cuda --allow-small
python test_maichi_pipeline.py --vits-dir ./vits
```

The pipeline downloads and verifies a pinned VITS checkout when necessary.
The notebooks use readable code and contain no base64 source payloads.

## Sources and rights

[ZeroTTS](https://github.com/zeroweight-ai/ZeroTTS),
[ZeroSpeech](https://huggingface.co/datasets/zeroweight-ai/ZeroSpeech),
[ZeroBench-TTS](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS),
[VITS](https://github.com/jaywalnut310/vits),
[Teacher in the Loop](https://github.com/MithrilMan/teacher-in-the-loop).

New pipeline code is MIT. VITS attribution is retained in
[LICENSE_VITS.txt](LICENSE_VITS.txt). Dataset/source/voice rights are separate:
ZeroSpeech and ZeroBench are CC-BY-NC-4.0. This research workflow does not
establish commercial rights to a distilled voice model.
