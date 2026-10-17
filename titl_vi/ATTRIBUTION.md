# Provenance

The original Paradee/TITL neural architecture is retained. Upstream code:
https://github.com/MithrilMan/teacher-in-the-loop at
`8d457dd80a73619fbf2e75074c5b707cbb604fbd`.

- `upstream/student.py`, `student_decoder.py`, `discriminators.py`: unchanged
  upstream training files, Apache-2.0. Original licenses/NOTICE retained.
- `upstream/exportable.py`: `ConvSTFT` and `Speaker` extracted from upstream
  `training/export.py`; imports reduced to torch/nn/functional. Apache-2.0.
- `calibration_en.json`: four shortest English text/phoneme-ID rows selected
  from upstream `eval/heldout.json`. Wikipedia contributors / WikiText-103 /
  Paradee attribution and CC-BY-SA-4.0 terms are retained in
  [CALIBRATION_LICENSE.md](CALIBRATION_LICENSE.md). Selection is our change;
  selected text/IDs were not rewritten. Share-alike applies to this list.
- Released weights are downloaded from `MithrilMan/teacher-in-the-loop` at
  `115a6c0469632a84fbcc6c8ed0ad7b1e06154b92` (Apache-2.0); not committed here.
- New Vietnamese adaptation code outside `upstream` is MIT, except the
  calibration list above. It is not an official upstream Vietnamese release.

Vietnamese CTC is downloaded from
`nguyenvulebinh/wav2vec2-base-vi-vlsp2020` at
`50a30dadb3ec98a0d4cdb1eb1ea315aff538f7c2`. Its custom forward head is
implemented locally with strict state loading. PhoWhisper is downloaded for
independent evaluation; its resolved revision is recorded in each gate report.
These tools and data/voice rights retain their respective upstream terms.
