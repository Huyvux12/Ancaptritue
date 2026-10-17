# Fine-tune TITL/Paradee 8M cho tiếng Việt, từ WAV đến text-to-speech

Notebook chính: [02 — RTX 6000](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/02_MaiChi_Train_RTX6000.ipynb).
Notebook lấy dữ liệu: [01 — T4](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/01_MaiChi_Data_T4.ipynb).

## Quyết định kiến trúc

Dùng **checkpoint af_heart thật** của [Teacher in the Loop](https://huggingface.co/MithrilMan/teacher-in-the-loop),
giữ `TextStudent('s+mlp')` và `StudentDecoder('A')` từ
[mã upstream](https://github.com/MithrilMan/teacher-in-the-loop). Không thay bằng
DiT/Flow tự thiết kế, không khởi tạo lại vocoder, không ghép neural vocoder ngoài.

| Phần inference | Số tham số |
| --- | ---: |
| Text/duration/F0/N/features, sau mở rộng từ vựng | 4,261,908 |
| Decoder và harmonic source gốc | 3,848,578 |
| **Toàn bộ student** | **8,110,486** |

Đây là cách đếm mọi `Parameter` trong model được tải, kể cả weight normalization
và harmonic source đang đóng băng. CTC, PhoWhisper, discriminator, optimizer và
bản model anchor chỉ dùng trong train/đánh giá, không có trong artifact inference.
Bootstrap, load và export đều kiểm tra tổng **<10,000,000**.

Chỉ mở rộng hai ma trận embedding từ 178 lên 272 hàng. Các hàng cũ và mọi trọng
số decoder được chép nguyên. Hàng mới khởi tạo gần với ký tự/IPA gốc tương ứng;
đây là warm start, không phải bộ chuyển âm vị tiếng Việt. Kiểm tra English parity
với cùng RNG phải khớp chính xác giữa model trước/sau mở rộng embedding.

Frontend tiếng Việt dùng ký tự NFC, chữ thường; `a, à, á, ả, ã, ạ` có ID khác nhau.
Giữ phân biệt `ă/â/ê/ô/ơ/ư/đ`. Không đưa văn bản tiếng Việt qua Misaki tiếng Anh.
Số được mở rộng bằng `zerotts==0.1.5` utility, không tải teacher weights. Với số,
tên riêng, viết tắt hoặc từ mượn khó, nên bổ sung `spoken_text` đã duyệt. Chuỗi
ngoài từ vựng bị báo lỗi, không biến âm thầm thành padding. Câu dài chia thành
cụm tối đa 380 ký tự lúc inference; có khoảng nghỉ 120 ms giữa các cụm.

## Đây là một adaptation mới

Recipe TITL gốc dùng teacher Kokoro đóng băng, features/F0/N nội bộ, cùng excitation
và loss ở head iSTFT. WAV ZeroTTS/Mai Chi **không cung cấp các tensor đó**. Vì vậy
pipeline này là fine-tune có giám sát bằng audio của kiến trúc/checkpoint gốc,
không phải tái lập nguyên recipe paper, và chưa có bằng chứng bảo đảm chất lượng
tiếng Việt chỉ từ kết quả tiếng Anh.

`N` là tín hiệu prosody học được của Kokoro, không đồng nhất với log-RMS. Hiện tại
dùng log-RMS tiếng Việt, chuẩn hóa bằng thống kê **chỉ train**, rồi khớp mean/std
với `N` dự đoán trên bốn câu English gốc. Đây là surrogate gần đúng, có thể là
nút thắt khiến decoder không tái tạo tốt. Nó được ghi rõ trong metadata; không
đặt tên là teacher internal feature hoặc oracle reconstruction.

## Dữ liệu và hai runtime

1. T4 chạy notebook 01: teacher tạo WAV theo batch, lưu shard + checksum lên Drive.
2. RTX chạy notebook 02: tải raw WAV/text/splits về SSD; tải checkpoint TITL;
   căn chỉnh lại, overfit thử, kiểm tra chất lượng, rồi mới train full nếu đạt.
3. Snapshot prepared features, checkpoint và optimizer lên Drive mỗi 10 phút và
   cuối stage, giữ hai snapshot thành công gần nhất. WAV raw nằm riêng nên không
   nhân bản trong mỗi training snapshot. Khi VM mất, khôi phục snapshot và import
   raw bundle, chạy lại prepare để rebind đường dẫn.

Bundle COMPLETE v3 hiện có dùng lại được. Feature Hop=256 của Flow/VITS bị bỏ qua,
không được đưa vào decoder gốc. Notebook 01 vẫn tạo legacy features để tương
thích bundle; phần này có chi phí CPU dư. Không cần tạo lại WAV chỉ vì đổi student.

Đường dẫn mặc định:

| Thành phần | Vị trí |
| --- | --- |
| Bundle T4 trên Drive | `MaiChi_Distillation_v03/datasets/{pilot,full}` |
| Raw WAV SSD | `/content/maichi_raw_titl8m/MODE/RUN_LABEL` |
| Prepared/training SSD | `/content/maichi_titl8m/MODE/RUN_LABEL` |
| Backup Drive | `MaiChi_Distillation_v03/training_titl8m/MODE/RUN_LABEL` |

Raw-data producer commit và training commit là hai provenance riêng. Resume train
checkout commit trong `latest.json`, không checkout producer cũ vốn chưa có adapter
8M. Source hash, corpus/feature hash, vocab, optimizer configuration phải khớp.
Đổi data/code/hyperparameter cần ROOT và RUN_LABEL mới. Có thể tăng `steps` để
tiếp tục cùng run; iterator được tạo lại nên không cam kết lặp chính xác batch cũ.

## Đưa corpus riêng vào

Thay bước `restore_raw` bằng tạo `RAW_ROOT/corpus.jsonl`; mỗi dòng là một JSON:

```json
{"id":"vi_000001","text":"Xin chào, mình là Mai Chi.","spoken_text":"xin chào mình là mai chi","wav":"/absolute/path/vi_000001.wav","split":"train"}
```

`spoken_text` là tùy chọn; `text`, `id`, `wav`, `split` là bắt buộc. ID nên là
chuỗi duy nhất `[A-Za-z0-9_-]`. Split hợp lệ `train/val/test`; duy trì ranh giới
split theo nguồn/người đọc nếu corpus nhiều nguồn. WAV phải mono và được resample
24 kHz khi prepare. Hiện hỗ trợ câu 1–20 giây, tối đa 400 ký tự NFC. Nên dùng audio
sạch cùng một giọng, có đủ sáu thanh, vần đóng/mở và dấu câu. Không trộn test vào
train. `benchmark_manifest.jsonl` tùy chọn dùng để chặn các text benchmark đã biết;
kiểm tra trùng nghĩa/gần trùng theo nguồn vẫn cần thêm khi xây corpus riêng.

Đối với dữ liệu từ notebook 01, transcript, checksum audio, split và benchmark
reservation đã có trong bundle. Prepare lưu accepted/rejected và nguyên nhân,
không âm thầm bỏ qua âm thanh lỗi.

## Đồng hồ, căn chỉnh và đặc trưng

| Tensor | Tần số/shape |
| --- | --- |
| WAV | 24,000 Hz, trim tới bội của 600 samples |
| Token duration và expanded ASR features | 40 Hz, **600 samples/frame** |
| F0, loudness surrogate | 80 Hz, **300 samples/frame** |
| BOS/EOS | mỗi token một frame, tổng duration khớp chiều dài WAV |

Dùng [wav2vec2-base-vi-vlsp2020](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi-vlsp2020)
đã fine-tune CTC, không dùng model base pretraining chưa có head CTC. Model có
ba lớp Linear/BatchNorm/LeakyReLU custom trước lm_head; `titl_vi/ctc.py` triển khai
đúng cấu trúc, yêu cầu **không missing/unexpected/mismatched weights**, không chạy
remote code. Blank ID lấy từ vocab/config (95), không mặc định là 0.

Viterbi CTC xử lý cả ký tự lặp cần blank ngăn cách; centers đổi từ clock CTC sang
40 Hz, ép duration dương và tổng frame chính xác. Reject khi CER aligner >35% hoặc
mean aligned probability <0.10. Đây là heuristic lọc chất lượng transcript/alignment,
không chứng minh student nói được. WORLD DIO + StoneMask lấy F0 12.5 ms; giữ 0 cho
unvoiced, không nội suy voiced qua vùng im lặng. Log-RMS lấy trên cửa sổ 300 samples.

## Các stage

| Stage | Default steps | Trọng số cập nhật | Điều kiện |
| --- | ---: | --- | --- |
| English bootstrap | 0 | Không | Tải đủ weights, exact parity, waveform hữu hạn |
| Small bridge | 2,000 | Text/duration/F0/N/features | 32 câu train, decoder đóng băng |
| Small joint | 4,000 | Text + decoder | Init small bridge; harmonic source đóng băng |
| Overfit gate | — | Không | PhoWhisper độc lập, train subset đã chọn |
| Full bridge | 8,000 | Text | Overfit gate đúng SHA; ≥1,000 train, ≥20 val |
| Full joint | 20,000 | Text + decoder | Validation `pre_joint` gate đúng SHA |
| GAN tùy chọn | 5,000 | Text + decoder + critics | Validation `pre_gan`; mặc định tắt |
| Export | — | Không | Validation final gate đúng checkpoint/corpus |

Text LR 1e-4, decoder LR 2e-5, warmup 200 steps và decay theo step; AdamW,
gradient clip 5, FP32. Đo VRAM/tốc độ thực trên runtime trước khi tăng batch từ 8.
Không có benchmark RTX đã đo để cam kết thời gian hoàn tất.

Loss: 5×log-power-mel L1 + 2×multi-resolution STFT, duration log1p L1, voiced/
unvoiced F0 L1, N L1, latent feature anchor nhẹ. Không dùng waveform L1 vì teacher
và student có phase/excitation khác nhau. Segment waveform 64 frames = 1.6 s.
Bridge dùng measured F0/N; joint/GAN tăng dần predicted-prosody mixing tới 100%
qua 2,000 steps. GAN dùng MPD/MRD gốc, LSGAN và feature matching. Không cần GAN
để kiểm tra khả năng phát âm ban đầu.

## Gate và chẩn đoán lỗi

Mỗi gate phát ba bản cùng text: aligned + measured prosody, aligned + predicted
prosody, free text. Nếu bản đầu đã rè, kiểm tra surrogate N/feature-domain/decoder;
nếu bản đầu rõ và bản hai kém, kiểm tra F0/N predictor; nếu chỉ bản free text kém,
kiểm tra duration/frontend. Đây là hướng điều tra, không kết luận nguyên nhân chỉ
từ một loại waveform.

[PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) độc lập với aligner
chép teacher WAV và student free-text WAV. Teacher CER ≤20%; student CER ≤min(20%,
teacher+10 điểm phần trăm). Có kiểm tra RMS/peak cơ bản. Gate report ghi revision
ASR, corpus fingerprint và checkpoint SHA; không dùng report cho checkpoint khác.
Gate thất bại ném lỗi, chặn full stage/export. Đừng sửa `passed=true` hoặc nới ngưỡng
chỉ để chạy tiếp. Kiểm tra WAV, transcript và ba diagnostic views trước.

Gate mặc định 8 câu chỉ là sàng lọc intelligibility; không đủ chứng minh thanh điệu,
naturalness hay giống giọng. Cần validation đa dạng và nghe thử mù với các cặp sáu
thanh, âm cuối, số/ngày tháng, từ dài và câu ngoài corpus. Việc mỗi dấu có ID riêng
không bảo đảm model đã học đúng thanh. Export giữ `quality_verified=false` để phản
ánh giới hạn này ngay cả khi gate tự động đạt.

## Chạy từ terminal

Sau khi cài `requirements-titl-vi.txt`, WORLD và zerotts utility:

```bash
python -m titl_vi.cli bootstrap --root /ssd/vi8m --device cuda
python -m titl_vi.cli import --root /ssd/vi8m --raw-root /ssd/raw --bundle-dir /mounted/Drive/datasets/full
python -m titl_vi.cli prepare --root /ssd/vi8m --raw-root /ssd/raw --device cuda
python -m titl_vi.cli train --root /ssd/vi8m --run small_bridge --phase bridge --overfit 32 --steps 2000 --device cuda
python -m titl_vi.cli train --root /ssd/vi8m --run small_joint --phase joint --overfit 32 --steps 4000 --init /ssd/vi8m/runs/small_bridge/best.pt --device cuda
python -m titl_vi.cli gate --root /ssd/vi8m --checkpoint /ssd/vi8m/runs/small_joint/best.pt --name overfit --split train --overfit 32 --count 8 --device cuda
```

Chỉ chạy full theo notebook sau khi gate đạt. `speak` dùng artifact `export/student_vi_8m.pt`;
chạy text-to-WAV không cần CTC, ASR hay teacher weights. Tạm thời inference là Python/
PyTorch; chưa hứa ONNX/TFLite/mobile export và chưa có phép đo latency trên thiết bị.

## Bằng chứng và giấy phép

[Evidence kỹ thuật công khai](../validation/TITL8M_VI_evidence.json) ghi kiểm tra checkpoint/implementation,
không công bố metadata/hash audio hoặc kết quả ASR từng mẫu. Môi trường kiểm tra hiện tại là CPU;
chưa train full hoặc đo GPU/memory RTX. Kết quả overfit ngắn không tương đương bản
tiếng Việt dùng được. Không phân phối một checkpoint Vietnamese chất lượng cao đã
được xác nhận trong thay đổi này.

[Attribution](../titl_vi/ATTRIBUTION.md) giữ license Apache-2.0 của upstream và
CC-BY-SA của English calibration text. Data/teacher voice rights độc lập; corpus
ZeroSpeech/ZeroBench có điều kiện CC-BY-NC-4.0, pipeline này không cấp quyền thương mại.
