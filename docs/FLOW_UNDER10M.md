# Mai Chi Flow v2 — student dưới 10 triệu tham số

Mở [notebook Colab](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/MaiChi_Flow_Under10M_Colab.ipynb), kết nối GPU và chạy từ đầu. Notebook và mã nguồn dùng tiếng Anh. Link `MaiChi_Clone_Colab.ipynb` cũ cũng đã chuyển sang v2.

## Lựa chọn kiến trúc

Student mới dùng **DiT nhỏ + optimal-transport conditional flow matching**, dự đoán mel rồi đưa qua vocoder nhỏ tự train. Đây là hướng thử nghiệm phù hợp với giới hạn chặt về số tham số và giọng Mai Chi cố định. Các phương pháp tham khảo: [F5-TTS](https://github.com/SWivid/F5-TTS), [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS), [ZipVoice](https://github.com/k2-fsa/ZipVoice). Không dùng trọng số hay tuyên bố tái lập chính xác các model này.

Text encoder có 3 block, width 128; flow có 10 block DiT, width 192, 4 attention head và adaLN-zero. Text giữ dấu tiếng Việt. MAS học alignment đơn điệu trong train, có prior gần đường chéo giảm dần trong 10.000 bước đầu. Duration predictor quyết định độ dài lúc inference; DiT sinh mel bằng 4/8/16 lần tính velocity; vocoder chuyển mel thành waveform 24 kHz. F0/voicing là giám sát phụ để kiểm tra bằng ablation, không lấy F0 teacher ở inference.

Vocoder là mạng convolution kiểu HiFi-GAN; việc đổi acoustic model sang flow matching không khiến mọi thành phần khác trở thành kiến trúc mới. Toàn bộ model khởi tạo từ đầu, không có codec lớn ẩn phía sau. Dữ liệu teacher cung cấp text–audio, tức sequence-level distillation; chưa triển khai vòng sửa lỗi online Teacher in the Loop.

**Không thể khẳng định đây là model tốt nhất hoặc đảm bảo ngang model lớn hiện đại.** Đây là lựa chọn triển khai dưới ngân sách đã cho; kết luận chất lượng cần full training, benchmark và nghe mù.

## Ngân sách đo thật

Với vocabulary 256: acoustic **7.793.635** + vocoder **1.724.929** = **9.518.564 tham số**. Với vocabulary 512: **9.551.332**. Tổng bao gồm cả pitch head dù chỉ phục vụ loss phụ. Bộ đếm dùng vocabulary thực sau khi lấy dữ liệu và chặn nếu tổng >=10.000.000.

160 phần tử buffer mean/std được báo riêng; guard export còn kiểm tra cả tổng tensor của state dưới 10M. Discriminator GAN, optimizer và bản sao EMA chỉ dùng khi train, không được xuất. Model teacher và ASR dùng để tạo/chấm dữ liệu không nằm trong student triển khai.

## Cách chạy

1. Bắt đầu `MODE="pilot"`: 512 câu, vocoder 2.000 bước, acoustic 4.000 bước. Đây là kiểm tra pipeline, chưa kỳ vọng giọng tự nhiên.
2. Chuyển `MODE="full"`: 20.000 câu, vocoder 100.000 bước, acoustic 200.000 bước. Đây là cấu hình xuất phát, không phải cam kết hội tụ. Đọc validation curve và nghe mẫu trước khi kết luận.
3. RTX PRO 6000 96 GB: giữ PyTorch CUDA phù hợp runtime, batch khởi đầu 32. Notebook kiểm tra tên GPU, VRAM và phép tính CUDA thật. Runtime ngoài Colab đặt `USE_DRIVE=False`, trỏ `PERSISTENT_BASE` tới ổ bền vững.
4. Stage A train vocoder trên mel/WAV teacher với mel, STFT, feature matching và GAN loss. Stage B load EMA vocoder tốt nhất theo validation rồi đóng băng, train acoustic model. Không bỏ qua Stage A.
5. Nếu multiprocessing không được runtime cho phép, đặt `TRAIN_WORKERS=0`; feature extraction có cấu hình riêng. Chọn BF16 khi GPU hỗ trợ, FP16 hoặc FP32 nếu cần.
6. Sau full train, sinh và chấm đủ 137 câu với cùng text thô cho teacher/student. Grid 4/8/16 được định trước; không chọn hyperparameter theo test set. Bật `RUN_ABLATION=True` để chạy thêm cùng kiến trúc với pitch loss bằng 0.

Cần khoảng 60 GB trống cho full run, thay đổi theo độ dài audio/cache. Teacher phát 48 kHz, student 24 kHz. Không đưa ra ước tính giờ train hay mức sử dụng VRAM khi chưa đo trên GPU này.

## Dữ liệu, resume và export

Notebook tải teacher có revision cố định, nghe thử, giữ riêng benchmark trước khi chọn text, lọc overlap exact/substring, tạo WAV Mai Chi có provenance, audit ASR, trích WORLD F0 và chia train/val/test theo hash. Mel mean/std chỉ tính từ train. Lọc overlap không bảo đảm hết paraphrase hoặc trùng nguồn.

`ROOT` mới tách biệt với thư mục VITS cũ. `flow_runs/vocoder/` chứa Stage A; `flow_runs/flow10m/` chứa Stage B. Mỗi run lưu `last.pt`, `best.pt`, config, log train/validation. Resume khôi phục optimizer, scaler, RNG và bước train; iterator dữ liệu khởi tạo lại nên không bảo đảm replay batch tuyệt đối. Đổi source, corpus, batch hoặc cấu hình cần run/thư mục mới. Muốn tiếp tục bản cũ, đặt `REPO_REF` bằng commit đã ghi trong `source_revision.json`.

`student_deploy.pt` chỉ chứa một state acoustic+vocoder, vocabulary và normalization. Không kèm discriminator hay bản sao EMA. Export chọn EMA từ checkpoint có validation loss tốt nhất (hoặc last nếu chưa có best), không chọn theo benchmark. Inference từ text không yêu cầu waveform/F0/duration teacher. Câu vượt trần 2.000 mel frame sẽ báo lỗi để chia câu, không cắt âm thầm.

## Đánh giá và tình trạng kiểm chứng

Notebook giữ official dual-ASR scorer, UTMOSv2 tùy chọn, RTF trên cùng thiết bị và WavLM teacher–student Mai Chi tách riêng. Fixed-Mai-Chi không phải zero-shot cloning các giọng tham chiếu gốc; không so trường similarity đã đổi protocol với SSIM của leaderboard. RTF cần cùng phần cứng/thread; UTMOS không phải human MOS. Worksheet nghe mù tạo thứ tự A/B ngẫu nhiên và đáp án riêng.

Đã chạy CPU: bộ đếm/guard; alignment và gradient DiT; giữ dấu và padding; BF16 objective; flow integration; export/reload khớp chính xác; train vocoder 2 bước trên WAV teacher thật, acoustic 2 bước rồi resume bước 3, waveform hữu hạn. Có script tái chạy trong repo. Multiprocessing bị môi trường kiểm thử chặn local socket nên chưa xác minh đường nhiều worker tại đây.

**Chưa chạy full GPU training, chưa xác nhận runtime RTX PRO 6000, chưa có benchmark chất lượng 137 câu hay nghiên cứu nghe của student đã hội tụ.** Xem [evidence v2](../validation/MaiChi_flow_v2_evidence.json). Các sample WAV trong repo là teacher, không đại diện chất lượng student mới.

Bản VITS vẫn giữ để so sánh nghiên cứu trong standalone notebook và tài liệu legacy. Medium VITS >10M không được notebook v2 sử dụng.
