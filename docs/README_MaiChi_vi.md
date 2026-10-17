# Mai Chi — notebook distillation end-to-end

Mở [MaiChi_Clone_Colab.ipynb](../notebooks/MaiChi_Clone_Colab.ipynb) trong Colab/Jupyter có GPU. Bản này tự git clone repo. Bản nhúng toàn bộ source là [MaiChi_Standalone_Colab.ipynb](../notebooks/MaiChi_Standalone_Colab.ipynb). Notebook và phần giải thích trong notebook dùng tiếng Anh; mã nguồn đầy đủ nằm trong notebook và `maichi_pipeline.py`, không dùng base64.

## Cách chạy

1. Kết nối GPU thật với kernel. Notebook kiểm tra CUDA, phép nhân ma trận và convolution; không giả định tên GPU hay dung lượng VRAM.
2. Chạy `MODE="pilot"` trước: 512 câu, 2.000 bước train. Đây là phép thử luồng xử lý, chưa kỳ vọng giọng tự nhiên.
3. Khi luồng chạy ổn, đổi `MODE="full"`: 20.000 câu, 100.000 bước. Chạy lại từ ô cấu hình. Hai chế độ có thư mục riêng.
4. Để so sánh có/không giám sát F0, bật `RUN_ABLATION=True`. Bật `RUN_MEDIUM=True` nếu cần thêm cấu hình lớn hơn. Các lượt này tăng chi phí tương ứng.
5. Full mode bật benchmark 137 câu. Mặc định bỏ UTMOSv2; bật `RUN_UTMOS=True` để cài và chấm thêm. Giá trị chưa chấm luôn để trống, không thay bằng 0.

Nếu dùng Colab với một máy GPU RTX PRO 6000 96GB bên ngoài, đặt `USE_DRIVE=False` nếu máy đó không hỗ trợ mount Google Drive, rồi đặt `PERSISTENT_BASE` vào ổ bền vững. Khi dùng Colab hosted thông thường, để `USE_DRIVE=True`.

Batch size mặc định 32; có thể tăng 64 sau khi quan sát VRAM. Không ép dùng hết 96GB vì tốc độ còn phụ thuộc độ dài câu, CPU, I/O và vòng sinh tuần tự của teacher. Dự trù ít nhất khoảng 60GB trống cho full run; dung lượng thực tế tùy độ dài và cách lưu feature. Kiểm tra riêng quota Drive nếu lưu ở đó.

## Pipeline có gì?

- Tải ZeroTTS ở commit cố định, cố định preset `maichi`, thử ba câu và xuất WAV.
- Tải ZeroBench trước, giữ toàn bộ 137 câu ngoài train/validation.
- Lấy văn bản ZeroSpeech bằng các trang JSON của Hugging Face; kiểm tra SHA của cache khớp bản dataset đã cố định. Không tải audio gốc.
- Loại trùng, sàng lọc overlap nguyên văn/chuỗi con, chia train/val/test theo hash văn bản.
- Tổng hợp lại bằng Mai Chi; kiểm tra waveform, duration, clipping; lưu ngay từng WAV và metadata để tiếp tục khi ngắt phiên.
- Audit nhỏ bằng PhoWhisper, lưu cả transcript đích và transcript nhận dạng.
- Trích F0/voicing bằng WORLD; train student VITS nhỏ, tự học alignment, có tùy chọn giám sát và conditioning F0.
- Lưu generator, discriminator, optimizer, scaler, RNG và số bước. Tiếp tục từ `last.pt`; chọn `best.pt` bằng validation reconstruction loss.
- Export `student_deploy.pt` gồm cả waveform generator, bỏ posterior encoder và discriminator chỉ dùng khi train.
- Sinh WAV teacher/student trên 137 câu; chấm bằng bộ chấm chính thức và xuất bảng, transcript, RTF, số tham số, dung lượng, speaker similarity theo protocol Mai Chi riêng.
- Tạo bảng nghe A/B với thứ tự ngẫu nhiên và đáp án tách riêng.

Notebook tự ghi lại phiên bản thư viện thực tế, tên GPU, revision nguồn và config. Resume giữ trạng thái tối ưu hóa; thứ tự batch không được đảm bảo giống tuyệt đối vì iterator được tạo lại. Đổi kiến trúc/dữ liệu phải dùng tên run mới.

## Kiến trúc và giới hạn

Đây là **sequence-level distillation**: ZeroTTS cung cấp cặp text–audio. Student là **compact VITS có F0 conditioning**, không phải kiến trúc Paradee của Teacher in the Loop và không tái lập shared excitation/iSTFT-head loss của paper. Sự điều chỉnh này tránh yêu cầu ZeroTTS cung cấp các đặc trưng Kokoro mà nó không có.

Student dùng văn bản NFC giữ nguyên dấu tiếng Việt, duration, latent âm học và decoder tạo waveform 24kHz. Pitch predictor có đặc trưng vị trí trong từng token để biểu diễn đường cao độ thay đổi. Đây là thiết kế cần kiểm chứng bằng ablation, chưa có bằng chứng vượt baseline. Không dùng checkpoint VITS tiếng Anh làm checkpoint tiếng Việt.

Cấu hình `small` khoảng **7,2 triệu tham số khi triển khai**, `medium` khoảng **17,9 triệu**, có thay đổi nhỏ theo vocabulary. Số chính xác được tính từ model, bao gồm waveform generator. Checkpoint train lớn hơn vì còn posterior encoder. Không công bố kết quả ONNX/INT8 trong bản này.

Teacher phát 48kHz; student phát 24kHz. Phải ghi rõ khác biệt này khi báo cáo. RTF cần đo trên cùng phần cứng, số thread và điều kiện. Benchmark cố định Mai Chi không phải đánh giá zero-shot cloning; trường speaker similarity với 59 giọng tham chiếu gốc trong báo cáo chính thức không có ý nghĩa cho protocol này. Notebook dùng thêm cosine WavLM giữa student và teacher Mai Chi trên cùng câu, không so trực tiếp nó với SSIM gốc 0,936.

Chọn hyperparameter bằng validation riêng. Benchmark công khai nhỏ và việc loại trùng không loại hết paraphrase hay trùng nguồn tài liệu. UTMOS là đánh giá tự động, không phải human MOS. Audit ba câu không chứng minh tỷ lệ lỗi toàn bộ corpus.

## Đã kiểm tra trước khi giao

- Đã tải trọng số thật và sinh ba WAV Mai Chi trên CPU, 4 thread.
- Thời gian sinh: 3,178 / 2,851 / 3,146 giây; độ dài WAV: 4,32 / 3,60 / 3,60 giây. RTF tương ứng khoảng 0,736 / 0,792 / 0,874. Đây là một lượt thử cho mỗi câu, không phải phép đo tốc độ chuẩn hóa nhiều lần.
- PhoWhisper-small nhận đúng câu tiếng Việt thông thường; câu liệt kê thanh và từ Python còn sai khác. Kết quả này chứa cả lỗi có thể từ ASR; chưa kết luận lỗi phát âm của teacher.
- Đã train ba bước trên WAV thật với GAN, KL, duration và pitch loss; kiểm tra resume, export và reload inference.
- Regression kiểm tra giữ dấu, alignment đơn điệu, gradient, đếm tham số, reload chính xác, ablation và BF16 forward trên CPU.
- Đã tải bộ benchmark và tạo manifest đủ 137 mục; đã thử lấy văn bản thực từ ZeroSpeech.

**Chưa chạy full training trên GPU, chưa hoàn thành benchmark chất lượng 137 câu và chưa chứng minh student giữ được chất lượng teacher.** Xem `MaiChi_validation_evidence.json` để phân biệt kết quả đo thật và các bước còn cần chạy.

## Chạy source ngoài notebook

Cài dependencies theo ô cài đặt trong notebook. Sau đó có thể gọi:

```bash
python maichi_pipeline.py smoke --root /path/to/run --device cpu
python maichi_pipeline.py benchmark --root /path/to/run
python maichi_pipeline.py texts --root /path/to/run --count 20000
python maichi_pipeline.py generate --root /path/to/run --device cuda
python maichi_pipeline.py features --root /path/to/run --workers 4
python maichi_pipeline.py train --root /path/to/run --vits-dir /path/to/vits \
  --name small_pitch --size small --steps 100000 --batch-size 32 --device cuda
```

`bootstrap()` tải VITS về nếu thư mục chưa có và kiểm tra commit. Không dùng một checkout VITS khác revision cho cùng run. Có thể cung cấp JSONL riêng chứa `text`, `id`, `kind`, `source` qua `texts --source-jsonl`; vẫn phải chuẩn bị benchmark trước.

Bộ kiểm tra CPU: `python test_maichi_pipeline.py --vits-dir /path/to/vits`.

## Nguồn và giấy phép

- ZeroTTS: https://github.com/zeroweight-ai/ZeroTTS
- ZeroSpeech: https://huggingface.co/datasets/zeroweight-ai/ZeroSpeech
- ZeroBench: https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS
- VITS: https://github.com/jaywalnut310/vits — bản MIT; giữ `LICENSE_VITS.txt`.
- Teacher in the Loop: https://github.com/MithrilMan/teacher-in-the-loop

Mã mới dùng MIT; ZeroSpeech/ZeroBench dùng CC-BY-NC-4.0 và quyền của nguồn văn bản/giọng vẫn áp dụng. Đây là bản phục vụ nghiên cứu, chưa phải xác nhận quyền thương mại hóa student.
