# Mai Chi v3 — T4 tạo dữ liệu, RTX 6000 train riêng

Có đúng **hai notebook thực thi chính**, theo thứ tự:

1. [01_MaiChi_Data_T4.ipynb](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/01_MaiChi_Data_T4.ipynb): chạy trên T4, tạo dữ liệu giọng Mai Chi, trích đặc trưng và lưu shard lên Google Drive.
2. [02_MaiChi_Train_RTX6000.ipynb](https://colab.research.google.com/github/Huyvux12/Ancaptritue/blob/main/notebooks/02_MaiChi_Train_RTX6000.ipynb): chạy trên RTX PRO 6000 96GB, tải shard xuống SSD, train, export và benchmark. Không tạo lại corpus, không tải teacher để train.

Các link notebook clone/end-to-end cũ chỉ còn trang chỉ dẫn sang hai notebook này. Standalone VITS là baseline cũ, không phải luồng mới.

## Notebook 01 — T4

Giữ `MODE="pilot"` để thử 512 câu; dùng `MODE="full"` cho 20.000 câu. Mount Drive từ ô cấu hình, giữ `DRIVE_PARENT` thống nhất với notebook 02. Mặc định thư mục gốc là `/content/drive/MyDrive/MaiChi_Distillation_v03`.

Teacher bắt đầu batch 2. Có thể thử 4 sau khi đo tốc độ và VRAM. **Batch thật diễn ra ở text encoder, global transformer và codec; local sampler vẫn xử lý từng câu** do giới hạn graph ONNX công khai. Batch 2 trong sampler gốc là hai nhánh CFG của cùng một câu, không phải hai câu. Không thay sampling/precision để giả lập batch; chưa có số đo chứng minh tăng tốc trên T4.

Notebook tự kiểm tra CUDA và synthesize batch thật trước khi lấy corpus. Mỗi câu có RNG riêng theo ID, nhưng sai khác số học giữa phần cứng/batch vẫn có thể đổi kết quả lấy mẫu. Batch gặp OOM sẽ thử giảm kích thước; lỗi graph/dữ liệu được báo để sửa, không tự chuyển CPU.

Luồng dữ liệu: nghe thử teacher → giữ riêng toàn bộ 137 câu benchmark → lấy transcript có revision cố định → loại overlap/trùng → synthesize Mai Chi → audit ASR → trích WORLD F0/mel → vocabulary chỉ từ train. Lọc exact/substring không loại hết paraphrase hay trùng nguồn.

WAV/feature đang xử lý nằm trên SSD T4. Sau mỗi block 128 văn bản, các shard hoàn tất và manifest checksum được sao lưu lên Drive. Rerun cùng cấu hình sẽ khôi phục dữ liệu, không tổng hợp lại WAV đã lưu. Nếu VM mất đột ngột, tối đa block chưa sync có thể phải làm lại. Feature extraction chưa publish xong có thể phải chạy lại trên CPU; WAV đã sync được tái dùng.

Full mode tạo thêm 137 WAV teacher làm tham chiếu để RTX không cần teacher. Chờ ô cuối in **READY FOR NOTEBOOK 02**. Chỉ lúc đó bundle mới có trạng thái COMPLETE. Cần kiểm tra dung lượng SSD và quota Drive; dự trù khoảng 60GB hoặc hơn tùy dữ liệu/cache/snapshot.

## Notebook 02 — RTX 6000

Chọn đúng `MODE` và `DRIVE_PARENT`. Nếu runtime ngoài Colab không mount Drive bằng `google.colab`, đặt `USE_DRIVE=False` và trỏ đến thư mục Drive đã mount/sync sẵn. Notebook tự chọn đúng commit đã dùng tạo dữ liệu, xác minh SHA-256 rồi giải nén xuống SSD RTX. Không đọc hàng triệu lượt file nhỏ trực tiếp từ Drive.

Mặc định batch 32, BF16 nếu GPU hỗ trợ; nếu không dùng FP16. Kiểm tra tên GPU/VRAM và phép tính CUDA thật. `TRAIN_WORKERS=0` khi runtime không cho multiprocessing. Chưa đo hiệu năng/VRAM trên GPU này.

Train hai giai đoạn: vocoder 1,72M tham số trên mel/WAV teacher với mel/STFT/GAN/feature-matching loss, rồi đóng băng vocoder và train acoustic model DiT/OT-CFM. Pilot: 2.000 + 4.000 bước; full: 100.000 + 200.000 bước. Đây là ngân sách xuất phát, không phải cam kết hội tụ/chất lượng.

Đường dẫn corpus được gắn lại theo máy mới. Hash corpus được chuẩn hóa để việc chuyển thư mục không làm hỏng resume. Dữ liệu thay đổi sau khi train đã bắt đầu cần tạo run/root mới.

Checkpoint/optimizer/RNG được lưu ở SSD và sao lưu Drive mỗi 10 phút, cùng lúc kết thúc mỗi stage. Backup giữ đối tượng được tham chiếu bởi hai snapshot thành công gần nhất, có best/last/deploy. Khi VM mất, tiến độ sau checkpoint đã sao lưu có thể mất. Iterator dữ liệu khởi tạo lại nên không replay batch chính xác tuyệt đối. Đổi batch, precision, source hoặc dữ liệu phải dùng root/run mới; notebook v3 không tiếp tục trực tiếp checkpoint v2 khác source.

Cuối cùng xuất `student_deploy.pt`, so 4/8/16 bước suy luận và chấm dual-ASR, tùy chọn UTMOS, speaker similarity Mai Chi riêng, rồi tạo worksheet nghe mù. Kết quả đánh giá được lưu lên Drive riêng. Nếu đánh giá bị ngắt, có thể chạy lại mà không train lại.

## Ngân sách và đánh giá đúng cách

Student vẫn là text encoder 3 block width 128 + DiT 10 block width 192/4 head, học alignment MAS và duration, mel OT-CFM, F0/voicing loss phụ, vocoder convolution kiểu HiFi-GAN. Đây là triển khai nhỏ tự train, tham khảo F5-TTS/Matcha-TTS; không phải trọng số chính thức hoặc Teacher in the Loop online.

Vocabulary 256: acoustic **7.793.635** + vocoder **1.724.929** = **9.518.564 tham số**. Vocabulary thực được kiểm tra; init/export/reload chặn tổng >=10M. 160 buffer mean/std cũng nằm trong guard tensor export. Discriminator, optimizer và bản sao EMA chỉ dùng khi train. Không có codec/vocoder lớn ẩn phía sau student.

Teacher phát 48kHz, student 24kHz. Teacher benchmark được đo single-request trên T4, student trên RTX: **không suy ra speedup từ hai RTF khác phần cứng**; CSV ghi rõ GPU/protocol. Chất lượng so trên cùng text; fixed Mai Chi không phải zero-shot cloning các giọng gốc. WavLM teacher–student là protocol riêng, không so trực tiếp với SSIM leaderboard. UTMOS không phải human MOS. Không tune theo 137 câu test.

## Đã kiểm chứng

CPU với teacher thật: batch 2, hai độ dài khác nhau; batch 1 khớp API gốc; codec batch có padding khớp giải mã riêng; corpus writer và cache resume. Kiểm tra shard/đường dẫn/checksum, từ chối partial/corrupt bundle, backup và retention. Đã train ngắn trên WAV thật, chuyển data/checkpoint sang thư mục khác rồi resume optimizer từ bước acoustic 2 lên 3, export và synthesize hữu hạn. Regression student vẫn kiểm tra budget, gradient, alignment và reload.

**Chưa chạy T4/RTX thực tế, chưa full train hoặc benchmark chất lượng student đã hội tụ.** Mô phỏng Drive bằng thư mục cục bộ chỉ kiểm tra tính đúng của cơ chế truyền file, chưa xác minh hành vi FUSE/quota Drive. Xem [evidence v3](../validation/MaiChi_split_v3_evidence.json). Không thể cam kết đây là model tốt nhất hoặc ngang model lớn chỉ từ kiến trúc và smoke test.
