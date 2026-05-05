# Drone Few-shot Detection & Tracking

<p align="center">
  <strong>Pipeline định vị và theo dõi đối tượng few-shot trong video drone.</strong><br>
  Từ một tập ảnh support nhỏ và một video truy vấn, hệ thống tìm đối tượng mục tiêu trên từng frame và xuất kết quả dạng bounding box.
</p>

<p align="center">
  <code>YOLOE Visual Prompt</code> · <code>MobileCLIP2</code> · <code>Siamese Verifier</code> · <code>KCF Tracking</code> · <code>JSON Submission</code>
</p>

## Tổng Quan

Bài toán được xử lý theo thiết lập few-shot cho video drone:

| Thành phần | Mô tả |
| --- | --- |
| Input | 3 ảnh reference của đối tượng mục tiêu và 1 video drone |
| Nhiệm vụ | Tìm cùng đối tượng đó trên từng frame, kể cả khi thay đổi góc nhìn, tỉ lệ và nền |
| Output | Bounding boxes trong `result/submission.json`, kèm report và biểu đồ thống kê |
| Chiến lược chính | Sinh proposal bằng YOLOE, xác thực bằng mô hình similarity, gộp score rồi ổn định bằng KCF tracker |

<p align="center">
  <img src="assets/fewshot_input_output_person1.png" alt="Few-shot input support images, drone query frame, and localized output" width="96%">
</p>

## Pipeline

<p align="center">
  <img src="assets/pipeline.png" alt="Drone few-shot detection and tracking pipeline" width="96%">
</p>

1. `preprocessing.py` tạo YOLOE visual prompt embeddings từ ảnh reference và các background augmentation.
2. `predict.py` dùng YOLOE để sinh object proposals trên từng query frame.
3. MobileCLIP2 so khớp từng proposal crop với các support crops.
4. Siamese branch bổ sung similarity score phụ nếu có checkpoint.
5. `fusion.py` chuẩn hóa và gộp detector score, CLIP score và Siamese score.
6. KCF tracker duy trì bbox ổn định giữa các lần detector re-init.
7. Kết quả cuối được xuất thành JSON, kèm debug frames, report và plot nếu bật tùy chọn tương ứng.

## Cấu Trúc Repo

```text
.
├── assets/                         # Hình minh họa cho README
├── data/                           # Dataset local, ignore khỏi git
├── models/                         # Model weights local, ignore khỏi git
├── preprocessed_data/              # VPE/features sinh ra từ preprocessing
├── result/                         # Submission, report, plot, checkpoint
├── preprocessing.py                # Tạo reference crops, VPE, Siamese embeddings
├── predict.py                      # Hybrid inference + tracker + reporting
├── predict.sh                      # Runner end-to-end cho public test
├── fusion.py                       # Chuẩn hóa và weighted fusion score
├── tracker_adapter.py              # Chọn backend KCF tracker
├── check_env.py                    # Kiểm tra môi trường chạy
├── setup_env.sh                    # Cài dependency hỗ trợ
├── siamese/                        # Train/evaluate Siamese verifier
└── yoloe/                          # Helper scripts cho YOLO/YOLOE
```

## Môi Trường

Khuyến nghị dùng môi trường Python có CUDA nếu chạy full inference.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_env.py
```

Hoặc dùng script có sẵn:

```bash
./setup_env.sh
```

Các dependency chính:

| Dependency | Vai trò |
| --- | --- |
| `torch`, `torchvision` | Inference model và Siamese training |
| `ultralytics` | Load YOLOE/YOLO |
| `opencv-contrib-python` | KCF tracker |
| `timm` | Backbone cho Siamese |
| `scikit-learn` | Tiện ích preprocessing và thống kê màu |
| `matplotlib` | Vẽ biểu đồ kết quả inference |

`predict.py` import MobileCLIP qua local checkout `ml-mobileclip/open_clip/src`. Thư mục `ml-mobileclip/` đang bị ignore khỏi git nhưng vẫn phải tồn tại trên máy chạy.

## Artifact Cần Chuẩn Bị

Đặt model weights trong `models/`:

```text
models/
├── yolov8n.pt
├── yoloe-11l-seg.pt
├── yoloe-11s-seg.pt                  # fallback nếu thiếu 11l
└── mobileclip2_image_encoder_fp16.pt
```

Siamese checkpoint, nếu dùng:

```text
result/siamese/best.pt
```

Layout public test:

```text
data/public_test/samples/<SampleID>/
├── drone_video.mp4
└── object_images/
    ├── *.jpg
    └── *.png
```

Layout train data:

```text
data/train/
├── annotations/annotations.json
└── samples/<SampleID>/
    ├── drone_video.mp4
    └── object_images/
```

## Chạy Nhanh

Chạy toàn bộ pipeline public test:

```bash
./predict.sh
```

Smoke test nhanh:

```bash
LIMIT_SAMPLES=1 MAX_FRAMES=120 ./predict.sh
```

`predict.sh` thực hiện hai bước:

1. Preprocess `data/public_test` vào `preprocessed_data/public_test`.
2. Chạy hybrid inference và ghi `result/submission.json`.

Output mặc định:

```text
result/submission.json
result/submission_report.txt
result/submission_stats.png
```

## Cấu Hình Inference

`predict.sh` expose các tham số chính qua environment variables:

```bash
YOLOE_CONF=0.001 \
TOP_K=24 \
FUSED_THRESHOLD=0.52 \
W_DET=0.30 \
W_CLIP=0.35 \
W_SIAM=0.35 \
SIMILARITY_ADD_THRESHOLD=0.80 \
NUM_BACKGROUNDS=4 \
MIN_AUG_SCALE=0.04 \
MAX_AUG_SCALE=0.20 \
VPE_IMGSZ=640 \
./predict.sh
```

Best proxy config hiện tại:

| Parameter | Value |
| --- | --- |
| `yoloe_conf` | `0.001` |
| `top_k_proposals` | `24` |
| `fused_accept_threshold` | `0.52` |
| `w_det` / `w_clip` / `w_siam` | `0.30` / `0.35` / `0.35` |
| `similarity_add_threshold` | `0.80` |
| `max_reference_samples` | `20` |
| `crop_padding_ratio` | `0.04` |
| `support_aggregation` | `mean` |

Kết quả proxy trên public test hiện tại:

| Metric | Value |
| --- | ---: |
| Mean detection rate | `59.76%` |
| Minimum detection rate | `38.24%` |
| Mean fused score | `0.7175` |

Đây là proxy score vì repo hiện chưa có ground-truth IoU/mAP cho public-test split.

## Preprocessing Thủ Công

```bash
python preprocessing.py \
  --dataset data/public_test \
  --output-dir preprocessed_data/public_test \
  --yoloe-weights models/yoloe-11l-seg.pt \
  --yolov8-weights models/yolov8n.pt \
  --siamese-checkpoint result/siamese/best.pt \
  --num-backgrounds 4 \
  --min-aug-scale 0.04 \
  --max-aug-scale 0.20 \
  --vpe-imgsz 640
```

Preprocess một sample để debug:

```bash
python preprocessing.py \
  --dataset data/public_test \
  --output-dir preprocessed_data/debug_public_test \
  --yoloe-weights models/yoloe-11l-seg.pt \
  --limit-samples 1
```

## Inference Thủ Công

Chạy với cấu hình mặc định:

```bash
python predict.py \
  --preprocessed-dir preprocessed_data/public_test \
  --output-json result/submission.json
```

Reproduce best proxy config:

```bash
python predict.py \
  --preprocessed-dir preprocessed_data/public_test \
  --output-json result/submission.json \
  --yoloe-weights models/yoloe-11l-seg.pt \
  --clip-encoder-path models/mobileclip2_image_encoder_fp16.pt \
  --siamese-checkpoint result/siamese/best.pt \
  --yoloe-conf 0.001 \
  --top-k 24 \
  --fused-threshold 0.52 \
  --w-det 0.30 \
  --w-clip 0.35 \
  --w-siam 0.35 \
  --similarity-add-threshold 0.80 \
  --max-reference-samples 20 \
  --crop-padding-ratio 0.04
```

Lưu debug frames để kiểm tra trực quan:

```bash
python predict.py \
  --preprocessed-dir preprocessed_data/public_test \
  --output-json result/debug_subset.json \
  --limit-videos 1 \
  --max-frames 1200 \
  --save-frames \
  --frames-output-dir result/debug_frames
```

## Siamese Verifier

Siamese verifier được train từ các crop có annotation và dùng như một nhánh similarity phụ trong inference.

Train:

```bash
python siamese/train.py \
  --config siamese/train_config.yaml \
  --force-rebuild-cache
```

Evaluate:

```bash
python siamese/evaluate.py \
  --checkpoint result/siamese/best.pt \
  --config siamese/train_config.yaml \
  --output result/siamese/evaluation.json
```

Cấu hình train hiện tại:

| Setting | Value |
| --- | --- |
| Backbone | `resnet18` |
| Embedding dimension | `128` |
| Loss | `triplet` |
| Split strategy | `stratified` |
| Learning rate | `1e-5` |
| Dropout | `0.4` |
| Weight decay | `1e-3` |
| Early stopping patience | `5` |

Kết quả checkpoint mới nhất trong quá trình tuning:

| Metric | Value |
| --- | ---: |
| `val_loss` | `0.554510` |
| `val_acc` | `0.495500` |
| `val_precision` | `0.495500` |
| `val_recall` | `1.000000` |
| `val_f1` | `0.662655` |

Siamese branch hiện hữu ích như một soft positive boost, chưa đủ tin cậy để dùng như verifier độc lập.

## Ghi Chú Vận Hành

- Default `yoloe_conf=0.001` và `top_k=24` ưu tiên recall cao. Nếu metric phạt false positive nặng, thử `YOLOE_CONF=0.005 TOP_K=16`.
- Cần `opencv-contrib-python` để có KCF. Tracker policy hiện ưu tiên `kcf`, sau đó `legacy_kcf`.
- `data/`, `models/`, `preprocessed_data/`, `result/`, và `ml-mobileclip/` là local artifacts, hầu hết được ignore khỏi git.
- `MobileCLIP2-S0` có thể in warning lúc khởi tạo model; runtime vẫn load image encoder local từ `models/mobileclip2_image_encoder_fp16.pt`.

## Hướng Phát Triển

1. Tạo một validation subset nhỏ có ground truth để đo IoU, precision, recall và F1.
2. Inspect debug frames cho các class khó như `CardboardBox_0` và `LifeJacket_*`.
3. Calibrate `YOLOE_CONF`, `TOP_K`, và `FUSED_THRESHOLD` theo chi phí false positive thật.
4. Cải thiện Siamese bằng hard negative mining hoặc protocol validation mạnh hơn trước khi dùng như strict verifier.
