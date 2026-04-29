# Drone Few-shot Detection & Tracking

Repo này chạy pipeline few-shot object detection/tracking cho video drone. Mỗi sample có 3 ảnh reference và 1 video; hệ thống tìm và theo dõi object tương ứng trong video, rồi xuất JSON submission.

Code thực tế nằm trong thư mục `ban_git/`. File tổng hợp lịch sử tuning gần nhất nằm ở `../task.md`.

## Pipeline

1. `preprocessing.py` tạo YOLOE visual prompt embedding (VPE) từ ảnh reference và video background.
2. `predict.py` dùng YOLOE sinh proposals trên từng frame.
3. MobileCLIP tính similarity giữa proposal crop và support crops.
4. Siamese verifier tính similarity phụ.
5. `fusion.py` gộp detector/CLIP/Siamese scores.
6. KCF tracker duy trì bbox giữa các lần detector re-init.
7. Xuất `result/submission.json`, report text và plot.

## Cấu Trúc Chính

```text
ban_git/
  predict.py                  # Inference hybrid + tracker + report
  predict.sh                  # Script end-to-end preprocess + inference
  preprocessing.py            # Build VPE, support crops, optional Siamese embeddings
  fusion.py                   # Score normalization/fusion helpers
  tracker_adapter.py          # KCF tracker backend handling
  utils.py                    # IO/path/seed helpers
  siamese/
    train.py                  # Train Siamese verifier
    evaluate.py               # Evaluate Siamese checkpoint
    data.py                   # Crop cache + pair dataset + split strategy
    siamese_model.py          # ResNet Siamese, contrastive/triplet loss
  yoloe/                      # YOLO/YOLOE helper scripts
```

## Yêu Cầu Môi Trường

Khuyến nghị dùng môi trường Python có CUDA nếu muốn chạy full inference nhanh.

```bash
pip install -r requirements.txt
python check_env.py
```

Các dependency quan trọng:

- `torch`, `torchvision`
- `ultralytics`
- `opencv-contrib-python` để có KCF tracker
- `timm`
- `scikit-learn`
- `matplotlib`

`predict.py` bootstrap local `ml-mobileclip/open_clip/src` để import `open_clip`. Thư mục `ml-mobileclip/` đang được `.gitignore` vì là artifact/vendor local, nhưng runtime vẫn cần nó tồn tại trên máy chạy.

## Dữ Liệu Và Model Cần Có

Models đặt trong `models/`:

```text
models/yolov8n.pt
models/yoloe-11l-seg.pt
models/yoloe-11s-seg.pt              # fallback nếu thiếu 11l
models/mobileclip2_image_encoder_fp16.pt
```

Siamese checkpoint hiện tại:

```text
result/siamese/best.pt
```

Public test expected layout:

```text
data/public_test/samples/<SampleID>/
  drone_video.mp4
  object_images/
    *.jpg / *.png
```

Train data expected layout:

```text
data/train/annotations/annotations.json
data/train/samples/<SampleID>/
  drone_video.mp4
  object_images/
```

`data/`, `models/`, `preprocessed_data/`, `result/`, `ml-mobileclip/` đều là local artifacts và hầu hết được ignore khỏi git.

## Chạy End-to-end

Lệnh chuẩn:

```bash
./predict.sh
```

Smoke test nhanh:

```bash
LIMIT_SAMPLES=1 MAX_FRAMES=120 ./predict.sh
```

`predict.sh` sẽ:

1. preprocess `data/public_test` vào `preprocessed_data/public_test`
2. chạy inference và ghi `result/submission.json`

Các biến env có thể override:

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

## Best Proxy Config Hiện Tại

Do chưa có ground truth IoU/mAP, cấu hình hiện tại được chọn theo proxy trên public test:

- mean detection rate
- min detection rate giữa 6 video
- mean fused score
- proxy = detection rate x fused score

Best full-run hiện tại:

```text
Mean detection rate: 59.76%
Min detection rate: 38.24%
Mean fused score: 0.7175
```

Best config:

```yaml
preprocessed_dir: preprocessed_data/public_test_20260428
yoloe_conf: 0.001
top_k_proposals: 24
fused_accept_threshold: 0.52
w_det: 0.30
w_clip: 0.35
w_siam: 0.35
similarity_add_threshold: 0.80
max_reference_samples: 20
crop_padding_ratio: 0.04
support_aggregation: mean
```

Command reproduce:

```bash
python predict.py \
  --preprocessed-dir preprocessed_data/public_test_20260428 \
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

Rủi ro: `yoloe_conf=0.001` và `top_k=24` mở detector rất rộng. Cấu hình này tăng coverage mạnh theo proxy, nhưng có thể tăng false positives nếu metric chính thức phạt FP nặng.

Biến thể an toàn hơn đã test:

```yaml
yoloe_conf: 0.005
top_k_proposals: 16
fused_accept_threshold: 0.52
```

Kết quả biến thể này: mean detection rate `48.00%`, mean fused score `0.7063`.

## Preprocessing Thủ Công

Preprocess public test với VPE 4 backgrounds và Siamese embeddings:

```bash
python preprocessing.py \
  --dataset data/public_test \
  --output-dir preprocessed_data/public_test_20260428 \
  --yoloe-weights models/yoloe-11l-seg.pt \
  --yolov8-weights models/yolov8n.pt \
  --siamese-checkpoint result/siamese/best.pt \
  --num-backgrounds 4 \
  --min-aug-scale 0.04 \
  --max-aug-scale 0.20 \
  --vpe-imgsz 640
```

## Inference Thủ Công

Chạy full:

```bash
python predict.py \
  --preprocessed-dir preprocessed_data/public_test_20260428 \
  --output-json result/submission.json
```

Giới hạn video/frame để debug:

```bash
python predict.py \
  --preprocessed-dir preprocessed_data/public_test_20260428 \
  --output-json result/debug_subset.json \
  --limit-videos 1 \
  --max-frames 1200
```

Lưu debug frames:

```bash
python predict.py \
  --preprocessed-dir preprocessed_data/public_test_20260428 \
  --output-json result/debug_frames_subset.json \
  --limit-videos 1 \
  --max-frames 1200 \
  --save-frames \
  --frames-output-dir result/debug_frames
```

## Siamese Training

Siamese hiện tại đã được cải tiến về training loop nhưng checkpoint vẫn chưa tốt.

Config chính:

```text
siamese/train_config.yaml
```

Các thay đổi đang có:

- freeze ResNet backbone
- train projection head
- dropout `0.4`
- weight decay `1e-3`
- lr `1e-5`
- triplet batch-hard loss
- early stopping
- metrics: accuracy, precision, recall, F1
- split strategy: `stratified`

Train lại:

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

Kết quả checkpoint mới nhất trong quá trình tuning:

```text
val_loss      = 0.554510
val_acc       = 0.495500
val_precision = 0.495500
val_recall    = 1.000000
val_f1        = 0.662655
```

Nhận định: Siamese vẫn gần như predict positive toàn bộ, nên chưa phải verifier đáng tin cậy. Trong best proxy config, Siamese score hoạt động giống một positive boost hơn là bộ lọc chính xác.

## Output Hiện Giữ Lại

Sau cleanup, `result/` chỉ giữ các file gần nhất/tối thiểu:

```text
result/submission.json
result/submission_report.txt
result/submission_stats.png
result/siamese/best.pt
result/siamese/history.json
result/siamese/metrics.json
result/siamese/evaluation_20260428.json
```

Các run logs, experiment outputs, old submissions, demo plots và checkpoint backups đã bị xóa khỏi `result/`.

## Git Ignore / Artifact Policy

`.gitignore` hiện bỏ qua:

- datasets trong `data/`
- model weights trong `models/`
- preprocessed caches
- generated outputs trong `result/`
- local/vendor `ml-mobileclip/`
- demo artifacts `fewshot_input_output_person1.*`
- `*.pt`, `*.pdparams`, cache Python

Nếu clone repo mới, cần tự chuẩn bị lại:

- model weights
- data
- `ml-mobileclip/`
- preprocessed cache nếu muốn chạy ngay không preprocess lại

## Tracker Backend

`predict.py` dùng `tracker_adapter.py`, ưu tiên strict KCF:

1. `kcf`
2. `legacy_kcf`

Nếu tracker init bằng backend khác trong run chuẩn, code sẽ raise lỗi. Cần `opencv-contrib-python` để có KCF.

## Known Issues

- Chưa có ground truth public test, nên best hiện tại là best theo proxy, không phải best mAP/IoU.
- Best proxy config có thể over-detect vì detector conf rất thấp.
- Siamese checkpoint hiện tại chưa phân biệt tốt positive/negative.
- `MobileCLIP2-S0` có warning lúc create model rằng chưa load pretrained, nhưng sau đó code load local encoder `models/mobileclip2_image_encoder_fp16.pt`; warning này không fatal.
- `ml-mobileclip/` bị ignore nhưng vẫn cần cho runtime.

## Next Steps

1. Tạo ground truth nhỏ cho public/test để đo IoU precision/recall/F1.
2. Inspect debug frames cho best config, đặc biệt `CardboardBox_0` và `LifeJacket_*`.
3. Nếu leaderboard phạt FP nặng, thử lại biến thể an toàn `conf=0.005/top_k=16`.
4. Cải thiện Siamese bằng hard negative mining, threshold calibration, hoặc loss khác.
5. Thêm config dump mỗi lần inference để reproduce dễ hơn.
