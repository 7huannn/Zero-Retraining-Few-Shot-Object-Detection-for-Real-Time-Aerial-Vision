# Work Summary - Drone Few-shot Detection/Tracking

File nay tom tat ngan gon nhung thay doi da lam de ban khac co the nam duoc da sua gi, tune gi va ket qua hien tai.

## Muc tieu

Toi uu pipeline few-shot detection/tracking tren video drone:

- Input: 3 anh reference + 1 video.
- Pipeline chinh: YOLOE proposals -> MobileCLIP similarity -> Siamese similarity -> fusion score -> KCF tracker -> JSON output.
- Muc tieu gan day: tang coverage/detection rate tren public test khi chua co ground truth IoU/mAP chinh thuc.

## Nhung thay doi chinh

### Inference (`predict.py`, `predict.sh`)

- Dung seed chung tu `utils.seed_everything()` va bo conflict `cudnn.benchmark=True` de ket qua on dinh hon.
- Them/ho tro cac tham so inference moi:
  - `--disable-siamese`
  - `--similarity-add-threshold`
  - `--max-reference-samples`
  - `--crop-padding-ratio`
- Neu khong co Siamese checkpoint hoac disable Siamese, weight Siamese duoc chuyen sang CLIP.
- Log them diem thanh phan cho accepted detections:
  - detection score
  - CLIP score
  - Siamese score
- `predict.sh` da expose cac bien env de tune nhanh: `YOLOE_CONF`, `TOP_K`, `FUSED_THRESHOLD`, `W_DET`, `W_CLIP`, `W_SIAM`, `SIMILARITY_ADD_THRESHOLD`, `DISABLE_SIAMESE`, `NUM_BACKGROUNDS`, `MIN_AUG_SCALE`, `MAX_AUG_SCALE`, `VPE_IMGSZ`.

### Preprocessing (`preprocessing.py`)

- Tang so background augmentation mac dinh: `2 -> 4`.
- Doi scale paste object cho VPE: `0.05-0.15 -> 0.04-0.20`.
- Them `--vpe-imgsz`, hien tai dung `640`.
- Co the encode va luu Siamese reference embeddings neu truyen `--siamese-checkpoint`.
- Public test da preprocess vao:

```text
preprocessed_data/public_test_20260428
```

### Siamese (`siamese/*`)

- Them `freeze_backbone` de freeze backbone ResNet18 va chi train projection head.
- Them augmentation manh hon: crop scale rong hon, rotation, blur, random erasing.
- Them loss `TripletBatchHardLoss`.
- Them split strategy: `random`, `stratified`, `leave_one_class_out`.
- Train/evaluate co them metrics: accuracy, precision, recall, F1, mean cosine.
- Them early stopping va checkpoint theo metric cau hinh.
- Config hien tai trong `siamese/train_config.yaml` dung:
  - `freeze_backbone: true`
  - `loss: triplet`
  - `margin: 0.5`
  - `split_strategy: stratified`
  - `lr: 0.00001`
  - `dropout: 0.4`

## Cau hinh inference tot nhat hien tai

Day la default hien tai trong `predict.py`/`predict.sh` theo best proxy run:

```yaml
yoloe_conf: 0.001
top_k_proposals: 24
w_det: 0.30
w_clip: 0.35
w_siam: 0.35
fused_accept_threshold: 0.52
min_clip_similarity: 0.10
min_siam_similarity: 0.10
similarity_add_threshold: 0.80
max_reference_samples: 20
crop_padding_ratio: 0.04
support_aggregation: mean
tracker_reinit_interval: 10
tracker_bonus: 0.04
```

Lenh chay nhanh:

```bash
./predict.sh
```

Hoac chay ro rang:

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

## Ket qua hien tai

Output chinh sau cleanup:

```text
result/submission.json
result/submission_report.txt
result/submission_stats.png
```

Best proxy tren public test:

```text
Mean detection rate = 59.76%
Min detection rate  = 38.24%
Mean fused score    = 0.7175
Proxy score         = 0.4288
```

So voi output cu truoc khi tune:

```text
Old mean detection rate = 44.40%
Old min detection rate  = 22.01%
Old mean fused score    = 0.6707
Old proxy score         = 0.2978
```

## Diem can luu y

- Chua co ground truth, nen day la proxy result, khong phai mAP/IoU chinh thuc.
- Config hien tai mo detector rat rong (`yoloe_conf=0.001`, `top_k=24`), giup tang coverage nhung co rui ro false positives.
- Siamese checkpoint moi van yeu:
  - validation accuracy khoang `49.55%`
  - precision khoang `49.55%`
  - recall `100%`
  - mean cosine cao, score Siamese trong inference thuong gan `0.97-0.98`
- Vi vay Siamese hien tai co xu huong boost positive gan nhu hang so, chua phai verifier dang tin.

## Cau hinh an toan hon neu false positive bi phat nang

```yaml
yoloe_conf: 0.005
top_k: 16
fused_threshold: 0.52
w_det: 0.30
w_clip: 0.35
w_siam: 0.35
similarity_add_threshold: 0.80
```

Ket qua proxy cua cau hinh nay:

```text
Mean detection rate = 48.00%
Min detection rate  = 24.26%
Mean fused score    = 0.7063
Proxy score         = 0.3391
```

## Nen lam tiep

1. Tao hoac lay ground truth nho de tinh IoU/precision/recall/F1 that.
2. Inspect debug frames, dac biet `CardboardBox_0`, de xem co drift/false positive khong.
3. Neu GT cho thay false positive cao, giam `top_k`, tang `yoloe_conf`, hoac disable/giam weight Siamese.
4. Neu tiep tuc Siamese, can fix collapse positive bang better mining/loss/calibration truoc khi tin no la verifier.

