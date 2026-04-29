#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SIAMESE_CHECKPOINT="${SIAMESE_CHECKPOINT:-result/siamese/best.pt}"
YOLOE_CONF="${YOLOE_CONF:-0.001}"
TOP_K="${TOP_K:-24}"
FUSED_THRESHOLD="${FUSED_THRESHOLD:-0.52}"
W_DET="${W_DET:-0.30}"
W_CLIP="${W_CLIP:-0.35}"
W_SIAM="${W_SIAM:-0.35}"
SIMILARITY_ADD_THRESHOLD="${SIMILARITY_ADD_THRESHOLD:-0.80}"
NUM_BACKGROUNDS="${NUM_BACKGROUNDS:-4}"
MIN_AUG_SCALE="${MIN_AUG_SCALE:-0.04}"
MAX_AUG_SCALE="${MAX_AUG_SCALE:-0.20}"
VPE_IMGSZ="${VPE_IMGSZ:-640}"

preprocess_args=(
  --dataset data/public_test \
  --output-dir preprocessed_data/public_test \
  --yoloe-weights models/yoloe-11l-seg.pt \
  --limit-samples "$LIMIT_SAMPLES" \
  --num-backgrounds "$NUM_BACKGROUNDS" \
  --min-aug-scale "$MIN_AUG_SCALE" \
  --max-aug-scale "$MAX_AUG_SCALE" \
  --vpe-imgsz "$VPE_IMGSZ"
)
if [[ -f "$SIAMESE_CHECKPOINT" ]]; then
  preprocess_args+=(--siamese-checkpoint "$SIAMESE_CHECKPOINT")
fi
python preprocessing.py "${preprocess_args[@]}"

predict_siamese_args=()
if [[ "${DISABLE_SIAMESE:-0}" == "1" || ! -f "$SIAMESE_CHECKPOINT" ]]; then
  predict_siamese_args+=(--disable-siamese)
else
  predict_siamese_args+=(--siamese-checkpoint "$SIAMESE_CHECKPOINT")
fi

python predict.py \
  --preprocessed-dir preprocessed_data/public_test \
  --output-json result/submission.json \
  --limit-videos "$LIMIT_SAMPLES" \
  --max-frames "$MAX_FRAMES" \
  --yoloe-weights models/yoloe-11l-seg.pt \
  --clip-encoder-path models/mobileclip2_image_encoder_fp16.pt \
  --yoloe-conf "$YOLOE_CONF" \
  --top-k "$TOP_K" \
  --fused-threshold "$FUSED_THRESHOLD" \
  --w-det "$W_DET" \
  --w-clip "$W_CLIP" \
  --w-siam "$W_SIAM" \
  --similarity-add-threshold "$SIMILARITY_ADD_THRESHOLD" \
  "${predict_siamese_args[@]}"
