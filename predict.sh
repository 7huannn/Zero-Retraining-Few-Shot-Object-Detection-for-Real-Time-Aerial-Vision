#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
MATCHER="${MATCHER:-mobileclip2}"
SIAMESE_CHECKPOINT="${SIAMESE_CHECKPOINT:-result/siamese/best.pt}"
YOLOE_CONF="${YOLOE_CONF:-0.001}"
TOP_K="${TOP_K:-24}"
FUSED_THRESHOLD="${FUSED_THRESHOLD:-0.52}"
W_DET="${W_DET:-0.30}"
W_MATCH="${W_MATCH:-${W_CLIP:-0.35}}"
MATCH_THRESHOLD="${MATCH_THRESHOLD:-${CLIP_THRESHOLD:-0.10}}"
SIMILARITY_ADD_THRESHOLD="${SIMILARITY_ADD_THRESHOLD:-0.80}"
NUM_BACKGROUNDS="${NUM_BACKGROUNDS:-4}"
MIN_AUG_SCALE="${MIN_AUG_SCALE:-0.04}"
MAX_AUG_SCALE="${MAX_AUG_SCALE:-0.20}"
VPE_IMGSZ="${VPE_IMGSZ:-640}"

preprocess_args=(
  --dataset data/public_test
  --output-dir preprocessed_data/public_test
  --yoloe-weights models/yoloe-11l-seg.pt
  --limit-samples "$LIMIT_SAMPLES"
  --num-backgrounds "$NUM_BACKGROUNDS"
  --min-aug-scale "$MIN_AUG_SCALE"
  --max-aug-scale "$MAX_AUG_SCALE"
  --vpe-imgsz "$VPE_IMGSZ"
)
python preprocessing.py "${preprocess_args[@]}"

run_inference() {
  local matcher="$1"
  local output_json="$2"
  local frames_dir="result/debug_frames_${matcher}"

  if [[ "$matcher" == "siamese" && ! -f "$SIAMESE_CHECKPOINT" ]]; then
    echo "Missing Siamese checkpoint: $SIAMESE_CHECKPOINT" >&2
    exit 2
  fi

  local inference_args=(
    --preprocessed-dir preprocessed_data/public_test
    --output-json "$output_json"
    --limit-videos "$LIMIT_SAMPLES"
    --max-frames "$MAX_FRAMES"
    --yoloe-weights models/yoloe-11l-seg.pt
    --matcher "$matcher"
    --clip-encoder-path models/mobileclip2_image_encoder_fp16.pt
    --siamese-checkpoint "$SIAMESE_CHECKPOINT"
    --yoloe-conf "$YOLOE_CONF"
    --top-k "$TOP_K"
    --fused-threshold "$FUSED_THRESHOLD"
    --w-det "$W_DET"
    --w-match "$W_MATCH"
    --match-threshold "$MATCH_THRESHOLD"
    --similarity-add-threshold "$SIMILARITY_ADD_THRESHOLD"
    --frames-output-dir "$frames_dir"
  )

  python inference.py "${inference_args[@]}"
}

case "$MATCHER" in
  mobileclip2|siamese)
    run_inference "$MATCHER" "result/submission.json"
    ;;
  all|compare)
    run_inference mobileclip2 "result/submission_mobileclip2.json"
    run_inference siamese "result/submission_siamese.json"
    ;;
  *)
    echo "Unsupported MATCHER='$MATCHER'. Use mobileclip2, siamese, or all." >&2
    exit 2
    ;;
esac
