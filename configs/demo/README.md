# Demo Configs (Public Data)

Configs:

- `mobileclip2_demo_blackbox0.json`
- `siamese_demo_blackbox0.json`

Both configs are tuned for:

- public sample `BlackBox_0`
- frame window `293..542` (250 frames, ~10s at 25 FPS)
- lower pipeline load than slide defaults (`top_k=20`, `yoloe_conf=0.0015`)

Run MobileCLIP2 demo:

```bash
python inference.py \
  --preprocessed-dir preprocessed_data/public_test \
  --output-json result/demo/submission_mobileclip2_demo.json \
  --limit-videos 1 \
  --max-frames 600 \
  --matcher mobileclip2 \
  --yoloe-conf 0.0015 \
  --top-k 20 \
  --fused-threshold 0.52 \
  --w-det 0.30 \
  --w-match 0.35 \
  --match-threshold 0.10 \
  --similarity-add-threshold 0.80

python scripts/render_bboxes_video.py \
  --json result/demo/submission_mobileclip2_demo.json \
  --video data/public_test/samples/BlackBox_0/drone_video.mp4 \
  --output result/demo/mobileclip2_demo_10s.mp4 \
  --video-id BlackBox_0 \
  --start-frame 293 \
  --num-frames 250 \
  --label MobileCLIP2 \
  --zoom-inset --inset-size 240
```

Run Siamese demo:

```bash
python inference.py \
  --preprocessed-dir preprocessed_data/public_test \
  --output-json result/demo/submission_siamese_demo.json \
  --limit-videos 1 \
  --max-frames 600 \
  --matcher siamese \
  --yoloe-conf 0.0015 \
  --top-k 20 \
  --fused-threshold 0.52 \
  --w-det 0.30 \
  --w-match 0.35 \
  --match-threshold 0.10 \
  --similarity-add-threshold 0.80

python scripts/render_bboxes_video.py \
  --json result/demo/submission_siamese_demo.json \
  --video data/public_test/samples/BlackBox_0/drone_video.mp4 \
  --output result/demo/siamese_demo_10s.mp4 \
  --video-id BlackBox_0 \
  --start-frame 293 \
  --num-frames 250 \
  --label Siamese \
  --zoom-inset --inset-size 240
```

## BlackBox_0 Scene-Split Demos (v2)

- `mobileclip2_blackbox0_sceneA_12s.json`
- `siamese_blackbox0_sceneB_12s.json`

These two outputs use different scenes in the same public video:

- Scene A: frames `295..430` for MobileCLIP2
- Scene B: frames `4851..4983` for Siamese

Rendered at `11 FPS` to keep runtime short segments while producing ~12s demo videos.

## Recommended For Slides (v2)

- `mobileclip2_blackbox0_sceneA_v2.json` -> `result/demo_v2/blackbox_0_mobileclip2_sceneA_v2_12s.mp4`
- `siamese_blackbox0_sceneB_v2.json` -> `result/demo_v2/blackbox_0_siamese_sceneB_v2_14s.mp4`

These v2 clips use tighter frame ranges where the box is more consistently on the black case.

## Intro-Then-Detect (v4)

- `mobileclip2_blackbox0_intro_v4.json`
- `siamese_blackbox0_intro_v4.json`

Both clips start with no bbox for a few seconds, then begin detecting and tracking.

Siamese improved variant:

- `siamese_blackbox0_intro_v5.json` -> `result/demo_v2/blackbox_0_siamese_intro_v5_15s.mp4`
- `siamese_blackbox0_intro_v6.json` -> `result/demo_v2/blackbox_0_siamese_intro_v6_15s.mp4` (recommended)
