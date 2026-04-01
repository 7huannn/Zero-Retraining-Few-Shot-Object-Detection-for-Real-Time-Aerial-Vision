# Drone Few-Shot Pipeline

This repository contains a minimal end-to-end pipeline for:

- installing the official YOLOE stack
- preprocessing reference samples for YOLOE visual prompting
- training a Siamese embedding model from `data/train/annotations/annotations.json`
- evaluating the trained Siamese checkpoint

## Quick Start

Install the local dependencies and the official YOLOE repository:

```bash
./setup_env.sh
```

Check the environment:

```bash
python predict.py check-env
```

## YOLOE Preprocessing

Prepare YOLOE references for the training split:

```bash
python predict.py preprocess-yoloe \
  --dataset data/train \
  --output-dir preprocessed_data/train
```

Prepare YOLOE references for the public test split:

```bash
python predict.py preprocess-yoloe \
  --dataset data/public_test \
  --output-dir preprocessed_data/public_test
```

If you already trained the Siamese model, add `--siamese-checkpoint result/siamese/best.pt` to save `reference_embeddings.pt` next to each sample.

## Siamese Training

Train the Siamese model:

```bash
python predict.py train-siamese --config siamese/train_config.yaml
```

Evaluate the best checkpoint:

```bash
python predict.py eval-siamese --checkpoint result/siamese/best.pt
```

Run a simple pairwise comparison demo:

```bash
python predict.py compare-siamese --config siamese/siamese_config.yaml
```

To compare with the trained checkpoint instead of a plain pretrained backbone, set `checkpoint_path: result/siamese/best.pt` in `siamese/siamese_config.yaml`.

## Smoke Tests

Run YOLOv8 and YOLOE demo predictions:

```bash
python predict.py run-yolo --config yoloe/yolo_config.yaml
python predict.py run-yoloe --config yoloe/yoloe_config.yaml
```
