# EyeTimeline Benchmark Pipeline

This directory contains the first-pass downstream benchmark pipeline for the
current EyeTimeline stage: CFP/Fundus and OCT MAE encoders compared against
RETFound, EyeCLIP, ImageNet-MAE, and random-init ViT-Large under one protocol.

The repository stores only code and manifests. Datasets, pretrained weights,
external repositories, checkpoints, and benchmark outputs are ignored by Git.

## Directory Layout

```text
benchmark_data/       # downloaded or manually placed public datasets
baseline_weights/     # RETFound / EyeCLIP / ImageNet MAE weights
external_repos/       # optional official baseline repositories
benchmark_outputs/    # fine-tuned checkpoints, predictions, metrics
benchmarks/           # this code
```

## Standard Dataset Format

All downstream datasets are normalized to a single CSV format:

```csv
path,label,split
train/class_a/img001.png,class_a,train
val/class_a/img010.png,class_a,val
test/class_b/img100.png,class_b,test
```

`path` may be absolute or relative to `--data-root`. Labels may be integers or
strings. The training script writes `class_to_idx.json` into the output folder.

For RETFound-style image folders, build the CSV with:

```bash
python benchmarks/prepare_imagefolder_csv.py \
  --dataset-root benchmark_data/OCTID \
  --output benchmark_data/OCTID/labels.csv
```

## Environment

```bash
bash scripts/setup_benchmark_env.sh
```

The script creates/updates a conda environment and installs the project plus
benchmark-only dependencies.

## Assets

Download baseline code and weights:

```bash
bash scripts/download_baseline_assets.sh
```

RETFound HuggingFace models may require prior access approval and `HF_TOKEN`.
EyeCLIP weights are hosted on Google Drive and require `gdown`.

Download RETFound public benchmark split packages when the server has enough
disk space:

```bash
python benchmarks/download_benchmark_data.py \
  --datasets APTOS2019 IDRID PAPILA Glaucoma_fundus OCTID \
  --output-root benchmark_data \
  --continue-on-error \
  --extract
```

Some official datasets require Kaggle, Dataverse, Mendeley, or manual approval.
The downloader records those URLs but does not bypass access controls.

## Training

Linear probe:

```bash
python benchmarks/train_classifier.py \
  --labels-csv benchmark_data/OCTID/labels.csv \
  --data-root benchmark_data/OCTID \
  --backbone eyetimeline_mae \
  --checkpoint baseline_weights/ours/oct/checkpoint_best.pth \
  --freeze-encoder \
  --epochs 50 \
  --batch-size 64 \
  --output-dir benchmark_outputs/OCTID/ours_lp
```

Full fine-tuning:

```bash
python benchmarks/train_classifier.py \
  --labels-csv benchmark_data/OCTID/labels.csv \
  --data-root benchmark_data/OCTID \
  --backbone retfound_mae \
  --checkpoint baseline_weights/retfound/RETFound_mae_natureOCT \
  --epochs 50 \
  --batch-size 24 \
  --output-dir benchmark_outputs/OCTID/retfound_ft
```

Evaluate a trained checkpoint:

```bash
python benchmarks/evaluate_classifier.py \
  --labels-csv benchmark_data/OCTID/labels.csv \
  --data-root benchmark_data/OCTID \
  --split test \
  --checkpoint benchmark_outputs/OCTID/ours_lp/checkpoint_best.pth \
  --output-dir benchmark_outputs/OCTID/ours_lp/test_eval
```

## First-Pass Matrix

Recommended initial matrix:

- Fundus/CFP: `APTOS2019`, `IDRID`, `PAPILA`, `Glaucoma_fundus`
- OCT: `OCTID`, `OCTDL`
- Models: `ours`, `RETFound`, `EyeCLIP`, `ImageNet-MAE`, `random`
- Modes: `linear probe`, `full fine-tuning`, then `few-shot`

EyeCLIP is included as an image-encoder baseline for linear probing and
fine-tuning. Zero-shot, retrieval, and VQA should wait until EyeTimeline has
image-text alignment and a text encoder.

## Production Matrix Runner

After placing datasets and weights under `/data1/kechuang/EyeTimelineAssets`,
run the full supervised linear-probe benchmark matrix in the background:

```bash
cd /data1/kechuang/EyeTimeline
ASSET_ROOT=/data1/kechuang/EyeTimelineAssets \
BENCHMARK_GPUS=0,1,2,3 \
DETACH=1 \
bash scripts/run_benchmark_matrix.sh
```

Default matrix:

- datasets: `aptos2019 idrid papila glaucoma_fundus octid`
- models: `ours retfound eyeclip`
- mode: `linear`

Useful overrides:

```bash
MODELS="ours retfound eyeclip imagenet_mae random_mae"
MODES="linear full"
EPOCHS_LINEAR=50
EPOCHS_FULL=50
BATCH_SIZE_LINEAR=64
BATCH_SIZE_FULL=16
FORCE=1
```

Monitor a background run:

```bash
bash scripts/benchmark_status.sh
tail -f /data1/kechuang/EyeTimelineAssets/benchmark_outputs/run_logs/latest.log
```

Each job writes its own log under
`benchmark_outputs/job_logs/<mode>/<modality>/<dataset>/<model>.log`.
The runner writes progress to the latest run directory:

```text
benchmark_outputs/latest_run.txt
benchmark_outputs/runs/<timestamp>/status.json
benchmark_outputs/runs/<timestamp>/progress.jsonl
benchmark_outputs/summary.csv
benchmark_outputs/summary.md
```
