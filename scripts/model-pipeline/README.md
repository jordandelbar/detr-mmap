# INT8 Model Preparation Pipeline

This folder contains scripts for preparing and distributing RF-DETR INT8 models.

## Overview

The pipeline is split into two parts:

1. **Maintainer Pipeline**: Export ONNX, calibrate, and push to HuggingFace
2. **User Pipeline**: Download from HuggingFace and build TensorRT engine for local GPU

## Maintainer Pipeline

### Prerequisites

```bash
cd scripts/model-pipeline

# Install dependencies with uv
uv sync --extra tensorrt

# Or create a virtual environment first
uv venv
source .venv/bin/activate
uv sync --extra tensorrt
```

### Full Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
uv run prepare_and_publish.py --hf-repo your-org/rfdetr-small-int8
```

Options:
- `--model-variant`: `small` or `base` (default: `small`)
- `--calibration-images`: Number of images for calibration (default: 100)
- `--dry-run`: Print commands without executing
- `--skip-export`: Use existing ONNX model
- `--skip-calibration-download`: Use existing calibration images
- `--skip-upload`: Skip HuggingFace upload

### Individual Steps

You can also run each step manually:

```bash
# 1. Export ONNX model (512x512)
uv run export_onnx.py --output-dir ../../models/rfdetr_small/ --model-variant small

# 2. Download calibration images
uv run download_calibration_data.py --count 100

# 3. Generate calibration tensors (Rust crate)
cargo run -p calibration --release -- --count 100

# 4. Build calibration cache
uv run --extra tensorrt build_calibration_cache.py \
    --onnx-path ../../models/rfdetr_small/inference_model.onnx \
    --cache-path ../../models/rfdetr_small/calibration.cache
```

## User Pipeline

### Prerequisites

```bash
# With uv (recommended)
uv pip install huggingface_hub tensorrt pycuda numpy

# Or with pip
pip install huggingface_hub tensorrt pycuda numpy
```

### Build Engine

Download the model from HuggingFace and build a TensorRT engine:

```bash
uv run user_build_engine.py \
    --hf-repo your-org/rfdetr-small-int8 \
    --output ./rfdetr_int8.engine
```

Or use local files:

```bash
uv run user_build_engine.py \
    --onnx ./inference_model.onnx \
    --cache ./calibration.cache \
    --output ./rfdetr_int8.engine
```

### Standalone Usage

The `user_build_engine.py` script is designed to be standalone. Users can copy just this file and run it independently:

```bash
# Download the script
wget https://raw.githubusercontent.com/your-org/detr-mmap/main/scripts/model-pipeline/user_build_engine.py

# Install dependencies
uv pip install huggingface_hub tensorrt pycuda numpy

# Build engine
python user_build_engine.py --hf-repo your-org/rfdetr-small-int8 --output ./model.engine
```

## File Structure

```
scripts/
├── model-pipeline/                    # This folder
│   ├── README.md                      # This file
│   ├── pyproject.toml                 # Python dependencies
│   ├── export_onnx.py                 # Export RF-DETR to ONNX (512x512)
│   ├── download_calibration_data.py   # Download COCO images for calibration
│   ├── build_calibration_cache.py     # Build INT8 calibration cache
│   ├── prepare_and_publish.py         # Orchestrates full pipeline + HF push
│   └── user_build_engine.py           # User-facing: download from HF + build engine
├── quantization/                      # Calibration data storage
│   ├── calibration_data/              # Downloaded COCO images
│   └── calibration_tensors/           # Preprocessed tensors from Rust crate
```

## HuggingFace Repository Structure

After running the maintainer pipeline, the HuggingFace repo will contain:

```
your-org/rfdetr-small-int8/
├── inference_model.onnx    # F32 ONNX model (~110MB)
├── calibration.cache       # INT8 calibration cache (~120KB)
└── README.md               # Usage instructions
```

## Verification

### Test Maintainer Pipeline (Dry Run)

```bash
uv run prepare_and_publish.py --hf-repo test-org/test-model --dry-run
```

### Test User Pipeline

```bash
uv run user_build_engine.py \
    --hf-repo your-org/rfdetr-small-int8 \
    --output ./rfdetr_int8.engine
```

### Verify Engine Works

```bash
# Set MODEL_PATH to the new engine and run inference tests
MODEL_PATH=./rfdetr_int8.engine cargo test -p inference
```

## Technical Details

### Input Resolution

All models use fixed 512x512 input resolution for INT8 quantization. This matches the default preprocessing in `crates/preprocess`.

### Calibration

INT8 calibration uses 100 random images from COCO val2017. The calibration cache captures activation ranges for each layer, enabling accurate INT8 inference.

### GPU Compatibility

The calibration cache is GPU-agnostic, but the TensorRT engine is GPU-specific. Users must build their own engine for their target GPU using `user_build_engine.py`.
