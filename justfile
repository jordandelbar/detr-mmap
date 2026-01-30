#########
# Model #
#########

HF_REPO := "jordandelbar/rf-detr-s-int8"

# [Admin] Run the full model pipeline: export ONNX, calibrate, upload to HF
# Usage: just model-pipeline --hf-repo org/repo-name
# Options:
#   --model-variant small|base   Model variant (default: small)
#   --calibration-images N       Number of calibration images (default: 100)
#   --dry-run                    Print commands without executing
#   --skip-export                Skip ONNX export
#   --skip-calibration-download  Skip downloading calibration images
#   --skip-upload                Skip uploading to HuggingFace
model-pipeline *ARGS:
    cd scripts/model-pipeline && uv run main.py {{ARGS}}

# [User] Download ONNX model from HuggingFace (for CPU/docker-compose usage)
# Usage: just download-model
# Downloads ONNX to models/ directory
download-model:
    cd scripts/model-setup && uv run download.py --hf-repo {{HF_REPO}} --output ../../models

# [User] Build TensorRT INT8 engine from HuggingFace model (requires NVIDIA GPU)
# Usage: just build-engine [--output path/to/engine.engine]
# Requires: NVIDIA drivers, CUDA, TensorRT
# By default, downloads from HF_REPO and outputs to models/rfdetr_int8.engine
build-engine *ARGS:
    cd scripts/model-setup && uv run --extra tensorrt build.py --hf-repo {{HF_REPO}} --output ../../models/rfdetr_int8.engine {{ARGS}}

#######
# Dev #
#######

fmt:
    @cargo fmt --all
    @clang-format -i ./crates/inference-cpp/**/*.{cpp,hpp}

#######
# Run #
#######

build:
    @docker buildx bake

# k3d deployement
up:
    @./scripts/setup-k3d.sh
    @./scripts/deploy-k3d.sh

down:
    @k3d cluster delete detr-mmap

# Local development (builds from source, no k8s, CPU-only)
dev-up:
    @docker compose -f docker/compose.local.yml up --build

dev-down:
    @docker compose -f docker/compose.local.yml down

# Demo (uses pre-built images from GHCR)
demo-up:
    @docker compose up

demo-down:
    @docker compose down

open-webpage:
    @if command -v xdg-open > /dev/null; then xdg-open index.html; \
    elif command -v open > /dev/null; then open index.html; \
    elif command -v start > /dev/null; then start index.html; \
    else echo "No suitable command found to open the file."; fi

########
# Test #
########

test:
    cargo test --workspace

coverage:
    cargo llvm-cov --workspace --lib --tests

bench FILTER="":
    @BENCH_FILTER="{{FILTER}}" docker compose -f docker/compose.benchmark.yml up --build
