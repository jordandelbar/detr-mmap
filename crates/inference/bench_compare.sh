#!/bin/bash
# Benchmark comparison script for ORT vs TensorRT backends

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Inference Backend Benchmark Comparison"
echo "======================================"
echo ""

# Check if models exist
ONNX_MODEL="../../models/model.onnx"
TRT_MODEL="../../models/model_fp16.engine"

if [ ! -f "$ONNX_MODEL" ]; then
    echo "⚠️  Warning: ONNX model not found at $ONNX_MODEL"
    echo "   ONNX benchmarks will be skipped"
fi

if [ ! -f "$TRT_MODEL" ]; then
    echo "⚠️  Warning: TensorRT model not found at $TRT_MODEL"
    echo "   TensorRT benchmarks will be skipped"
fi

echo ""

# Set LD_LIBRARY_PATH for TensorRT
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}

# Run ORT backend benchmarks (inference only)
echo "Running ONNX Runtime (ORT) Backend Benchmarks..."
echo "------------------------------------------------"
cargo bench --features ort-backend --bench inference_pipeline -- inference 2>&1 | tee /tmp/ort_bench.txt
echo ""

# Run TensorRT backend benchmarks (inference only)
echo "Running TensorRT (TRT) Backend Benchmarks..."
echo "--------------------------------------------"
cargo bench --features trt-backend --bench inference_pipeline -- inference 2>&1 | tee /tmp/trt_bench.txt
echo ""

# Extract and compare inference times
echo "======================================"
echo "Inference Performance Comparison"
echo "======================================"
echo ""

ORT_TIME=$(grep "ort_rtdetr_640x640" /tmp/ort_bench.txt | grep -oP "time:\s+\[\K[^\]]+(?=\])" | head -1 || echo "N/A")
TRT_TIME=$(grep "trt_rtdetr_640x640" /tmp/trt_bench.txt | grep -oP "time:\s+\[\K[^\]]+(?=\])" | head -1 || echo "N/A")

echo "┌─────────────────────┬────────────────────────┐"
echo "│ Backend             │ Inference Time         │"
echo "├─────────────────────┼────────────────────────┤"
printf "│ %-19s │ %-22s │\n" "ONNX Runtime" "$ORT_TIME"
printf "│ %-19s │ %-22s │\n" "TensorRT" "$TRT_TIME"
echo "└─────────────────────┴────────────────────────┘"
echo ""

echo "📊 Detailed reports available in target/criterion/"
echo ""
echo "To view HTML reports:"
echo "  - ORT:  target/criterion/inference/ort_rtdetr_640x640/report/index.html"
echo "  - TRT:  target/criterion/inference/trt_rtdetr_640x640/report/index.html"
echo ""
