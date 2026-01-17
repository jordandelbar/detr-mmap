ARG GPU_BASE_IMAGE=detr-mmap-gpu-base:latest
FROM ${GPU_BASE_IMAGE}

# Install TensorRT development libraries and libclang for benchmarking
RUN apt-get update && apt-get install -y \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY models/ models/

# Set TENSORRT_ROOT for build.rs to find TensorRT
ENV TENSORRT_ROOT=/usr

# Ensure ONNX Runtime libraries can be found at runtime
ENV LD_LIBRARY_PATH=/workspace/target/release:/workspace/target/release/deps:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Pre-build all benchmarks to cache dependencies and compilation artifacts
RUN cargo build --release --benches --workspace --features "ort-backend trt-backend"

CMD ["cargo", "bench", "--workspace", "--features", "ort-backend trt-backend"]
