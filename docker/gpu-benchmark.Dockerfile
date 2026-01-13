ARG GPU_BASE_IMAGE=bridge-rt-gpu-base:latest
FROM ${GPU_BASE_IMAGE}

# Install TensorRT development libraries for benchmarking
RUN apt-get update && apt-get install -y \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY models/ models/

# Set TENSORRT_ROOT for build.rs to find TensorRT
ENV TENSORRT_ROOT=/usr

CMD ["cargo", "bench", "-p", "inference", "--bench", "inference_pipeline"]
