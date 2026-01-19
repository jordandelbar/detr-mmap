ARG GPU_BASE_IMAGE=detr-mmap-gpu-base:latest
FROM ${GPU_BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    cmake \
    nasm \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY models/ models/

ENV TENSORRT_ROOT=/usr

ENV LD_LIBRARY_PATH=/workspace/target/release:/workspace/target/release/deps:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN cargo build --release --benches --workspace --features "ort-backend trt-backend"

CMD ["cargo", "bench", "--workspace", "--features", "ort-backend trt-backend"]
