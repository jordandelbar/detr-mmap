ARG GPU_BASE_IMAGE=detr-mmap-gpu-base:latest
FROM ${GPU_BASE_IMAGE} AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

ENV TENSORRT_ROOT=/usr
RUN nvcc --ptx -arch=sm_75 \
    crates/preprocess/cuda/preprocess.cu \
    -o crates/preprocess/cuda/preprocess.ptx
RUN cargo build --release --bin inference --features gpu-preprocess,trt-backend,cudarc/cuda-12060 --no-default-features

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libnvinfer10 \
    libnvinfer-plugin10 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/inference /usr/local/bin/app

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV GPU_PREPROCESS=true

ENTRYPOINT ["/usr/local/bin/app"]
