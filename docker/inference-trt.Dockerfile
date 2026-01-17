ARG GPU_BASE_IMAGE=detr-mmap-gpu-base:latest
FROM ${GPU_BASE_IMAGE} AS builder

WORKDIR /build

# Install TensorRT development libraries
# We use the system paths which are /usr/include and /usr/lib/x86_64-linux-gnu
RUN apt-get update && apt-get install -y \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build with TensorRT backend
# We set TENSORRT_ROOT to /usr so build.rs finds headers in /usr/include and libs in /usr/lib/x86_64-linux-gnu
ENV TENSORRT_ROOT=/usr
RUN cargo build --release --bin inference --features trt-backend --no-default-features

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libnvinfer10 \
    libnvinfer-plugin10 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/inference /usr/local/bin/app

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENTRYPOINT ["/usr/local/bin/app"]
