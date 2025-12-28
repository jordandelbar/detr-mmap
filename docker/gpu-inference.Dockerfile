ARG GPU_BASE_IMAGE=bridge-rt-gpu-base:latest
FROM ${GPU_BASE_IMAGE} AS builder

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --bin inference

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/inference /usr/local/bin/app

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENTRYPOINT ["/usr/local/bin/app"]
