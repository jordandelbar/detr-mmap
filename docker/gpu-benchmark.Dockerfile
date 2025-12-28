ARG GPU_BASE_IMAGE=bridge-rt-gpu-base:latest
FROM ${GPU_BASE_IMAGE}

WORKDIR /workspace

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY models/ models/

CMD ["cargo", "bench", "-p", "inference", "--bench", "inference_pipeline"]
