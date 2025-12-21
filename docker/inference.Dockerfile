FROM rust:1.92-bookworm AS builder

RUN apt-get update && apt-get install -y \
    clang \
    libclang-dev \
    pkg-config \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/google/flatbuffers/releases/download/v24.3.25/Linux.flatc.binary.clang++-15.zip && \
    unzip -q Linux.flatc.binary.clang++-15.zip && \
    chmod +x flatc && \
    mv flatc /usr/local/bin/ && \
    rm Linux.flatc.binary.clang++-15.zip

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --bin inference

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/inference /usr/local/bin/inference

RUN mkdir -p /dev/shm

ENTRYPOINT ["/usr/local/bin/inference"]
