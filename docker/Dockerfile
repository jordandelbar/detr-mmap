FROM rust:1.92-trixie AS builder

ARG BINARY_NAME
ARG FLATC_VERSION=24.3.25

RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libclang-dev \
    pkg-config \
    libssl-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/google/flatbuffers/releases/download/v${FLATC_VERSION}/Linux.flatc.binary.clang++-15.zip \
 && unzip -q Linux.flatc.binary.clang++-15.zip \
 && install -m 0755 flatc /usr/local/bin/flatc \
 && rm -f flatc Linux.flatc.binary.clang++-15.zip

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY crates/bridge/Cargo.toml crates/bridge/
COPY crates/common/Cargo.toml crates/common/
COPY crates/capture/Cargo.toml crates/capture/
COPY crates/inference/Cargo.toml crates/inference/
COPY crates/gateway/Cargo.toml crates/gateway/
COPY crates/schema/Cargo.toml crates/schema/
COPY crates/controller/Cargo.toml crates/controller/

RUN mkdir -p crates/bridge/src crates/common/src crates/capture/src crates/inference/src crates/gateway/src crates/schema/src crates/controller/src \
             crates/bridge/benches crates/inference/benches \
    && touch crates/bridge/src/lib.rs \
    && touch crates/common/src/lib.rs \
    && touch crates/schema/src/lib.rs \
    && touch crates/capture/src/main.rs \
    && touch crates/inference/src/main.rs \
    && touch crates/gateway/src/main.rs \
    && touch crates/controller/src/main.rs \
    && touch crates/bridge/benches/frame_throughput.rs \
    && touch crates/bridge/benches/detection_throughput.rs \
    && touch crates/inference/benches/inference_pipeline.rs

RUN cargo fetch

COPY crates/ crates/

RUN cargo build --release --bin ${BINARY_NAME} \
 && strip target/release/${BINARY_NAME}

FROM gcr.io/distroless/cc-debian13:latest
ARG BINARY_NAME

COPY --from=builder /build/target/release/${BINARY_NAME} /usr/local/bin/app
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/app"]
