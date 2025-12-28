FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
ENV PATH="/root/.cargo/bin:${PATH}"

RUN wget -q https://github.com/google/flatbuffers/releases/download/v24.3.25/Linux.flatc.binary.clang++-15.zip && \
    unzip -q Linux.flatc.binary.clang++-15.zip && \
    chmod +x flatc && \
    mv flatc /usr/local/bin/ && \
    rm Linux.flatc.binary.clang++-15.zip

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
