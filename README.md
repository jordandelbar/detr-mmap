# RF-DETR on Edge Devices with Zero-Serialization Memory-Mapped IPC
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Build](https://github.com/jordandelbar/detr-mmap/actions/workflows/ci.yaml/badge.svg)](https://github.com/jordandelbar/detr-mmap/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/jordandelbar/detr-mmap/branch/main/graph/badge.svg?token=GFI0VJOZ9G)](https://codecov.io/gh/jordandelbar/detr-mmap)

An RF-DETR implementation with Rust, CXX | ORT, FlatBuffers and k3d

## Overview

This project implements a complete edge AI pipeline with RF-DETR object detection, designed for minimal latency and maximum throughput through zero-serialization shared memory communication.
A sentry mode state machine with hysteresis reduces computation by switching to standby when no humans are detected.
The state machine requires multiple consecutive detections before transitioning to alarmed mode (30 FPS), and multiple consecutive non-detections before returning to standby (3 FPS), preventing rapid flickering between modes.

![screencast](https://github.com/user-attachments/assets/814a93cf-61cf-4937-8a9b-a6c5bb66af69)

<sub>Video by [Martina Tomšič](https://www.pexels.com/video/dog-waiting-in-front-of-a-door-6477490/) on Pexels</sub>

## Tech Stack

  - Capture: Camera frame acquisition using [v4l]
  - Preprocessing: CPU (default) or CUDA (automatic with TensorRT backend)
  - Inference: [RF-DETR] model inference:
    - Rust + [ONNX] [ORT] Runtime (CPU / CUDA)
    - TensorRT via C++ (integrated into Rust using [CXX])
  - Controller: State machine managing sentry mode (Standby/Alarmed) based on human detection, publishes events to [MQTT]
  - Gateway: WebSocket [Axum] server streaming frames + detections to clients
  - IPC: Memory-mapped files with [FlatBuffers] for zero-serialization, synchronized with POSIX mqueue semaphores.
  - Broker: [mosquitto] MQTT broker for centralized event collection
  - Observability: [OpenTelemetry] + [Jaeger] for tracing and metrics

## Architecture

![Alt text](./docs/schemas/detr-mmap-context.svg)

![Alt text](./docs/schemas/detr-mmap-app.svg)

## Technical decisions

Edge devices have limited CPU and memory. Network protocol overhead (TCP, HTTP/2, serialization) adds latency and CPU usage.

Memory-mapped files (`mmap`) provide true zero-serialization IPC. There is no runtime serialization or copying; data is written once and read zero-copy via memory mapping.
The trade-off is that it only works for local IPC, this is not secured for cloud deployment with shared machine but clearly fitting for edge deployments.

I used k3s even though it adds some memory footprint for the ease of use when it comes to edge deployment.

The pipeline is intentionally asynchronous. Frames are displayed immediately while detections correspond to the previous frame,
introducing <1-frame temporal skew (multiple frame skew if using the CPU). This design maximizes throughput and minimizes perceived latency.

GPU preprocessing is enabled automatically with the TensorRT backend. On edge devices where the CPU is typically the bottleneck, offloading preprocessing to the GPU frees CPU resources for other tasks.

## Installation

For quick demo:
 - **Docker** with buildx

For full setup:
 - **Docker** with buildx
 - **Rust** 1.92+ (`rustup default 1.92.0`)
 - **k3d** (lightweight k3s in docker) - [install guide](https://k3d.io/stable/#installation)
 - **uv** (Python package manager) - [install guide](https://docs.astral.sh/uv/getting-started/installation/)
 - **ONNX Runtime** (installed automatically in containers)
 - **CUDA** (to run GPU inference)

### Quick Start

```bash
# Clone repository
git clone https://github.com/jordandelbar/detr-mmap.git
cd detr-mmap
```

#### Quick Demo (pre-built images)

Run with pre-built images from GHCR (model baked in, no build required):

```bash
# Start all services
just demo-up

# Open webpage
just open-webpage

# Stop services
just demo-down
```

#### Local Development

Build and run locally from source with CPU inference:

```bash
# Download ONNX model from HuggingFace
just download-model

# Build and start all services
just dev-up

# Open webpage
just open-webpage

# Stop services
just dev-down
```

#### k3d Deployment (GPU)

For GPU inference with TensorRT, you need to build the INT8 engine for your specific GPU:

```bash
# Build TensorRT INT8 engine (requires NVIDIA GPU, CUDA, TensorRT)
just build-engine

# Create k3d cluster + deploy services
just up

# Check deployment
kubectl get pods -n detr-mmap

# View logs
kubectl logs -n detr-mmap -l component=inference --follow
```
By default, this runs the TensorRT version of inference, so ensure you have configured your Docker
daemon to run with CUDA. See the next section for setup instructions.

## Running with CUDA

Follow [this guide](https://github.com/jordandelbar/yolo-tonic/blob/a146a7820c173545c47c5c1bac7cdf0417773150/docs/setup/nvidia_docker.md) to set up CUDA correctly.

## Benchmarks & Performance

Benchmarks run on NVIDIA RTX 2060 Super and AMD Ryzen 7 9800x3D with 1920x1080 RGB input frames.

> [!NOTE]
> These benchmarks were run on high-end desktop hardware. Edge device performance will vary.
> Benchmarks on edge device (Raspberry Pi, Jetson) are welcome.

### Benchmarks

All benchmarks measured with 1920x1080 RGB input frames.

#### Inference

| Backend    | Latency   | Throughput |
|------------|-----------|------------|
| ORT (CPU)  | 66.8 ms   | ~15 FPS    |
| ORT (CUDA) | 15.1 ms   | ~66 FPS    |
| TensorRT   |  3.7 ms   | ~270 FPS   |

#### Preprocessing

| Backend                   | Latency  | Speedup |
|---------------------------|----------|---------|
| CPU (AMD Ryzen 9800x3D)   | 444 µs   | 1x      |
| GPU (NVIDIA RTX 2060S)    | 21 µs    | 21x     |

> [!NOTE]
> GPU preprocessing benchmarks measure kernel execution only, excluding host-to-device transfer.
> In production with TensorRT, frames stay on GPU (decoder → preprocess → inference), achieving these speeds.

#### Other Components

| Component      | Latency  |
|----------------|----------|
| Postprocessing | 23 µs    |
| Frame write    | 309 µs   |
| Frame read     | 46 ns    |

#### IPC Performance (FlatBuffers + mmap)

| Scenario           | Write    | Read     | Roundtrip |
|--------------------|----------|----------|-----------|
| No detections      | 22 ns    | 29 ns    | 53 ns     |
| Single detection   | 51 ns    | 51 ns    | 101 ns    |
| Few detections (5) | 113 ns   | 120 ns   | 228 ns    |
| Many detections    | 352 ns   | 350 ns   | 703 ns    |
| Crowded scene      | 1.6 µs   | 1.6 µs   | 3.2 µs    |

Run benchmarks yourself:
```bash
just bench
# HTML reports output to benchmark-reports/
```

### Performance

Here is an example of a typical trace span:

![Alt text](./docs/images/traces.png)

| Stage                    | Component | Latency     |
|--------------------------|-----------|-------------|
| MJPEG decoding           | Capture   | ~2.5 ms     |
| Frame write (mmap)       | Capture   | ~300 µs     |
| Host-to-device transfer  | Inference | ~600 µs     |
| Preprocessing (kernel)   | Inference | ~150 µs     |
| Model inference          | Inference | ~4 ms       |
| Postprocessing           | Inference | ~30 µs      |
| Detection write (mmap)   | Inference | ~50 µs      |
| **Total**                |           | **~7.5 ms** |
| JPEG encoding (async)    | Gateway   | ~2.5 ms     |

> [!NOTE]
> Gateway runs asynchronously, encoding the previous frame while inference processes the current frame.

## Testing without a camera

```bash
sudo modprobe v4l2loopback video_nr=0
ffmpeg -re -stream_loop -1 -i video.mp4 -vf "scale=1920:1080" -c:v mjpeg -f v4l2 /dev/video0
```

## Ideas about what to do with this repo

 - DevOps: Deploy with KubeEdge instead of K3s (KinD + KubeEdge)
 - SWE: Replace WebSocket streaming with proper H.264 setup

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Citation

This project uses RF-DETR by Roboflow. If you use this software, please cite:

```bibtex
@software{robinson2025rf-detr,
  author = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei},
  title = {RF-DETR},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/roboflow/rf-detr}
}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

> [!NOTE]
> This project is intended for educational and research purposes.

<!--references-->
[RF-DETR]: https://github.com/roboflow/rf-detr
[v4l]: https://crates.io/crates/v4l
[ONNX]: https://onnx.ai/
[CXX]: https://cxx.rs/
[MQTT]: https://mqtt.org/
[Axum]: https://docs.rs/axum/latest/axum/
[mosquitto]: https://mosquitto.org/
[FlatBuffers]: https://flatbuffers.dev/
[ORT]: https://ort.pyke.io/
[OpenTelemetry]: https://opentelemetry.io/
[Jaeger]: https://www.jaegertracing.io/
