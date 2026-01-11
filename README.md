# RT-DETR Object Detection on edge devices with zero-copy memory-mapped IPC
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Build](https://github.com/jordandelbar/bridge-rt/actions/workflows/ci.yaml/badge.svg)](https://github.com/jordandelbar/bridge-rt/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/jordandelbar/bridge-rt/branch/main/graph/badge.svg?token=GFI0VJOZ9G)](https://codecov.io/gh/jordandelbar/bridge-rt)

A RT-DERT implementation with Rust ORT, FlatBuffers and k3d

## 📝 Overview

This project implements a complete edge AI pipeline with RT-DETR v2 object detection, designed for minimal latency and maximum throughput through zero-copy shared memory communication.

## Tech Stack

  - capture: Camera frame acquisition using [nokhwa] and V4L2
  - inference: RT-DETR model inference via [ORT] with CUDA support
  - controller: State machine managing sentry mode (Standby/Alarmed) based on human detection, publishes events to MQTT
  - gateway: WebSocket server streaming frames + detections to connected clients
  - mosquitto: MQTT broker for centralized event collection (deployed on central node)

Use of mmap with FlatBuffer for zero serialization + mqueue semaphore

## Architecture

## Technical decisions

Edge devices have limited CPU and memory. Network protocol overhead (TCP, HTTP/2, serialization) adds latency and CPU usage.

Memory-mapped files (`mmap`) provide true zero-copy IPC. There is no serialization: reader accesses data directly in writer's memory
The trade-off is that it only works for local IPC, this is not secured for cloud deployment with shared machine but clearly fitting for edge deployments.

## Installation
**Rust** 1.92+ (`rustup default 1.92.0`)
**Docker** with buildx
**k3d** (lightweight Kubernetes)
**ONNX Runtime** (installed automatically in containers)
**V4L2-compatible USB camera** (for capture service)

### Quick Start

```bash
# Clone repository
git clone https://github.com/jordandelbar/detr-mmap.git
cd detr-mmap

# Create k3d cluster + deploy services
make up

# Check deployment
kubectl get pods -n bridge-rt

# View logs
kubectl logs -n bridge-rt -l component=inference --follow
```

## ⚡ Running with CUDA

## Performance & Benchmarks

<!--references-->
[nokhwa]: (https://github.com/l1npengtul/nokhwa)
[ONNX]: https://onnx.ai/
[Axum]: https://docs.rs/axum/latest/axum/
[ORT]: https://ort.pyke.io/

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Note: This project is intended for educational and research purposes.
