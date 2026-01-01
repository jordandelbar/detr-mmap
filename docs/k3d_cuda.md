# K3D with CUDA

This guide explains how to run GPU-accelerated workloads in k3d for the Bridge-RT project.

## Overview

The default k3s image uses Alpine Linux, which doesn't support the NVIDIA Container Runtime. To enable GPU support, you need to build a custom k3s image based on a CUDA-compatible base image (Ubuntu).

## Prerequisites

### 1. NVIDIA Drivers

Ensure you have NVIDIA drivers installed on your host:

```bash
nvidia-smi
```

You should see your GPU(s) listed with driver version information.

### 2. NVIDIA Container Toolkit

Install the NVIDIA Container Toolkit to allow Docker to use GPUs by following the instruction [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


### 3. Verify Docker GPU Access

Test that Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

You should see the same GPU information as the host `nvidia-smi` command.

## Building the Custom k3s Image

### 1. Image Architecture

The custom k3s image (`docker/k3s-gpu.Dockerfile`) is built in two stages:

1. Copy k3s binaries from the official k3s image
2. Build on top of NVIDIA CUDA base image with:
- NVIDIA Container Toolkit installed
- containerd configured for GPU support
- NVIDIA device plugin manifest included

### 2. Build the Image

Build the GPU-enabled k3s image:

```bash
# Set your desired k3s and CUDA versions
K3S_VERSION="v1.31.5-k3s1"
CUDA_VERSION="12.9.1-base-ubuntu22.04"

# Build the image
docker build \
  --build-arg K3S_TAG=${K3S_VERSION} \
  --build-arg CUDA_TAG=${CUDA_VERSION} \
  -t k3s-gpu:${K3S_VERSION} \
  -f docker/k3s-gpu.Dockerfile \
  .
```

> [!NOTE]
> The version in `k3d-config.yaml` (line 5) must match the tag you build.

## References

- [Official k3d CUDA Guide](https://k3d.io/v5.8.3/usage/advanced/cuda/)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
