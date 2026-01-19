variable "REGISTRY" {
  default = "localhost:5000"
}

variable "TAG" {
  default = "latest"
}

group "default" {
  targets = ["capture", "controller", "inference-cpu", "gateway"]
}

target "common" {
  context = "."
  dockerfile = "docker/Dockerfile"
  platforms = ["linux/amd64"]
}

target "capture" {
  inherits = ["common"]
  tags = ["detr-mmap-capture:${TAG}", "${REGISTRY}/detr-mmap-capture:${TAG}"]
  args = {
    BINARY_NAME = "capture"
  }
}

target "controller" {
  inherits = ["common"]
  tags = ["detr-mmap-controller:${TAG}", "${REGISTRY}/detr-mmap-controller:${TAG}"]
  args = {
    BINARY_NAME = "controller"
  }
}

target "inference-cpu" {
  inherits = ["common"]
  tags = ["detr-mmap-inference-cpu:${TAG}", "${REGISTRY}/detr-mmap-inference-cpu:${TAG}"]
  args = {
    BINARY_NAME = "inference"
  }
}

target "inference-ort-cuda" {
  contexts = {
    "detr-mmap-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/inference-ort-cuda.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["detr-mmap-inference-ort-cuda:${TAG}", "${REGISTRY}/detr-mmap-inference-ort-cuda:${TAG}"]
}

target "inference-trt" {
  contexts = {
    "detr-mmap-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/inference-trt.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["detr-mmap-inference-trt:${TAG}", "${REGISTRY}/detr-mmap-inference-trt:${TAG}"]
}

target "gateway" {
  inherits = ["common"]
  tags = ["detr-mmap-gateway:${TAG}", "${REGISTRY}/detr-mmap-gateway:${TAG}"]
  args = {
    BINARY_NAME = "gateway"
  }
}

# Gpu images

target "gpu-base" {
  context = "."
  dockerfile = "docker/gpu-base.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["detr-mmap-gpu-base:${TAG}"]
}


target "gpu-benchmark" {
  contexts = {
    "detr-mmap-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/gpu-benchmark.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["detr-mmap-benchmark-gpu:${TAG}", "${REGISTRY}/detr-mmap-benchmark-gpu:${TAG}"]
}
