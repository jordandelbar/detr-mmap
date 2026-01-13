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
  tags = ["bridge-rt-capture:${TAG}", "${REGISTRY}/bridge-rt-capture:${TAG}"]
  args = {
    BINARY_NAME = "capture"
  }
}

target "controller" {
  inherits = ["common"]
  tags = ["bridge-rt-controller:${TAG}", "${REGISTRY}/bridge-rt-controller:${TAG}"]
  args = {
    BINARY_NAME = "controller"
  }
}

target "inference-cpu" {
  inherits = ["common"]
  tags = ["bridge-rt-inference-cpu:${TAG}", "${REGISTRY}/bridge-rt-inference-cpu:${TAG}"]
  args = {
    BINARY_NAME = "inference"
  }
}

target "inference-ort-cuda" {
  contexts = {
    "bridge-rt-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/inference-ort-cuda.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-inference-ort-cuda:${TAG}", "${REGISTRY}/bridge-rt-inference-ort-cuda:${TAG}"]
}

target "inference-trt" {
  contexts = {
    "bridge-rt-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/inference-trt.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-inference-trt:${TAG}", "${REGISTRY}/bridge-rt-inference-trt:${TAG}"]
}

target "gateway" {
  inherits = ["common"]
  tags = ["bridge-rt-gateway:${TAG}", "${REGISTRY}/bridge-rt-gateway:${TAG}"]
  args = {
    BINARY_NAME = "gateway"
  }
}

# Gpu images

target "gpu-base" {
  context = "."
  dockerfile = "docker/gpu-base.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-gpu-base:${TAG}"]
}


target "gpu-benchmark" {
  contexts = {
    "bridge-rt-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/gpu-benchmark.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-benchmark-gpu:${TAG}", "${REGISTRY}/bridge-rt-benchmark-gpu:${TAG}"]
}
