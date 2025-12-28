variable "REGISTRY" {
  default = "localhost:5000"
}

variable "TAG" {
  default = "latest"
}

group "default" {
  targets = ["gateway", "inference", "logic"]
}

target "common" {
  context = "."
  dockerfile = "docker/Dockerfile"
  platforms = ["linux/amd64"]
}

target "gateway" {
  inherits = ["common"]
  tags = ["bridge-rt-gateway:${TAG}", "${REGISTRY}/bridge-rt-gateway:${TAG}"]
  args = {
    BINARY_NAME = "gateway"
  }
}

target "inference" {
  inherits = ["common"]
  tags = ["bridge-rt-inference:${TAG}", "${REGISTRY}/bridge-rt-inference:${TAG}"]
  args = {
    BINARY_NAME = "inference"
  }
}

target "gpu-base" {
  context = "."
  dockerfile = "docker/gpu-base.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-gpu-base:${TAG}"]
}

target "inference-gpu" {
  contexts = {
    "bridge-rt-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/gpu-inference.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-inference-gpu:${TAG}", "${REGISTRY}/bridge-rt-inference-gpu:${TAG}"]
}

target "logic" {
  inherits = ["common"]
  tags = ["bridge-rt-logic:${TAG}", "${REGISTRY}/bridge-rt-logic:${TAG}"]
  args = {
    BINARY_NAME = "logic"
  }
}

target "benchmark-gpu" {
  contexts = {
    "bridge-rt-gpu-base:latest" = "target:gpu-base"
  }
  context = "."
  dockerfile = "docker/gpu-benchmark.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-benchmark-gpu:${TAG}", "${REGISTRY}/bridge-rt-benchmark-gpu:${TAG}"]
}
