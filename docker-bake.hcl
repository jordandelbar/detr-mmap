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

target "inference-gpu" {
  context = "."
  dockerfile = "docker/inference-gpu.Dockerfile"
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
  context = "."
  dockerfile = "docker/benchmark-gpu.Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["bridge-rt-benchmark-gpu:${TAG}", "${REGISTRY}/bridge-rt-benchmark-gpu:${TAG}"]
}
