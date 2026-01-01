#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
K3S_TAG="${K3S_TAG:-v1.31.5-k3s1}"
CUDA_TAG="${CUDA_TAG:-12.9.1-base-ubuntu22.04}"
IMAGE_NAME="${IMAGE_NAME:-k3s-gpu}"
IMAGE_TAG="${IMAGE_TAG:-${K3S_TAG}}"

echo "=== Building custom k3s GPU image ==="
echo "K3S version: $K3S_TAG"
echo "CUDA version: $CUDA_TAG"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""

# Build the image
docker build \
  --build-arg K3S_TAG="$K3S_TAG" \
  --build-arg CUDA_TAG="$CUDA_TAG" \
  -t "$IMAGE_NAME:$IMAGE_TAG" \
  -f docker/k3s-gpu.Dockerfile \
  .

echo ""
echo "=== Build complete ==="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "To use this image with k3d:"
echo "  Update k3d-config.yaml to set: image: $IMAGE_NAME:$IMAGE_TAG"
echo "  Then run: ./scripts/setup-k3d.sh"
