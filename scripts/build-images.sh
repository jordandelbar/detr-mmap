#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Building bridge-rt Docker images ==="

echo "Building gateway image..."
docker build \
  -f docker/gateway.Dockerfile \
  -t bridge-rt-gateway:latest \
  .

echo "Building inference image..."
docker build \
  -f docker/inference.Dockerfile \
  -t bridge-rt-inference:latest \
  .

echo ""
echo "=== Build complete ==="
echo "Gateway image: bridge-rt-gateway:latest"
echo "Inference image: bridge-rt-inference:latest"
echo ""
echo "Next steps:"
echo "  For kind: ./scripts/deploy-kind.sh"
echo "  For k3s: ./scripts/deploy-k3s.sh"
