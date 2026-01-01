#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLUSTER_NAME="${K3D_CLUSTER_NAME:-bridge-rt}"

echo "=== Setting up k3d cluster for bridge-rt ==="

if k3d cluster list | grep -q "^${CLUSTER_NAME}"; then
    echo "Cluster '${CLUSTER_NAME}' already exists. Delete it first with:"
    echo "  k3d cluster delete ${CLUSTER_NAME}"
    exit 1
fi

echo "Creating k3d cluster..."
k3d cluster create --config k3d-config.yaml

echo ""
echo "=== k3d cluster created successfully ==="
echo ""
echo "Registry available at: localhost:5000"
echo ""
echo "Next steps:"
echo "  1. Build and push images: ./scripts/build-images.sh"
echo "  2. Deploy: ./scripts/deploy-k3d.sh"
