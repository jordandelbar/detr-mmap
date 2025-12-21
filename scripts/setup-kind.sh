#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-kind}"

echo "=== Setting up kind cluster for bridge-rt ==="

if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "Cluster '${CLUSTER_NAME}' already exists. Delete it first with:"
    echo "  kind delete cluster --name ${CLUSTER_NAME}"
    exit 1
fi

echo "Creating kind cluster with models volume mount..."
kind create cluster --name "${CLUSTER_NAME}" --config kind-config.yaml

echo ""
echo "=== kind cluster created successfully ==="
echo ""
echo "Next steps:"
echo "  1. Build images: ./scripts/build-images.sh"
echo "  2. Deploy: ./scripts/deploy-kind.sh"
