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

echo "Labeling agent node as edge device..."
# Wait for nodes to be ready
sleep 5
# Label the agent node to simulate an edge device (e.g., Raspberry Pi)
kubectl label node k3d-bridge-rt-agent-0 node-role=edge --overwrite

echo ""
echo "=== k3d cluster created successfully ==="
echo ""
