#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLUSTER_NAME="${K3D_CLUSTER_NAME:-detr-mmap}"

echo "=== Setting up k3d cluster for detr-mmap ==="

if k3d cluster list | grep -q "^${CLUSTER_NAME}"; then
    echo "Cluster '${CLUSTER_NAME}' already exists. Delete it first with:"
    echo "  k3d cluster delete ${CLUSTER_NAME}"
    exit 1
fi

echo "Creating k3d cluster..."
k3d cluster create --config k3d-config.yaml

echo "Labeling nodes..."
# Wait for nodes to be ready
sleep 5
# Label the agent node to simulate an edge device (e.g., Raspberry Pi)
kubectl label node k3d-detr-mmap-agent-0 node-role=edge --overwrite
# Label the server node as central (for MQTT broker and other central services)
kubectl label node k3d-detr-mmap-server-0 node-role=central --overwrite

echo ""
echo "=== k3d cluster created successfully ==="
echo ""
