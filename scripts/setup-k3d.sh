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
echo "Registry available at: localhost:5000"
echo ""
echo "Cluster topology:"
echo "  - Server node (k3d-bridge-rt-server-0): Control plane"
echo "  - Agent node (k3d-bridge-rt-agent-0): Edge device (labeled: node-role=edge)"
echo ""
echo "All Bridge-RT pods will be scheduled on the agent node (simulating Raspberry Pi)"
echo ""
echo "Next steps:"
echo "  1. Build and push images: ./scripts/build-images.sh"
echo "  2. Deploy: ./scripts/deploy-k3d.sh"
