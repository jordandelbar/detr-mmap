#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLUSTER_NAME="${K3D_CLUSTER_NAME:-bridge-rt}"
REGISTRY="localhost:5000"

echo "=== Deploying bridge-rt to k3d cluster ==="

if ! k3d cluster list | grep -q "^${CLUSTER_NAME}"; then
    echo "Error: k3d cluster '${CLUSTER_NAME}' not found"
    echo "Create one with: ./scripts/setup-k3d.sh"
    exit 1
fi

echo "Building Docker images..."
docker compose build

echo "Tagging images for local registry..."
docker tag bridge-rt-gateway:latest ${REGISTRY}/bridge-rt-gateway:latest
docker tag bridge-rt-inference:latest ${REGISTRY}/bridge-rt-inference:latest
docker tag bridge-rt-logic:latest ${REGISTRY}/bridge-rt-logic:latest

echo "Pushing images to k3d registry..."
docker push ${REGISTRY}/bridge-rt-gateway:latest
docker push ${REGISTRY}/bridge-rt-inference:latest
docker push ${REGISTRY}/bridge-rt-logic:latest

echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/overlays/k3d

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Check status:"
echo "  kubectl get pods -n bridge-rt"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c gateway --follow"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c inference --follow"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c logic --follow"
echo ""
echo "Delete deployment:"
echo "  kubectl delete -k k8s/overlays/k3d"
