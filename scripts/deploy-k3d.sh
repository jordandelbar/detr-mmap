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

echo "Building and tagging Docker images..."
docker buildx bake --set "*.platform=linux/amd64"
docker buildx bake gpu-base --set "*.platform=linux/amd64"
docker buildx bake gpu-inference --set "*.platform=linux/amd64"

echo "Pushing images to k3d registry..."
docker push ${REGISTRY}/bridge-rt-capture:latest
docker push ${REGISTRY}/bridge-rt-inference-gpu:latest
docker push ${REGISTRY}/bridge-rt-logic:latest

echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/overlays/k3d

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Check status:"
echo "  kubectl get pods -n bridge-rt"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c capture --follow"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c inference --follow"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c logic --follow"
echo ""
echo "Delete deployment:"
echo "  kubectl delete -k k8s/overlays/k3d"
