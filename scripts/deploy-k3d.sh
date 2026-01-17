#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLUSTER_NAME="${K3D_CLUSTER_NAME:-detr-mmap}"
REGISTRY="localhost:5000"

echo "=== Deploying detr-mmap to k3d cluster ==="

if ! k3d cluster list | grep -q "^${CLUSTER_NAME}"; then
    echo "Error: k3d cluster '${CLUSTER_NAME}' not found"
    echo "Create one with: ./scripts/setup-k3d.sh"
    exit 1
fi

echo "Building and tagging Docker images..."
docker buildx bake --set "*.platform=linux/amd64"
docker buildx bake gpu-base --set "*.platform=linux/amd64"
docker buildx bake inference-trt --set "*.platform=linux/amd64"

echo "Pushing images to k3d registry..."
docker push ${REGISTRY}/detr-mmap-capture:latest
docker push ${REGISTRY}/detr-mmap-inference-trt:latest
docker push ${REGISTRY}/detr-mmap-gateway:latest
docker push ${REGISTRY}/detr-mmap-controller:latest

echo "Ensuring node labels are set..."
kubectl label node k3d-detr-mmap-agent-0 node-role=edge --overwrite
kubectl label node k3d-detr-mmap-server-0 node-role=central --overwrite

echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/overlays/k3d-gpu

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Check status:"
echo "  kubectl get pods -n detr-mmap"
echo "  kubectl logs -n detr-mmap -l app=detr-mmap -c capture --follow"
echo "  kubectl logs -n detr-mmap -l app=detr-mmap -c inference --follow"
echo "  kubectl logs -n detr-mmap -l app=detr-mmap -c gateway --follow"
echo ""
