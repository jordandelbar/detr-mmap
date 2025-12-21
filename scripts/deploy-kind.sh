#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-kind}"

echo "=== Deploying bridge-rt to kind cluster ==="

if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "Error: kind cluster '${CLUSTER_NAME}' not found"
    echo "Create one with: kind create cluster --name ${CLUSTER_NAME}"
    exit 1
fi

echo "Loading images into kind cluster..."
kind load docker-image bridge-rt-gateway:latest --name "${CLUSTER_NAME}"
kind load docker-image bridge-rt-inference:latest --name "${CLUSTER_NAME}"

echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/overlays/kind

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Check status:"
echo "  kubectl get pods -n bridge-rt"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c gateway --follow"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c inference --follow"
echo ""
echo "Delete deployment:"
echo "  kubectl delete -k k8s/overlays/kind"
