#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Deploying bridge-rt to k3s cluster ==="

# Note: For k3s, images should be in a registry or imported with k3s ctr
# Adjust the registry in k8s/overlays/k3s/kustomization.yaml

# Option 1: Import images directly (k3s-specific)
# echo "Importing images to k3s..."
# docker save bridge-rt-gateway:latest | sudo k3s ctr images import -
# docker save bridge-rt-inference:latest | sudo k3s ctr images import -

# Option 2: Push to local registry (recommended for production)
echo "Note: Ensure images are available in registry specified in k8s/overlays/k3s/kustomization.yaml"
echo "Example: docker tag bridge-rt-gateway:latest localhost:5000/bridge-rt-gateway:latest"
echo "         docker push localhost:5000/bridge-rt-gateway:latest"

echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/overlays/k3s

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Check status:"
echo "  kubectl get pods -n bridge-rt"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c gateway --follow"
echo "  kubectl logs -n bridge-rt -l app=bridge-rt -c inference --follow"
echo ""
echo "Delete deployment:"
echo "  kubectl delete -k k8s/overlays/k3s"
