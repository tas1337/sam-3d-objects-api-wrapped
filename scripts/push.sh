#!/bin/bash
# ============================================================================
# Push SAM3D Docker Image to Registry (for RunPod deployment)
# ============================================================================

set -e

# Configuration - UPDATE THESE FOR YOUR REGISTRY
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
USERNAME="${DOCKER_USERNAME:-your-username}"
IMAGE_NAME="${IMAGE_NAME:-sam3d-objects}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
REMOTE_IMAGE="${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "📤 Pushing SAM3D Docker image to registry"
echo "   Local:  ${LOCAL_IMAGE}"
echo "   Remote: ${REMOTE_IMAGE}"
echo ""

# Check if local image exists
if ! docker image inspect "${LOCAL_IMAGE}" &> /dev/null; then
    echo "❌ Error: Local image '${LOCAL_IMAGE}' not found."
    echo "   Run ./scripts/build.sh first."
    exit 1
fi

# Tag for remote registry
echo "🏷️  Tagging image..."
docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"

# Push to registry
echo "📤 Pushing to registry..."
docker push "${REMOTE_IMAGE}"

echo "✅ Push complete: ${REMOTE_IMAGE}"
echo ""
echo "📋 For RunPod deployment:"
echo "   1. Go to RunPod: https://www.runpod.io/console/pods"
echo "   2. Create a new pod with:"
echo "      - Image: ${REMOTE_IMAGE}"
echo "      - GPU: A100 or similar (24GB+ VRAM recommended)"
echo "      - Expose port: 8000"
echo "   3. Set environment variable: HF_TOKEN=<your-token>"
echo "   4. Access API at: https://your-pod-id-8000.proxy.runpod.net"

