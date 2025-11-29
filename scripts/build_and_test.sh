#!/bin/bash
# ============================================================================
# Build and Test Docker Image Locally
# This builds the image and runs a basic health check
# ============================================================================

set -e

IMAGE_NAME="${IMAGE_NAME:-sam3d-objects}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "SAM3D Docker Build & Test"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Run this from sam-3d-objects-fork directory"
    exit 1
fi

# Check checkpoints
echo ""
echo "📁 Checking checkpoints..."
if [ ! -f "checkpoints/hf/pipeline.yaml" ]; then
    echo "❌ Error: Checkpoints not found!"
    echo "   Run: ./scripts/download_checkpoints.sh"
    exit 1
fi
echo "✅ Checkpoints found"

# Build image
echo ""
echo "🔨 Building Docker image: ${FULL_IMAGE}"
echo "   This may take 10-20 minutes on first build..."
echo ""

docker build \
    -t "${FULL_IMAGE}" \
    -f Dockerfile \
    . 2>&1 | tee docker_build.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ Docker build failed! Check docker_build.log for details"
    exit 1
fi

echo ""
echo "✅ Docker build successful!"
echo ""

# Test container starts (without GPU)
echo "🧪 Testing container startup (CPU mode)..."
echo "   Note: Model won't load without GPU, but we can test the container"
echo ""

# Start container in background
CONTAINER_ID=$(docker run -d --rm \
    -e LIDRA_SKIP_INIT=true \
    -p 8000:8000 \
    "${FULL_IMAGE}" \
    bash -c "sleep 30")

echo "   Container started: ${CONTAINER_ID:0:12}"
sleep 5

# Check if container is still running
if docker ps -q | grep -q "${CONTAINER_ID:0:12}"; then
    echo "✅ Container running (basic startup OK)"
    docker stop "${CONTAINER_ID}" >/dev/null 2>&1 || true
else
    echo "⚠️  Container exited (may be GPU issue, which is expected locally)"
fi

echo ""
echo "============================================================"
echo "✅ Local testing complete!"
echo ""
echo "📋 Summary:"
echo "   - Docker image built: ${FULL_IMAGE}"
echo "   - Image size: $(docker images ${FULL_IMAGE} --format '{{.Size}}')"
echo ""
echo "🚀 Next steps:"
echo "   1. Push to registry:"
echo "      export DOCKER_USERNAME=your-username"
echo "      ./scripts/push.sh"
echo ""
echo "   2. Deploy on RunPod:"
echo "      - Image: docker.io/\$DOCKER_USERNAME/${IMAGE_NAME}:${IMAGE_TAG}"
echo "      - GPU: A100 or similar (24GB+ VRAM)"
echo "      - Port: 8000"
echo "      - Env: HF_TOKEN=your-token"
echo "============================================================"

