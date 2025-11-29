#!/bin/bash
# ============================================================================
# Build SAM3D Docker Image
# ============================================================================

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-sam3d-objects}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "🔨 Building SAM3D Docker image: ${FULL_IMAGE}"

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found. Run this from the sam-3d-objects-fork directory."
    exit 1
fi

# Check for checkpoints
if [ ! -f "checkpoints/hf/pipeline.yaml" ]; then
    echo "⚠️  Warning: Checkpoints not found locally."
    echo "   The image will download them at runtime if HF_TOKEN is set."
    echo "   For faster startup, download checkpoints first."
fi

# Build with build args
docker build \
    --build-arg HF_TOKEN="${HF_TOKEN:-}" \
    -t "${FULL_IMAGE}" \
    -f Dockerfile \
    .

echo "✅ Build complete: ${FULL_IMAGE}"
echo ""
echo "📋 Next steps:"
echo "   1. Test locally: docker-compose up"
echo "   2. Push to registry: ./scripts/push.sh"

