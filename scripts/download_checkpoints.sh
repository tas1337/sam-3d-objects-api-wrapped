#!/bin/bash
# ============================================================================
# Download SAM3D Checkpoints from HuggingFace
# Requires HF_TOKEN environment variable
# ============================================================================

set -e

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set"
    echo ""
    echo "To get a token:"
    echo "  1. Go to https://huggingface.co/settings/tokens"
    echo "  2. Create a new token with 'read' access"
    echo "  3. Run: export HF_TOKEN=your-token-here"
    echo "  4. Run this script again"
    exit 1
fi

echo "🔄 Downloading SAM3D checkpoints from HuggingFace..."
echo ""

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

# Download checkpoints
huggingface-cli download \
    --repo-type model \
    --local-dir checkpoints/hf-download \
    --max-workers 4 \
    facebook/sam-3d-objects

# Move to correct location
if [ -d "checkpoints/hf-download/checkpoints" ]; then
    rm -rf checkpoints/hf
    mv checkpoints/hf-download/checkpoints checkpoints/hf
fi
rm -rf checkpoints/hf-download

echo ""
echo "✅ Checkpoints downloaded successfully!"
echo "   Location: checkpoints/hf/"
ls -lh checkpoints/hf/

