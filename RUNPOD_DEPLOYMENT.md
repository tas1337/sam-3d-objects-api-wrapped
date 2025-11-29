# RunPod Deployment Guide for SAM-3D-Objects

This guide walks you through deploying SAM-3D-Objects as a Docker container on RunPod servers.

## Overview

The deployment consists of:
1. **Docker container** with all dependencies pre-installed
2. **Flask API server** exposing endpoints for 3D mesh generation
3. **GLB output** with textures, optimized for web download

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed locally
- [HuggingFace account](https://huggingface.co/) with access token
- [RunPod account](https://www.runpod.io/) with GPU credits
- (Optional) Docker Hub or other container registry account

## Quick Start

### 1. Get HuggingFace Token

The SAM-3D model requires accepting Meta's license:

1. Visit [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)
2. Accept the license agreement
3. Go to [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
4. Create a token with `read` access

```bash
export HF_TOKEN=your-token-here
```

### 2. Download Checkpoints (Optional but Recommended)

Pre-downloading checkpoints makes the Docker image self-contained:

```bash
cd sam-3d-objects-fork
./scripts/download_checkpoints.sh
```

### 3. Build Docker Image

```bash
./scripts/build.sh
```

Or with custom image name:

```bash
IMAGE_NAME=myregistry/sam3d IMAGE_TAG=v1.0 ./scripts/build.sh
```

### 4. Test Locally (Optional)

```bash
# Using docker-compose
docker-compose up

# Or manually
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    -v ./checkpoints:/workspace/sam-3d-objects/checkpoints:ro \
    sam3d-objects:latest
```

Test the API:

```bash
python scripts/test_api.py --url http://localhost:8000 --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png
```

### 5. Push to Registry

```bash
# Set your registry details
export DOCKER_REGISTRY=docker.io
export DOCKER_USERNAME=your-username

# Login to Docker Hub
docker login

# Push
./scripts/push.sh
```

### 6. Deploy on RunPod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click "Deploy" → "Custom Pod"
3. Configure:
   - **Image**: `docker.io/your-username/sam3d-objects:latest`
   - **GPU**: A100 (40GB), L40S, or similar (24GB+ VRAM recommended)
   - **Expose HTTP Port**: 8000
   - **Environment Variables**:
     - `HF_TOKEN`: Your HuggingFace token
4. Click "Deploy"

Wait for the pod to start (model loading takes ~2-3 minutes).

## API Endpoints

### Health Check

```bash
curl https://your-pod-id-8000.proxy.runpod.net/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "output_format": "glb"
}
```

### Generate 3D Mesh (Base64 Response)

```bash
curl -X POST https://your-pod-id-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{
    "image": "BASE64_ENCODED_IMAGE",
    "seed": 42,
    "with_texture": true
  }'
```

Response:
```json
{
  "success": true,
  "model_data": "BASE64_ENCODED_GLB",
  "format": "glb",
  "file_size_mb": 1.23,
  "inference_time_s": 45.2
}
```

### Generate 3D Mesh (File Download)

```bash
curl -X POST https://your-pod-id-8000.proxy.runpod.net/generate-file \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "seed": 42
  }' \
  --output model.glb
```

### Generate Gaussian Splat (PLY)

```bash
curl -X POST https://your-pod-id-8000.proxy.runpod.net/generate-ply \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_ENCODED_IMAGE"}' \
  --output splat.ply
```

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | required* | Base64 encoded image |
| `image_url` | string | required* | URL to download image |
| `seed` | int | 42 | Random seed for reproducibility |
| `simplify` | float | 0.95 | Mesh simplification ratio (0.0-1.0) |
| `texture_size` | int | 1024 | Texture resolution (512, 1024, 2048) |
| `with_texture` | bool | true | Bake textures onto mesh |
| `mask_mode` | string | "alpha" | Mask creation: "alpha", "center", "full" |

*Either `image` or `image_url` is required.

## Performance Tips

### GPU Recommendations

| GPU | VRAM | Inference Time | Recommended |
|-----|------|----------------|-------------|
| A100 (40GB) | 40GB | ~30-45s | ✅ Best |
| A100 (80GB) | 80GB | ~30-45s | ✅ Best |
| L40S | 48GB | ~40-50s | ✅ Great |
| A10G | 24GB | ~50-60s | ✅ Good |
| RTX 4090 | 24GB | ~50-60s | ✅ Good |
| T4 | 16GB | ~90-120s | ⚠️ Slow |

### Optimization Settings

For faster inference (lower quality):
```json
{
  "simplify": 0.98,
  "texture_size": 512,
  "with_texture": false
}
```

For best quality (slower):
```json
{
  "simplify": 0.9,
  "texture_size": 2048,
  "with_texture": true
}
```

## Troubleshooting

### Pod won't start
- Check that HF_TOKEN is set correctly
- Ensure you accepted the license on HuggingFace
- Check pod logs for errors

### Out of memory
- Use a GPU with more VRAM
- Reduce `texture_size` to 512
- Set `with_texture: false`

### Slow inference
- Upgrade to a faster GPU (A100 recommended)
- First inference is slower due to model loading

### API returns 500 error
- Check pod logs: `kubectl logs <pod-name>`
- Ensure checkpoints are downloaded
- Verify CUDA is available

## Cost Estimation

| GPU | $/hour | Inference/hour | $/inference |
|-----|--------|----------------|-------------|
| A100 | ~$1.50 | ~80-100 | ~$0.02 |
| L40S | ~$1.00 | ~70-90 | ~$0.01 |
| A10G | ~$0.40 | ~50-60 | ~$0.008 |

## Security Notes

1. **HF_TOKEN**: Never commit to git. Use environment variables.
2. **API Access**: Consider adding authentication for production.
3. **Rate Limiting**: Implement rate limiting for public endpoints.

## Support

- [SAM-3D Paper](https://arxiv.org/abs/2511.16624)
- [SAM-3D Website](https://ai.meta.com/sam3d/)
- [RunPod Docs](https://docs.runpod.io/)

