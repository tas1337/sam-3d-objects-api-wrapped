# SAM 3D Objects - Docker API Wrapper

REST API wrapper for Meta's [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects). Converts images to 3D models.

**Before starting:** You must request access to the model checkpoints on HuggingFace at [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) and accept Meta's license agreement. Then create a HuggingFace token with read access.

**Automation:** The `deploy_docker_hub.sh` script handles the entire build process automatically. It spins up a temporary DigitalOcean build server, downloads the 12GB model checkpoints from HuggingFace, builds both the API server and serverless Docker images, pushes them to Docker Hub, and then deletes the build server. This avoids needing a local machine with enough disk space and bandwidth for the large model files.

---

## Quick Start

### 1. Automated Build & Push to Docker Hub

Create `scripts/.env`:
```bash
DOCKER_USERNAME=your-dockerhub-username
DOCKER_TOKEN=your-dockerhub-token
GITHUB_TOKEN=your-github-token
HF_TOKEN=your-huggingface-token
DO_TOKEN=your-digitalocean-token
SSH_KEY_PATH=~/.ssh/id_ed25519
```

Run the deployment script:
```bash
./scripts/deploy_docker_hub.sh
```

This creates a temporary DigitalOcean droplet, downloads checkpoints, builds the Docker image, pushes to Docker Hub, and auto-deletes the server.

### 2. Deploy to RunPod Serverless

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Container Image:** `your-dockerhub-username/sam3d-objects-serverless:latest`
   - **GPU Type:** A100 80GB (recommended for high quality)
   - **Min Workers:** 0
   - **Max Workers:** 1-5

### 3. Use the API

RunPod will give you an endpoint and API key. Use it like this:

```bash
# Submit job
curl -X POST https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/run \
  -H "Authorization: Bearer YOUR-RUNPOD-API-KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/image.jpg",
      "output_format": "glb"
    }
  }'

# Check status
curl https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/status/JOB-ID \
  -H "Authorization: Bearer YOUR-RUNPOD-API-KEY"

# Get result (base64 encoded GLB in output.model)
```

---

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_url` | string | - | Image URL |
| `image` | string | - | Base64 encoded image |
| `output_format` | string | `"glb"` | `"glb"` or `"ply"` |
| `with_texture` | bool | `true` | Bake textures (GLB only) |
| `texture_size` | int | `2048` | 1024, 2048, or 4096 |
| `simplify` | float | `0.0` | 0.0 (max detail) to 0.95 |
| `inference_steps` | int | `50` | 25, 50, or 100 |
| `nviews` | int | `200` | 100, 200, or 300 |

---

## Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Detailed deployment guide
- **[META-README.md](META-README.md)** - Original Meta research docs

## Credits

Based on [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects) by Meta.
