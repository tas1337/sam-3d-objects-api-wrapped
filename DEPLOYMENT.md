# SAM-3D-Objects Deployment Guide

Deploy SAM-3D as a Docker container on RunPod (or any GPU server).

## Requirements

- **GPU**: 24GB+ VRAM (A100, L40S, A40, RTX 4090)
- **16GB GPUs** (RTX 3080 Ti, etc.): Only Gaussian Splat (PLY) output works, mesh generation needs 24GB+

---

## Step 1: Build Docker Image Locally

### 1.1 Get HuggingFace Token

1. Go to [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)
2. Accept Meta's license agreement
3. Go to [HuggingFace Tokens](https://huggingface.co/settings/tokens)
4. Create token with `read` access

### 1.2 Download Checkpoints

```bash
export HF_TOKEN=your-token-here

# Install HF CLI
pip install 'huggingface-hub[cli]<1.0'

# Login
huggingface-cli login --token $HF_TOKEN

# Download (~12GB)
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download \
  --repo-type model \
  --local-dir checkpoints/hf-download \
  --max-workers 1 \
  facebook/sam-3d-objects

# Move to correct location
mv checkpoints/hf-download/checkpoints/* checkpoints/
rm -rf checkpoints/hf-download

# Verify
ls -la checkpoints/
# Should see: pipeline.yaml, slat_generator.ckpt, ss_generator.ckpt, etc.
```

### 1.3 Build Docker Image

```bash
docker compose build
```

This builds `sam3d-objects:latest` with:
- All dependencies (conda, pytorch, kaolin, gsplat, nvdiffrast)
- Checkpoints baked in (~12GB)
- Flask API server

**Caching**: Dependencies are cached. Only rebuilds if you change `environments/default.yml` or `pyproject.toml`.

---

## Step 2: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag for your registry
docker tag sam3d-objects:latest YOUR-DOCKERHUB-USERNAME/sam3d-objects:latest

# Push (~25GB upload)
docker push YOUR-DOCKERHUB-USERNAME/sam3d-objects:latest
```

---

## Step 3: Deploy on RunPod

### 3.1 Create Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **Deploy** → **Deploy Pod**
3. Select GPU:
   - **A100 40GB** - Best ($1.50/hr)
   - **L40S 48GB** - Great ($1.00/hr)
   - **A40 48GB** - Good ($0.80/hr)
4. Click **Customize Deployment**
5. Set:
   - **Container Image**: `YOUR-DOCKERHUB-USERNAME/sam3d-objects:latest`
   - **Container Disk**: 50GB
   - **Expose HTTP Ports**: `8000`
6. Click **Deploy**

### 3.2 Wait for Startup

Model loading takes 2-3 minutes. Check logs in RunPod console.

### 3.3 Get Your API URL

RunPod Dashboard → Your Pod → **Connect** → Copy the HTTP URL

Format: `https://YOUR-POD-ID-8000.proxy.runpod.net`

---

## API Reference

### Health Check

```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda": true,
  "gpu": {
    "name": "NVIDIA A100-SXM4-40GB",
    "vram_gb": 40.0
  },
  "queue": {
    "queued": 2,
    "processing": 1,
    "max_queue_size": 10,
    "max_concurrent": 1
  }
}
```

### Queue System

The API uses a queue to process **1 request at a time**. Other requests wait in line.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Submit job to queue (async) |
| `/generate/sync` | POST | Submit and wait for result (sync) |
| `/job/<job_id>` | GET | Check job status & position |
| `/job/<job_id>/download` | GET | Download completed result |
| `/queue` | GET | Get queue stats |

### Generate 3D Model (Async - Recommended)

**Endpoint**: `POST /generate`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | - | Base64 encoded image |
| `image_url` | string | - | URL to download image |
| `seed` | int | 42 | Random seed |
| `output_format` | string | `"glb"` | `"glb"` (mesh) or `"ply"` (gaussian splat) |
| `with_texture` | bool | `true` | Bake textures (GLB only) |

*Either `image` or `image_url` is required.*

**Response:**
```json
{
  "job_id": "abc12345",
  "status": "queued",
  "position": 2,
  "status_url": "/job/abc12345",
  "message": "Job queued at position 2. Poll status_url for updates."
}
```

### Check Job Status

```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/job/abc12345
```

**Response (queued):**
```json
{
  "job_id": "abc12345",
  "status": "queued",
  "position": 2,
  "queue_length": 3,
  "message": "Waiting in queue, position 2"
}
```

**Response (completed):**
```json
{
  "job_id": "abc12345",
  "status": "completed",
  "position": 0,
  "download_url": "/job/abc12345/download",
  "processing_time": 45.2
}
```

### Download Result

```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/job/abc12345/download --output model.glb
```

### Sync Mode (Wait for Result)

If you want the old behavior (wait for result):

```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate/sync \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://...", "output_format": "glb"}' \
  --output model.glb
```

⚠️ This may timeout if queue is long. Use async `/generate` for production.

---

## Example Curl Commands

### Async Flow (Recommended)

**1. Submit job:**
```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "output_format": "glb", "with_texture": true}'
```

**2. Check status (poll until completed):**
```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/job/abc12345
```

**3. Download when ready:**
```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/job/abc12345/download --output cat.glb
```

### Sync Flow (Simple but may timeout)

**GLB Mesh with Textures (Best Quality):**
```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate/sync \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "output_format": "glb", "with_texture": true}' \
  --output cat.glb
```

**GLB Mesh with Vertex Colors (Faster):**
```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate/sync \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=512", "output_format": "glb", "with_texture": false}' \
  --output shoe.glb
```

**Gaussian Splat (PLY):**
```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate/sync \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=512", "output_format": "ply"}' \
  --output burger.ply
```

### Using Base64 Image (Windows PowerShell)

```powershell
$imageBytes = [System.IO.File]::ReadAllBytes("C:\path\to\image.jpg")
$base64 = [Convert]::ToBase64String($imageBytes)
$body = @{image = $base64; output_format = "glb"; with_texture = $true} | ConvertTo-Json
Invoke-WebRequest -Uri "https://YOUR-POD-ID-8000.proxy.runpod.net/generate" -Method POST -Body $body -ContentType "application/json" -OutFile "model.glb"
```

### Using Base64 Image (Git Bash)

```bash
echo '{"image": "' > /tmp/req.json
base64 -w 0 /path/to/image.jpg >> /tmp/req.json
echo '", "output_format": "glb", "with_texture": true}' >> /tmp/req.json

curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d @/tmp/req.json \
  --output model.glb
```

---

## GPU Memory Requirements

| Output | with_texture | VRAM Needed | GPUs |
|--------|--------------|-------------|------|
| PLY (Gaussian Splat) | - | ~14GB | RTX 3080 Ti, RTX 4080+ |
| GLB (Vertex Colors) | false | ~20GB | A40, L40S, A100 |
| GLB (Textured) | true | ~24GB+ | A40, L40S, A100 |

---

## Performance

| GPU | VRAM | GLB Time | PLY Time | Cost/hr |
|-----|------|----------|----------|---------|
| A100 40GB | 40GB | ~45s | ~25s | ~$1.50 |
| L40S | 48GB | ~50s | ~30s | ~$1.00 |
| A40 | 48GB | ~55s | ~30s | ~$0.80 |
| RTX 4090 | 24GB | ~60s | ~35s | ~$0.50 |

---

## Troubleshooting

### "Worker was sent SIGKILL! Perhaps out of memory?"

Your GPU doesn't have enough VRAM. Options:
1. Use `output_format: "ply"` (less memory)
2. Use `with_texture: false` (less memory)
3. Use a GPU with more VRAM

### "No module named 'nvdiffrast'"

Rebuild Docker image - nvdiffrast wasn't installed properly.

### Health check returns `model_loaded: false`

Model is still loading. Wait 2-3 minutes after pod starts.

### "Using center mask (rembg not available)"

Normal - rembg auto-segments the image. If it's not installed, a center crop is used instead.

### Connection refused

Pod isn't ready yet. Check RunPod logs.

---

## Local Testing (24GB+ GPU Only)

If you have a local GPU with 24GB+ VRAM:

```bash
docker run --gpus all -p 8000:8000 sam3d-objects:latest
```

Then test:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "output_format": "glb"}' --output model.glb
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | API port |
| `MAX_CONCURRENT` | 1 | Jobs processed at once (keep at 1 for GPU) |
| `MAX_QUEUE_SIZE` | 10 | Max jobs waiting in queue |

Example:
```bash
docker run --gpus all -p 8000:8000 \
  -e MAX_QUEUE_SIZE=20 \
  sam3d-objects:latest
```

---

## Quick Reference

| Item | Value |
|------|-------|
| Docker Image | `sam3d-objects:latest` |
| Port | 8000 |
| Min GPU VRAM | 24GB (GLB), 16GB (PLY only) |
| Model Size | ~12GB checkpoints |
| HuggingFace Model | `facebook/sam-3d-objects` |
| API Endpoints | `/health`, `/generate`, `/generate/sync`, `/job/<id>`, `/queue` |

