# SAM-3D-Objects Deployment Guide

Deploy SAM-3D as a Docker container on RunPod (or any GPU server).

## 📋 Setup Progress Checklist

**For Manual RunPod Setup:**
- [ ] Step 0: Clone repository
- [ ] Step 1.2: Download HuggingFace checkpoints
- [ ] Step 2.5.1: Install system dependencies
- [ ] Step 2.5.2: Install Miniconda
- [ ] Step 2.5.3: Create conda environment
- [ ] Step 2.5.4: Install Python dependencies
- [ ] Step 2.5.5: Verify checkpoints location
- [ ] Step 2.5.6: Run API server

**For Docker Setup:**
- [ ] Step 0: Clone repository
- [ ] Step 1.1: Get HuggingFace token
- [ ] Step 1.2: Download checkpoints
- [ ] Step 1.3: Build Docker image
- [ ] Step 2: Push to Docker Hub
- [ ] Step 3: Deploy on RunPod

---

## Step 0: Clone Repository

**To clone the repo (first time):**
```bash
git clone https://<YOUR_TOKEN>@github.com/tas1337/sam-3d-objects.git
```

**Or using oauth2 format:**
```bash
git clone https://oauth2:<YOUR_TOKEN>@github.com/tas1337/sam-3d-objects.git
```

**To pull updates (if already cloned):**
```bash
cd sam-3d-objects
git pull
```

**If pull asks for credentials, update the remote URL:**
```bash
git remote set-url origin https://<YOUR_TOKEN>@github.com/tas1337/sam-3d-objects.git
git pull
```

Replace `<YOUR_TOKEN>` with your personal access token.

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

---

## Step 2.5: Manual Setup on RunPod (Without Docker)

If you're setting up manually on a RunPod (not using Docker), follow these steps:

### 2.5.1 Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y wget curl git build-essential ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### 2.5.2 Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
rm /tmp/miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
```

### 2.5.3 Create Conda Environment

```bash
cd sam-3d-objects  # or wherever you cloned the repo

# Configure conda (exact same as Docker)
conda config --set remote_read_timeout_secs 600
conda config --set remote_connect_timeout_secs 120
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment
conda env create -f environments/default.yml
conda clean -ya
```

### 2.5.4 Install Python Dependencies

```bash
# CRITICAL: Source conda.sh first (exact same as Docker)
# Find your conda installation path
CONDA_PATH=$(conda info --base)
echo "Conda path: $CONDA_PATH"

# Source conda.sh to enable conda activate
source $CONDA_PATH/etc/profile.d/conda.sh

# Activate environment (exact same as Docker)
conda activate sam3d-objects

# Verify you're using the right pip (should show conda env path)
which pip
# Should show: $CONDA_PATH/envs/sam3d-objects/bin/pip (e.g., /root/miniconda3/envs/sam3d-objects/bin/pip)

# Set environment variables (exact same as Docker)
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
export CUDA_HOME=/usr/local/cuda
export LIDRA_SKIP_INIT=true
export CPATH=${CUDA_HOME}/include:${CPATH}

# Install HuggingFace CLI (exact same order as Docker)
pip install --no-cache-dir 'huggingface-hub[cli]<1.0'

# IMPORTANT: Install PyTorch FIRST (auto-gptq needs it to build)
# This fixes the "No module named 'torch'" error on RunPod
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchaudio==2.5.1+cu121

# Verify PyTorch is installed
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

# Install main package with dev and p3d extras (Docker does these together)
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[dev]'
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[p3d]'

# Install inference extras (Docker does this separately)
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[inference]'

# Install numpy 1.26.4 FIRST (kaolin requires numpy<2.0)
# This prevents rembg from upgrading to numpy 2.x
pip install --no-cache-dir "numpy==1.26.4"

# Install API dependencies + rembg (exact same as Docker)
# Note: rembg may show numpy conflict warnings, but numpy 1.26.4 will work
pip install --no-cache-dir flask flask-cors requests gunicorn rembg

# Reinstall numpy 1.26.4 to ensure kaolin compatibility (exact same as Docker)
pip install --no-cache-dir "numpy==1.26.4" --force-reinstall

# Install nvdiffrast (needs CUDA headers, exact same as Docker)
pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Apply hydra patch AFTER all dependencies are installed (hydra needs to be installed first)
if [ -f patching/hydra ]; then python patching/hydra; fi
```

### 2.5.5 Verify Checkpoints Location

```bash
# Check if checkpoints are in the right place
# They should be in: /workspace/sam-3d-objects/checkpoints/
cd /workspace/sam-3d-objects

ls -la checkpoints/
# OR if they're in checkpoints/hf/
ls -la checkpoints/hf/

# Should see: pipeline.yaml, slat_generator.ckpt, ss_generator.ckpt, etc.

# If checkpoints are in /workspace/checkpoints/ (outside the repo), move them:
if [ -d "/workspace/checkpoints" ] && [ ! -d "/workspace/sam-3d-objects/checkpoints" ]; then
    mv /workspace/checkpoints /workspace/sam-3d-objects/checkpoints
    echo "✅ Moved checkpoints to correct location"
fi
```

### 2.5.6 Run API Server

**Run in background (Recommended - prevents crashes):**

```bash
# Make sure you're in the repo directory
cd /workspace/sam-3d-objects

# Source conda.sh and activate environment
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate sam3d-objects

# Kill any existing gunicorn processes
pkill -f gunicorn || true

# Start in background with auto-restart (restarts worker after 10 requests to prevent memory leaks)
nohup gunicorn --bind 0.0.0.0:8000 \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --max-requests 10 \
    --max-requests-jitter 5 \
    --preload \
    --log-level info \
    --access-logfile /tmp/gunicorn_access.log \
    --error-logfile /tmp/gunicorn_error.log \
    api_server:app > /tmp/api_server.log 2>&1 &

# Check it's running
ps aux | grep gunicorn

# Monitor logs
tail -f /tmp/api_server.log
```

**Run in foreground (for testing):**

```bash
cd /workspace/sam-3d-objects
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate sam3d-objects

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export LIDRA_SKIP_INIT=true

# Run the server
gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 2 --timeout 300 api_server:app
```

**Note:** The `--max-requests 10` flag restarts the worker after 10 jobs to prevent memory leaks. This is recommended to prevent crashes after multiple generations.

**Status Tracking:**
- ✅ Repository cloned
- ✅ HuggingFace checkpoints downloaded
- ⏳ Conda environment created
- ⏳ Python dependencies installed
- ⏳ API server running

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
| `texture_size` | int | `4096` | **Texture resolution** (1024, 2048, 4096). Higher = better quality but slower. Default: 4096 (maximum) |
| `simplify` | float | `0.0` | **Mesh simplification** (0.0 = no simplification/max detail, 0.95 = aggressive). Lower = more detail. Default: 0.0 (maximum detail) |
| `inference_steps` | int | `100` | **Diffusion steps** (25 = fast, 50 = high, 100 = ultra). More = better quality but slower. Default: 100 (maximum) |
| `nviews` | int | `300` | **Texture baking views** (100 = default, 200 = high, 300 = ultra). More = better texture but slower. Default: 300 (maximum) |

**Quality Presets:**
- **Maximum Quality** ⭐ (default): `texture_size=4096`, `simplify=0.0`, `inference_steps=100`, `nviews=300` - Best quality, slowest (~5-10 min per job)
- **High Quality** (faster): `texture_size=2048`, `simplify=0.0`, `inference_steps=50`, `nviews=200` - Good quality, faster (~3-5 min per job)
- **Balanced** (fastest): `texture_size=2048`, `simplify=0.0`, `inference_steps=25`, `nviews=100` - Default quality, fastest (~2-3 min per job)

**Note:** Defaults are set to **maximum quality**. You can omit quality parameters to automatically use maximum settings, or specify lower values for faster generation.

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

**1. Submit job (Maximum Quality - Default):**
```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "output_format": "glb", "with_texture": true, "texture_size": 4096, "simplify": 0.0, "inference_steps": 100, "nviews": 300}'
```

**Or use defaults (already maximum quality):**
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

**Quick status check loop:**
```bash
# Check status every 10 seconds until completed
while true; do
  STATUS=$(curl -s https://YOUR-POD-ID-8000.proxy.runpod.net/job/abc12345 | grep -o '"status":"[^"]*"')
  echo "$(date): $STATUS"
  if echo "$STATUS" | grep -q "completed"; then
    echo "Job completed! Downloading..."
    curl https://YOUR-POD-ID-8000.proxy.runpod.net/job/abc12345/download --output model.glb
    break
  fi
  sleep 10
done
```

### Sync Flow (Simple but may timeout)

**GLB Mesh with Textures (Maximum Quality - Default):**
```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate/sync \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "output_format": "glb", "with_texture": true, "texture_size": 4096, "simplify": 0.0, "inference_steps": 100, "nviews": 300}' \
  --output cat.glb
```

**GLB Mesh with Textures (High Quality - Faster):**
```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate/sync \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "output_format": "glb", "with_texture": true, "texture_size": 2048, "simplify": 0.0, "inference_steps": 50, "nviews": 200}' \
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

### "CondaError: Run 'conda init' before 'conda activate'"

**Solution:** Source conda.sh first (exact same as Docker):
```bash
# Find conda path and source it
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate sam3d-objects
```

### "Building cuda extension requires PyTorch (>=1.13.0) being installed, please install PyTorch first: No module named 'torch'"

**RunPod only**: This happens when `auto-gptq` tries to build before PyTorch is installed, OR when the conda environment isn't activated.

**Solution:**
1. **Make sure conda environment is activated:**
   ```bash
   # Source conda.sh first
   CONDA_PATH=$(conda info --base)
   source $CONDA_PATH/etc/profile.d/conda.sh
   conda activate sam3d-objects
   which pip  # Should show conda env path, not system Python
   ```

2. **Install PyTorch first (see Step 2.5.4):**
   ```bash
   pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchaudio==2.5.1+cu121
   python -c "import torch; print('PyTorch OK')"  # Verify it works
   ```

3. **Then install package dependencies:**
   ```bash
   pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[dev]'
   ```

**If you see `/root/miniconda3/lib/` in error messages**, you're using system Python instead of the conda environment. Activate the conda environment first!

### "ERROR: pip's dependency resolver... numpy... incompatible"

**Expected warnings**: You may see numpy version conflict warnings when installing `rembg`. This is normal:
- `rembg` pulls `opencv-python-headless` which wants `numpy>=2.0`
- `kaolin` requires `numpy<2.0`
- The final `numpy==1.26.4` installation fixes this for kaolin compatibility

**Solution**: These warnings are safe to ignore. The final `numpy==1.26.4` installation ensures kaolin works correctly. If you see these warnings, the installation is still successful - just verify numpy version at the end:
```bash
python -c "import numpy; print(f'NumPy {numpy.__version__}')"  # Should show 1.26.4
```

### Health check returns `model_loaded: false`

Model is still loading. Wait 2-3 minutes after pod starts.

### "Using center mask (rembg not available)"

Normal - rembg auto-segments the image. If it's not installed, a center crop is used instead.

### Connection refused

Pod isn't ready yet. Check RunPod logs.

### Job Stuck in Queue (processing: 0, queued: 1)

**Problem:** Job is queued but not processing - worker thread may have crashed.

**Solution:** Restart the server:
```bash
# On pod
pkill -f gunicorn
cd /workspace/sam-3d-objects
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate sam3d-objects

nohup gunicorn --bind 0.0.0.0:8000 \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --max-requests 10 \
    --max-requests-jitter 5 \
    api_server:app > /tmp/api_server.log 2>&1 &
```

The stuck job will be lost, but you can resubmit. The worker monitor thread will auto-restart the worker if it crashes again.

### API Server Crashes After Job Completion

**Problem:** Server crashes after completing a generation job.

**Solution:** Use `--max-requests` flag to restart workers periodically:

```bash
# Kill old server
pkill -f gunicorn

# Restart with max-requests (restarts worker after 10 jobs to prevent memory leaks)
nohup gunicorn --bind 0.0.0.0:8000 \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --max-requests 10 \
    --max-requests-jitter 5 \
    --preload \
    api_server:app > /tmp/api_server.log 2>&1 &
```

This automatically restarts the worker after 10 requests, preventing memory leaks and crashes.

**Note:** Generated files are stored in `/tmp` and automatically cleaned up after 1 hour. Download your results promptly after job completion.

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
| File Storage | `/tmp` (auto-cleaned after 1 hour) |
| Processing Time | ~2-3 min (model load) + ~2-3 min per job |

## File Storage & Cleanup

- **Generated files** are stored in `/tmp` with random names (e.g., `/tmp/tmpXXXXXX.glb`)
- **Auto-cleanup**: Files are automatically deleted 1 hour after job completion
- **Download promptly**: Make sure to download your results within 1 hour
- **Multiple downloads**: You can download the same job multiple times within the 1-hour window

