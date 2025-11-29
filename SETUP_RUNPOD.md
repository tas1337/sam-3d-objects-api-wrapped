# RunPod Setup Guide

## Run Directly on GPU Pod

### 1. Create GPU Pod
- Template: `runpod/base:0.7.0-ubuntu2004`
- GPU: A100 40GB or L40S or A40 (need 24GB+ VRAM)
- Container Disk: 50GB
- Volume Disk: 100GB
- **Expose HTTP Port: 8888**

### 2. Clone Repo
```bash
cd /workspace
git clone https://github.com/tas1337/sam-3d-objects.git
cd sam-3d-objects
```

### 3. Download Checkpoints
```bash
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens
# Must accept license first: https://huggingface.co/facebook/sam-3d-objects

export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download \
  --repo-type model \
  --local-dir checkpoints/hf-download \
  --max-workers 1 \
  facebook/sam-3d-objects

mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download

# Verify (~12GB total)
ls -la checkpoints/hf/
```

### 4. Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
export PATH=/opt/conda/bin:$PATH
```

### 5. Accept Conda TOS
```bash
conda config --set remote_read_timeout_secs 600
conda config --set remote_connect_timeout_secs 120
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 6. Create Conda Environment (10-15 min)
```bash
conda env create -f environments/default.yml
```

### 7. Activate Environment
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate sam3d-objects
```

### 8. Set Environment Variables
```bash
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
export CUDA_HOME=/usr/local/cuda
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 9. Install Packages (EXACT ORDER - IMPORTANT!)
```bash
# Core dependencies
pip install -e '.[dev]'

# PyTorch3D (separate step due to dependency issues)
pip install -e '.[p3d]'

# Inference dependencies (includes kaolin, gsplat)
pip install -e '.[inference]'

# Apply patches
./patching/hydra

# Install nvdiffrast from source
pip install git+https://github.com/NVlabs/nvdiffrast.git

# Flask for API
pip install flask flask-cors gunicorn rembg
```

### 10. Kill Jupyter & Run API Server
```bash
pkill -f jupyter
PORT=8888 python api_server.py
```

---

## Test API from Local Machine

### Get Your Pod URL
RunPod Dashboard → Your Pod → Connect → HTTP Services
URL format: `https://YOUR-POD-ID-8888.proxy.runpod.net`

### Health Check
```bash
curl https://YOUR-POD-ID-8888.proxy.runpod.net/health
```

### Generate 3D Mesh from Image (Git Bash on Windows)
```bash
# Create JSON file with base64 image
echo '{"image": "' > /tmp/request.json
base64 -w 0 /c/Users/tas13/Desktop/demo-food.jpg >> /tmp/request.json
echo '"}' >> /tmp/request.json

# Generate and download GLB file
curl -X POST https://YOUR-POD-ID-8888.proxy.runpod.net/generate-file \
  -H "Content-Type: application/json" \
  -d @/tmp/request.json \
  --output model.glb
```

### PowerShell Alternative
```powershell
$imageBytes = [System.IO.File]::ReadAllBytes("C:\Users\tas13\Desktop\demo-food.jpg")
$base64 = [Convert]::ToBase64String($imageBytes)
$body = @{image = $base64} | ConvertTo-Json
Invoke-WebRequest -Uri "https://YOUR-POD-ID-8888.proxy.runpod.net/generate-file" -Method POST -Body $body -ContentType "application/json" -OutFile "model.glb"
```

---

## Troubleshooting

### "No module named pytorch3d"
```bash
pip install -e '.[p3d]'
```

### "No module named torch"
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### "No matching distribution for kaolin"
```bash
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install kaolin==0.17.0
```

### "cuda_runtime.h not found" (nvdiffrast build error)
```bash
export CUDA_HOME=/usr/local/cuda
export CPATH=$CUDA_HOME/include:$CPATH
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

### "GaussianRasterizationSettings not defined"
```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7 --force-reinstall
```

### Conda TOS error
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### HuggingFace hf_transfer error
```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### Port 8888 already in use (Jupyter)
```bash
pkill -f jupyter
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check if model loaded |
| `/generate` | POST | Upload image → Get GLB file back |

### Request
```json
{
  "image": "base64_encoded_image",
  "seed": 42
}
```
Or use image URL:
```json
{
  "image_url": "https://example.com/food.jpg",
  "seed": 42
}
```

### Response
Returns GLB file directly (binary download)

### How it works:
1. Upload any photo (JPG, PNG)
2. API auto-segments to find the object (using rembg)
3. Converts to 3D mesh with textures
4. Returns GLB file

---

## Quick Reference
- GitHub: `https://github.com/tas1337/sam-3d-objects`
- Port: 8888
- GPU needed: 24GB+ VRAM (A100, L40S, A40)
- HuggingFace: `facebook/sam-3d-objects`
