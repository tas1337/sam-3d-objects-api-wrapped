FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIDRA_SKIP_INIT=true
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    wget curl git build-essential ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace/sam-3d-objects

# Copy ONLY environment files first (for caching)
COPY environments/ environments/
COPY pyproject.toml requirements*.txt ./
COPY patching/ patching/
COPY setup.py* ./

# Create conda env (cached if environment files unchanged)
RUN conda config --set remote_read_timeout_secs 600 && \
    conda config --set remote_connect_timeout_secs 120 && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environments/default.yml && \
    conda clean -ya

ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Copy source code for pip install
COPY sam3d_objects/ sam3d_objects/
COPY notebook/ notebook/

# Install dependencies (cached if requirements unchanged)
RUN /opt/conda/envs/sam3d-objects/bin/pip install --no-cache-dir 'huggingface-hub[cli]<1.0' && \
    /opt/conda/envs/sam3d-objects/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[dev]' && \
    /opt/conda/envs/sam3d-objects/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[p3d]'

RUN /opt/conda/envs/sam3d-objects/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[inference]'

RUN if [ -f patching/hydra ]; then /opt/conda/envs/sam3d-objects/bin/python patching/hydra; fi

# Install API dependencies + rembg for auto-segmentation
RUN /opt/conda/envs/sam3d-objects/bin/pip install --no-cache-dir flask flask-cors requests gunicorn rembg

# Copy remaining files (api_server.py, checkpoints, etc.) - changes here don't affect above layers
COPY api_server.py demo.py ./
COPY checkpoints/ checkpoints/
COPY scripts/ scripts/
COPY client/ client/
COPY doc/ doc/

RUN echo '#!/bin/bash\n\
set -e\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate sam3d-objects\n\
export CUDA_HOME=/usr/local/cuda\n\
export LIDRA_SKIP_INIT=true\n\
cd /workspace/sam-3d-objects\n\
if [ ! -f "checkpoints/hf/pipeline.yaml" ] && [ ! -z "$HF_TOKEN" ]; then\n\
    huggingface-cli login --token $HF_TOKEN\n\
    huggingface-cli download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects\n\
    if [ -d "checkpoints/hf-download/checkpoints" ]; then mv checkpoints/hf-download/checkpoints checkpoints/hf; fi\n\
    rm -rf checkpoints/hf-download\n\
fi\n\
gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 2 --timeout 300 api_server:app\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

EXPOSE 8000
CMD ["/workspace/start.sh"]
