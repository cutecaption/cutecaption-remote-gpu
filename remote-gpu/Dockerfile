# CuteCaption Remote GPU Server - RunPod Image
# Optimized for NVIDIA GPUs (40 series, 50 series recommended)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUTECAPTION_MODEL_CACHE=/workspace/models \
    CUTECAPTION_HOST=0.0.0.0 \
    CUTECAPTION_PORT=8080 \
    CUTECAPTION_API_KEY=change-this-in-runpod-template

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy Python intelligence engines
COPY python/intelligence /workspace/python/intelligence

# Copy remote GPU server
COPY remote-gpu/server /workspace/server

# Copy dashboard
COPY remote-gpu/dashboard /workspace/dashboard

# Install Python dependencies
# First, install PyTorch with CUDA support
# Using newer version compatible with python 3.10 and recent transformers
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
COPY remote-gpu/server/requirements.txt /workspace/requirements.txt
COPY remote-gpu/requirements_docker.txt /workspace/requirements_base.txt

RUN pip3 install --no-cache-dir -r /workspace/requirements.txt && \
    pip3 install --no-cache-dir -r /workspace/requirements_base.txt

# Create model cache directory
RUN mkdir -p /workspace/models && chmod 777 /workspace/models

# Download models on build (optional - comment out for faster builds)
# Models will download on first use if not pre-cached
# RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')"
# RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('zer0int/LongCLIP-SAE-ViT-L-14')"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/status || exit 1

# Start server
CMD ["python3", "/workspace/server/main.py"]

