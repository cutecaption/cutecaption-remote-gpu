#!/bin/bash
# Simple setup script for RunPod - run this after copying files

cd /workspace

# Install Python dependencies
pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install server dependencies
pip3 install --no-cache-dir -r remote-gpu/server/requirements.txt

# Install AI dependencies
pip3 install --no-cache-dir \
    git+https://github.com/huggingface/transformers \
    accelerate \
    bitsandbytes \
    safetensors \
    faiss-cpu \
    git+https://github.com/openai/CLIP.git \
    qwen-vl-utils \
    sentencepiece \
    einops \
    timm \
    Pillow \
    tqdm \
    pyyaml \
    requests \
    huggingface-hub \
    psutil

# Create models directory
mkdir -p /workspace/models
chmod 777 /workspace/models

# Start server
cd /workspace
python3 remote-gpu/server/main.py

