#!/bin/bash
# =============================================================================
# CuteCaption Qwen Mini-Server - RunPod Deployment Script
# =============================================================================
#
# USAGE:
#   1. Start a RunPod pod with Ubuntu + 4090/5090
#   2. SSH or use Web Terminal into the pod
#   3. cd /workspace
#   4. Upload this folder (or git clone)
#   5. Run: bash setup.sh
#
# The script will:
#   - Install Python dependencies
#   - Download the Qwen3-VL-8B model (cached in /workspace/models)
#   - Start the web server with dashboard
#
# After setup, access http://<your-runpod-ip>:8080 for the dashboard
# =============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           CuteCaption Qwen Mini-Server - RunPod Setup                ║"
echo "║                                                                      ║"
echo "║  Offload Qwen VLM operations to your RunPod GPU                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
MODEL_CACHE="/workspace/models"
VENV_PATH="/workspace/venv"
PORT="${CUTECAPTION_PORT:-8080}"
API_KEY="${CUTECAPTION_API_KEY:-cutecaption-$(date +%s | sha256sum | head -c 16)}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Check GPU
echo ""
log_info "Step 1: Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log_success "GPU detected: $GPU_NAME ($GPU_MEM MB VRAM)"
else
    log_error "No GPU detected! nvidia-smi not found."
    exit 1
fi

# Step 2: Create model cache directory
echo ""
log_info "Step 2: Setting up model cache at $MODEL_CACHE..."
mkdir -p "$MODEL_CACHE"
export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
log_success "Model cache configured"

# Step 3: Set up Python environment
echo ""
log_info "Step 3: Setting up Python virtual environment..."

if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    log_success "Virtual environment created at $VENV_PATH"
else
    log_success "Virtual environment already exists"
fi

source "$VENV_PATH/bin/activate"
log_success "Virtual environment activated"

# Step 4: Install dependencies
echo ""
log_info "Step 4: Installing Python dependencies..."

pip install --upgrade pip -q

# Install PyTorch with CUDA (if not already installed)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log_info "Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
fi

# Install requirements
pip install -r requirements.txt -q

# Install transformers from git (required for Qwen3-VL)
pip install git+https://github.com/huggingface/transformers -q

log_success "Dependencies installed"

# Step 5: Pre-download model (optional but recommended)
echo ""
log_info "Step 5: Pre-downloading Qwen3-VL-8B model..."
log_warning "This may take 5-10 minutes on first run..."

python3 << 'PREFETCH_SCRIPT'
import os
os.environ['HF_HOME'] = '/workspace/models'

from huggingface_hub import snapshot_download

print("[SETUP] Downloading Qwen3-VL-8B-Instruct...")
snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    revision="main",
    max_workers=4
)
print("[SETUP] Model download complete!")
PREFETCH_SCRIPT

log_success "Model cached at $MODEL_CACHE"

# Step 6: Generate API key if not set
echo ""
log_info "Step 6: Configuring security..."

# Save API key for later reference
echo "$API_KEY" > /workspace/.cutecaption_api_key
chmod 600 /workspace/.cutecaption_api_key

log_success "API Key saved to /workspace/.cutecaption_api_key"
echo -e "  ${PURPLE}API Key: ${NC}${API_KEY}"

# Step 7: Start the server
echo ""
log_info "Step 7: Starting Qwen Mini-Server..."
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         SERVER STARTING                                      ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                              ║"
echo "║  Access Dashboard via RunPod Proxy:                                          ║"
echo -e "║    ${GREEN}https://<POD-ID>-$PORT.proxy.runpod.net${NC}                                  ║"
echo "║                                                                              ║"
echo -e "║  API Key: ${PURPLE}$API_KEY${NC}                              ║"
echo "║                                                                              ║"
echo "║  ┌────────────────────────────────────────────────────────────────────────┐  ║"
echo "║  │ INSTRUCTIONS:                                                          │  ║"
echo "║  │ 1. Open the dashboard URL in your browser                              │  ║"
echo "║  │ 2. Copy the Server URL and API Key from the dashboard                  │  ║"
echo "║  │ 3. In CuteCaption: Settings → Remote GPU → Paste URL and Key           │  ║"
echo "║  └────────────────────────────────────────────────────────────────────────┘  ║"
echo "║                                                                              ║"
echo "║  To restart later: bash start.sh                                             ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Export environment variables
export CUTECAPTION_API_KEY="$API_KEY"
export CUTECAPTION_PORT="$PORT"
export CUTECAPTION_MODEL_CACHE="$MODEL_CACHE"

# Run the server
cd "$(dirname "$0")"
python3 qwen_server.py

