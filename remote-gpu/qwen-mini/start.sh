#!/bin/bash
# =============================================================================
# CuteCaption Qwen Mini-Server - Quick Start
# =============================================================================
# Use this to restart the server after initial setup
# For first-time setup, use: bash setup.sh
# =============================================================================

VENV_PATH="/workspace/venv"
MODEL_CACHE="/workspace/models"
PORT="${CUTECAPTION_PORT:-8080}"

# Check if setup was run
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found. Run 'bash setup.sh' first."
    exit 1
fi

# Load saved API key
if [ -f "/workspace/.cutecaption_api_key" ]; then
    API_KEY=$(cat /workspace/.cutecaption_api_key)
else
    API_KEY="dev-key-change-in-production"
    echo "⚠️  No saved API key found. Using default."
fi

# Activate environment
source "$VENV_PATH/bin/activate"

# Export environment
export CUTECAPTION_API_KEY="$API_KEY"
export CUTECAPTION_PORT="$PORT"
export CUTECAPTION_MODEL_CACHE="$MODEL_CACHE"
export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           CuteCaption Qwen Mini-Server - Starting                    ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Port: $PORT                                                           ║"
echo "║  API Key: ${API_KEY:0:8}...                                            ║"
echo "║                                                                      ║"
echo "║  Access via RunPod:                                                  ║"
echo "║  https://<POD-ID>-$PORT.proxy.runpod.net                               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Start server
cd "$(dirname "$0")"
python3 qwen_server.py

