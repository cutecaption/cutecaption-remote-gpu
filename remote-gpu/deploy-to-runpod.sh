#!/bin/bash

# CuteCaption Remote GPU - Quick Deploy Script
# Usage: ./deploy-to-runpod.sh [docker-username]

set -e

DOCKER_USERNAME=${1:-"your-dockerhub-username"}
IMAGE_NAME="cutecaption-remote-gpu"
VERSION="latest"

echo "======================================"
echo "CuteCaption Remote GPU - Quick Deploy"
echo "======================================"
echo ""

# Step 1: Build Docker image
echo "[1/3] Building Docker image..."
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} -f remote-gpu/Dockerfile .

# Step 2: Push to Docker Hub
echo ""
echo "[2/3] Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}

# Step 3: Instructions
echo ""
echo "[3/3] Deploy to RunPod"
echo "======================================"
echo ""
echo "✅ Docker image pushed successfully!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Go to: https://www.runpod.io/console/user/templates"
echo "2. Click 'New Template'"
echo "3. Fill in:"
echo "   - Name: CuteCaption Remote GPU"
echo "   - Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo "   - Container Disk: 20 GB"
echo "   - Volume Disk: 50 GB → /workspace/models"
echo "   - Expose Ports: 8080"
echo "   - Environment Variables:"
echo "     CUTECAPTION_API_KEY=<generate-secure-key>"
echo ""
echo "4. Deploy Pod with RTX 4090 or RTX 5090"
echo "5. Copy Pod URL (e.g., https://abc123-8080.proxy.runpod.net)"
echo "6. In CuteCaption:"
echo "   Settings → AI Intelligence → Remote GPU"
echo "   - Enable: Yes"
echo "   - Server URL: <paste-pod-url>"
echo "   - API Key: <same-as-env-var>"
echo "   - Test Connection → ✅"
echo ""
echo "🎉 Ready to test!"
echo ""
echo "======================================"

