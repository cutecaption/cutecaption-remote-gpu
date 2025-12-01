# CuteCaption Remote GPU - RunPod Template Configuration

## Quick Deploy to RunPod

### Option 1: Using Docker Hub (Recommended)

1. **Build and Push Image** (one-time setup):
   ```bash
   # From working_8_30 directory
   docker build -t your-dockerhub-username/cutecaption-remote-gpu:latest -f remote-gpu/Dockerfile .
   docker push your-dockerhub-username/cutecaption-remote-gpu:latest
   ```

2. **Create RunPod Template**:
   - Go to https://www.runpod.io/console/user/templates
   - Click "New Template"
   - Fill in:
     - **Name**: `CuteCaption Remote GPU`
     - **Container Image**: `your-dockerhub-username/cutecaption-remote-gpu:latest`
     - **Container Disk**: `20 GB` (minimum)
     - **Volume Disk**: `50 GB` (recommended for model cache)
     - **Volume Mount Path**: `/workspace/models`
     - **Expose HTTP Ports**: `8080`
     - **Expose TCP Ports**: Leave empty
     - **Environment Variables**:
       ```
       CUTECAPTION_API_KEY=your-secure-api-key-here
       CUTECAPTION_MODEL_CACHE=/workspace/models
       ```

3. **Deploy Pod**:
   - Go to https://www.runpod.io/console/pods
   - Click "Deploy" on your template
   - Select GPU: **RTX 4090** or **RTX 5090** recommended
   - Start the Pod

4. **Access Dashboard**:
   - Once Pod is running, click "Connect"
   - Open the HTTP port (8080) in browser
   - You'll see the CuteCaption Remote GPU dashboard

### Option 2: Direct GitHub Deployment

If you host your code on GitHub, you can use RunPod's GitHub integration:

1. **Push to GitHub**:
   ```bash
   git add remote-gpu python/intelligence
   git commit -m "Add remote GPU server"
   git push
   ```

2. **Create Dockerfile in repo root** (RunPod will auto-detect)

3. **Use GitHub URL** in RunPod template:
   - Container Image: `github.com/your-username/your-repo`
   - RunPod will build automatically

### Option 3: Pre-Built RunPod Community Template

If you publish this, users can directly deploy from RunPod Community Templates:
- Search for "CuteCaption Remote GPU"
- Click "Deploy"
- Done!

---

## Recommended RunPod GPU Configuration

### For VLM (Qwen3-VL) Heavy Workloads:
- **GPU**: RTX 4090 (24GB VRAM) or RTX 5090 (32GB VRAM)
- **RAM**: 64GB minimum
- **Storage**: 50GB+ (for model cache)
- **Region**: Any with high availability

### For CLIP-Only Workloads:
- **GPU**: RTX 3090 (24GB) or RTX 4080 (16GB)
- **RAM**: 32GB minimum
- **Storage**: 30GB

### Cost Estimates (approximate):
- **RTX 4090**: ~$0.60-0.80/hour (Secure Cloud)
- **RTX 5090**: ~$1.20-1.50/hour (Secure Cloud)
- **Community Cloud**: 30-50% cheaper (but less reliable)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUTECAPTION_API_KEY` | `dev-key` | **REQUIRED** API key for auth |
| `CUTECAPTION_HOST` | `0.0.0.0` | Server bind address |
| `CUTECAPTION_PORT` | `8080` | Server port |
| `CUTECAPTION_MODEL_CACHE` | `/workspace/models` | HuggingFace model cache |

---

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| `8080` | HTTP | Dashboard & REST API |
| `8080` | WebSocket | Real-time updates |

---

## Volume Mounts (Optional but Recommended)

| Mount Path | Purpose | Recommended Size |
|------------|---------|------------------|
| `/workspace/models` | Model cache (persistent) | 50GB |

**Why persistent volume?**
- Models download once and persist across Pod restarts
- Dramatically faster startup (no re-download)
- Saves bandwidth costs

---

## Security Best Practices

1. **Change API Key**: Always set a strong `CUTECAPTION_API_KEY`
2. **Use Secure Cloud**: For production, use RunPod Secure Cloud (not Community Cloud)
3. **Enable Firewall**: In RunPod dashboard, restrict IP access if possible
4. **Monitor Logs**: Check dashboard activity logs regularly
5. **Auto-Stop**: Enable auto-stop when idle to save costs

---

## Troubleshooting

### Pod won't start
- Check logs in RunPod console
- Verify Docker image is public (or credentials are set)
- Ensure GPU has enough VRAM (24GB+ for VLM)

### Dashboard shows "No GPU Detected"
- Ensure Pod has GPU allocated
- Check CUDA drivers: `nvidia-smi` in terminal

### Models downloading slowly
- Use persistent volume for `/workspace/models`
- Consider pre-baking models into Docker image

### API requests failing
- Verify API key matches in CuteCaption settings
- Check firewall/network settings
- Test with: `curl http://your-pod-url:8080/status`

---

## Performance Tuning

### Faster Model Loading
Pre-download models during image build (uncomment in Dockerfile):
```dockerfile
RUN python3 -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', \
    cache_dir='/workspace/models')"
```

### Batch Requests
The server automatically handles concurrent requests efficiently.

### Lower Latency
- Choose geographically close RunPod datacenter
- Use Secure Cloud (faster networking than Community)
- Enable WebSocket in CuteCaption for real-time updates

---

## Monitoring & Maintenance

### Check Server Health
```bash
curl http://your-pod-url:8080/status
```

### View Logs (RunPod Console)
- Go to Pod → Logs tab
- Monitor for errors or warnings

### GPU Utilization
Dashboard shows real-time GPU metrics:
- VRAM usage
- GPU utilization %
- Temperature

### Auto-Restart on Crash
RunPod will automatically restart your Pod if it crashes.

---

## Cost Optimization

1. **Auto-Stop Idle Pods**: Enable in RunPod settings
2. **Use Community Cloud**: 30-50% cheaper (trade-off: less reliable)
3. **Persistent Volume**: Avoid re-downloading models (saves time & bandwidth)
4. **Right-Size GPU**: Don't over-provision VRAM
5. **Stop When Not Testing**: Only run during active development

---

## Support & Updates

- **Issues**: Report at https://github.com/cutecaption/issues
- **Discord**: [Join CuteCaption Discord](https://discord.gg/2gQ5eBtRCj)
- **Documentation**: Full docs at https://github.com/cutecaption

---

## Example API Usage (Testing)

Test VLM endpoint:
```bash
curl -X POST "http://your-pod-url:8080/api/v1/vlm/caption" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "max_resolution": 1024
  }'
```

---

## Changelog

### v1.0.0 (Initial Release)
- FastAPI server with lazy model loading
- Beautiful status dashboard
- VLM, Semantic, Pose service endpoints
- WebSocket support for real-time updates
- RunPod-optimized deployment

