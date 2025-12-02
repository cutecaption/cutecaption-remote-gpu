# CuteCaption Qwen Mini-Server for RunPod

Offload Qwen VLM operations to a RunPod GPU with one script.

---

## Quick Start (5 Minutes)

### Step 1: Create RunPod Pod

1. Go to [RunPod.io](https://runpod.io) and log in
2. Click **Deploy** → **GPU Pod**
3. Select a GPU with **16GB+ VRAM** (RTX 4090, A100, etc.)
4. Choose **RunPod Pytorch 2.1** template (or any Ubuntu template)
5. Set **Container Disk** to at least **30GB** (for model cache)
6. Set **Volume Disk** to **50GB** (persistent storage)
7. **IMPORTANT**: Under **Expose HTTP Ports**, add port `8080`
8. Click **Deploy**

### Step 2: Upload Files

Once the pod is running:

1. Click **Connect** → **Web Terminal** (or use SSH)
2. Navigate to workspace:
   ```bash
   cd /workspace
   ```
3. Upload files using the **File Manager** (click folder icon in RunPod):
   - Upload all files from this `qwen-mini` folder to `/workspace/qwen-mini/`
   
   OR clone/download directly:
   ```bash
   # If you have the files somewhere accessible:
   mkdir -p /workspace/qwen-mini
   cd /workspace/qwen-mini
   # Upload: setup.sh, qwen_server.py, dashboard.html, requirements.txt
   ```

### Step 3: Run Setup

```bash
cd /workspace/qwen-mini
bash setup.sh
```

This will:
- ✅ Check GPU availability
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Download Qwen3-VL-8B model (~16GB, takes 5-10 min)
- ✅ Generate an API key
- ✅ Start the server

### Step 4: Access Dashboard

Your dashboard URL is:
```
https://<POD-ID>-8080.proxy.runpod.net
```

Find your Pod ID in the RunPod dashboard (it's in the pod's URL).

Example: If your pod shows `abc123xyz` as the ID:
```
https://abc123xyz-8080.proxy.runpod.net
```

### Step 5: Connect from CuteCaption

1. Open **CuteCaption**
2. Go to **AI Intelligence Workstation** → **Settings** (gear icon)
3. Scroll to **Remote GPU (RunPod)** section
4. Enable **Use Remote GPU**
5. Enter **Server URL**: `https://<POD-ID>-8080.proxy.runpod.net`
6. Enter **API Key**: Copy from the dashboard's "Copy to CuteCaption Settings" section
7. Click **Test Connection**

---

## Files Included

| File | Purpose |
|------|---------|
| `setup.sh` | One-click deployment script |
| `qwen_server.py` | FastAPI server with Qwen VLM |
| `dashboard.html` | Web dashboard for monitoring |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/status` | GET | Server status, GPU info, connection details |
| `/api/v1/load` | POST | Load Qwen model into VRAM |
| `/api/v1/unload` | POST | Unload model from VRAM |
| `/api/v1/classify` | POST | Classify image (25 dimensions) |
| `/api/v1/caption` | POST | Generate image caption |
| `/api/v1/vqa` | POST | Visual question answering |
| `/api/v1/verify` | POST | Verify candidate images |
| `/api/v1/analyze_question` | POST | Analyze question intent |
| `/api/v1/synthesize` | POST | Synthesize answer from evidence |
| `/ws` | WebSocket | Real-time updates |

---

## Configuration

Environment variables (optional):

```bash
export CUTECAPTION_API_KEY="your-custom-key"  # Default: auto-generated
export CUTECAPTION_PORT="8080"                 # Default: 8080
export CUTECAPTION_MODEL_CACHE="/workspace/models"  # Model storage
```

---

## Troubleshooting

### "Connection failed" in CuteCaption
- Make sure port 8080 is exposed in RunPod settings
- Use `https://` (not `http://`) for the RunPod proxy URL
- Check that the API key matches exactly

### Model loading is slow
- First load downloads ~16GB model
- Subsequent starts use cached model in `/workspace/models`
- If pod restarts, model is still cached (if using volume storage)

### Out of VRAM
- Qwen3-VL-8B needs ~16GB VRAM
- Use a GPU with at least 16GB (RTX 4090, A100, etc.)
- Close other GPU processes before loading

### Server won't start
```bash
# Check logs
cd /workspace/qwen-mini
source /workspace/venv/bin/activate
python qwen_server.py
```

---

## Restarting the Server

If you stop the server and need to restart:

```bash
cd /workspace/qwen-mini
source /workspace/venv/bin/activate
export CUTECAPTION_API_KEY=$(cat /workspace/.cutecaption_api_key)
python qwen_server.py
```

Or just run `bash setup.sh` again (it will skip already-installed parts).

---

## Cost Optimization

- **Stop the pod** when not in use (models stay cached in volume)
- Use **Spot instances** for ~70% savings (may be interrupted)
- Smaller models coming soon for lower VRAM requirements

---

## Support

Issues? Check the CuteCaption GitHub or Discord.
