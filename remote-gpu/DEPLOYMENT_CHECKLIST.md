# CuteCaption RunPod Deployment Checklist

## 🎯 Goal
Get CuteCaption GPU server running on RunPod in under 10 minutes.

---

## ✅ Pre-Deployment Checklist

### 1. Prerequisites
- [ ] Docker installed and running
- [ ] Docker Hub account (free): https://hub.docker.com/signup
- [ ] RunPod account (free): https://www.runpod.io/console/signup
- [ ] RunPod credits added ($10 minimum for testing)

### 2. Project Files Ready
- [ ] `remote-gpu/Dockerfile` exists
- [ ] `remote-gpu/server/main.py` exists
- [ ] `remote-gpu/server/requirements.txt` exists
- [ ] `python/intelligence/` directory with AI engines

---

## 📦 Deployment Steps (Docker Hub Method)

### Step 1: Build Docker Image (5 min)

```bash
# Navigate to project
cd D:\CE_Working_9_3\CE_Rebuild_9_2\CE_pre_Qwen-Ready\working_8_30

# Login to Docker Hub
docker login
# Username: your-dockerhub-username
# Password: (enter your password)

# Build image for Linux AMD64 (RunPod uses this architecture)
docker build --platform linux/amd64 ^
  -t your-dockerhub-username/cutecaption-gpu:v1.0 ^
  -f remote-gpu/Dockerfile .

# This takes 5-10 minutes depending on internet speed
```

**Expected output:**
```
[+] Building 300.5s (15/15) FINISHED
 => exporting to image
 => => naming to docker.io/your-dockerhub-username/cutecaption-gpu:v1.0
```

- [ ] Build completed successfully

### Step 2: Push to Docker Hub (5 min)

```bash
# Push image to Docker Hub (makes it accessible to RunPod)
docker push your-dockerhub-username/cutecaption-gpu:v1.0

# This takes 5-10 minutes depending on internet speed
```

**Expected output:**
```
v1.0: digest: sha256:abc123... size: 2547
```

- [ ] Push completed successfully
- [ ] Verify on Docker Hub: https://hub.docker.com/r/your-dockerhub-username/cutecaption-gpu

### Step 3: Test Locally (Optional but Recommended)

```bash
# Test before deploying to RunPod
docker run -p 8080:8080 ^
  -e CUTECAPTION_API_KEY=test123 ^
  your-dockerhub-username/cutecaption-gpu:v1.0

# Open browser: http://localhost:8080
# Should see CuteCaption GPU Server dashboard
```

- [ ] Dashboard loads successfully
- [ ] GPU info shows (or "No GPU" if testing on CPU)
- [ ] Stop container: `Ctrl+C`

---

## 🚀 RunPod Template Creation (5 min)

### Step 4: Create Template

1. **Go to:** https://www.runpod.io/console/user/templates

2. **Click:** "New Template" (blue button, top-right)

3. **Fill Basic Info:**
   - [ ] **Template Name:** `CuteCaption GPU Server`
   - [ ] **Image:** `your-dockerhub-username/cutecaption-gpu:v1.0`
   - [ ] **Docker Command:** *(leave blank)*

4. **Configure Container:**
   - [ ] **Container Disk:** `20` GB
   - [ ] **Expose HTTP Ports:** `8080`
   - [ ] **Expose TCP Ports:** *(leave blank)*

5. **Configure Volume (Important!):**
   - [ ] **Volume Disk:** `50` GB
   - [ ] **Volume Mount Path:** `/workspace/models`
   - [ ] **Note:** This caches models so they don't re-download

6. **Add Environment Variables:**

   Click "+ Add Environment Variable" for each:

   **Variable 1:**
   - [ ] **Key:** `CUTECAPTION_API_KEY`
   - [ ] **Default Value:** *(leave empty - users enter on deploy)*
   - [ ] **Description:** `API key for authentication (generate: openssl rand -base64 32)`
   - [ ] **☑️ User Configurable:** ✅ Checked

   **Variable 2:**
   - [ ] **Key:** `CUTECAPTION_MODEL_CACHE`
   - [ ] **Default Value:** `/workspace/models`
   - [ ] **☐ User Configurable:** Unchecked

   **Variable 3:**
   - [ ] **Key:** `CUTECAPTION_PORT`
   - [ ] **Default Value:** `8080`
   - [ ] **☐ User Configurable:** Unchecked

   **Variable 4:**
   - [ ] **Key:** `CUTECAPTION_HOST`
   - [ ] **Default Value:** `0.0.0.0`
   - [ ] **☐ User Configurable:** Unchecked

7. **Save Template:**
   - [ ] Click "Save Template"
   - [ ] Template appears in list

---

## 🧪 Test Deployment (5 min)

### Step 5: Deploy Test Pod

1. **Go to:** https://www.runpod.io/console/pods

2. **Click:** "Deploy" button on your template

3. **Select GPU:**
   - [ ] **Recommended:** RTX 4090 (24GB, ~$0.79/hr)
   - [ ] **Budget:** RTX 3090 (24GB, ~$0.50/hr)
   - [ ] **High-end:** RTX 5090 (32GB, ~$1.49/hr)

4. **Cloud Type:**
   - [ ] **Secure Cloud** (recommended, stable)
   - [ ] *or* **Community Cloud** (cheaper, less reliable)

5. **Configure:**
   - [ ] **Pod Name:** `cutecaption-test-1`
   - [ ] **API Key:** Enter a test key (e.g., `test-key-abc123`)
   - [ ] **Volume:** ✅ Enable (50GB)

6. **Deploy:**
   - [ ] Click "Deploy On-Demand"
   - [ ] Wait 30-60 seconds for Pod to start
   - [ ] Status changes to "Running"

### Step 6: Verify Deployment

1. **Connect to Service:**
   - [ ] Click "Connect" button on Pod
   - [ ] See "HTTP Service [8080]" in list
   - [ ] Click "Connect to HTTP Service [8080]"

2. **Dashboard Verification:**
   - [ ] Browser opens to `https://xxx-8080.proxy.runpod.net`
   - [ ] Dashboard loads successfully
   - [ ] GPU detected: "NVIDIA GeForce RTX 4090" (or selected GPU)
   - [ ] VRAM shows: "24.0 GB" (or GPU's VRAM)
   - [ ] Services show: "Ready (Lazy-loaded)"

3. **API Test:**
   - [ ] Copy Pod URL from browser
   - [ ] Test status endpoint:
     ```bash
     curl https://xxx-8080.proxy.runpod.net/status
     ```
   - [ ] Returns JSON with GPU info

4. **Note Connection Info:**
   - [ ] **Server URL:** `https://_______________-8080.proxy.runpod.net`
   - [ ] **API Key:** `_________________________________`

---

## 🖥️ Connect CuteCaption App (2 min)

### Step 7: Configure App

1. **Open CuteCaption:**
   - [ ] Launch app on Windows/Mac

2. **Load Dataset:**
   - [ ] Click "Open Folder"
   - [ ] Select any image folder

3. **Open Settings:**
   - [ ] AI Intelligence panel appears (bottom)
   - [ ] Click "⚙️" (Settings) button (top-right)

4. **Configure Remote GPU:**
   - [ ] Scroll to "Remote GPU (RunPod)"
   - [ ] Toggle "Use Remote GPU" → **ON**
   - [ ] Enter **Server URL:** (paste from Step 6.4)
   - [ ] Enter **API Key:** (paste from Step 6.4)
   - [ ] Click "🔌 Test Connection"

5. **Verify Connection:**
   - [ ] Success message: "✅ Connected! GPU: RTX 4090, 24GB VRAM"
   - [ ] Click "Done" to save

### Step 8: Test Inference

1. **Test VLM Classification:**
   - [ ] Click "Analyze Dataset" in AI Intelligence panel
   - [ ] Wait for analysis (30-60s first time, 2-5s after)
   - [ ] Results appear (runs on RunPod GPU)

2. **Test VLM Question:**
   - [ ] Ask: "How many images show people?"
   - [ ] Wait for answer (VLM synthesis on RunPod)
   - [ ] Answer appears with visualizations

3. **Monitor Dashboard:**
   - [ ] Open RunPod dashboard in browser (Step 6.2 URL)
   - [ ] See "Activity Log" showing requests
   - [ ] See GPU usage spike during inference

---

## 🛑 Stop Pod When Done

### Step 9: Pause Billing

**Important:** RunPod charges by the hour. Stop Pod when not in use.

- [ ] Go to: https://www.runpod.io/console/pods
- [ ] Click "Stop" on Pod
- [ ] Billing pauses (resume anytime with "Start")

**Or enable auto-stop:**
- [ ] Pod → Settings → Auto-Stop
- [ ] Set: "5 minutes idle"
- [ ] Pod auto-stops when no requests

---

## 🎉 Success Criteria

You've successfully deployed if:

- ✅ Docker image pushed to Docker Hub
- ✅ RunPod template created
- ✅ Pod deploys and runs without errors
- ✅ Dashboard accessible at `https://xxx-8080.proxy.runpod.net`
- ✅ GPU detected in dashboard
- ✅ CuteCaption connects successfully
- ✅ Test inference works (VLM classification/questions)
- ✅ Activity logs show in dashboard

---

## 📊 Cost Tracking

### Estimated Costs

| Activity | Duration | GPU | Cost |
|----------|----------|-----|------|
| **Initial test** | 30 min | RTX 4090 | $0.40 |
| **1 hour testing** | 1 hr | RTX 4090 | $0.79 |
| **Dataset analysis** (1000 images) | ~2 hrs | RTX 4090 | $1.58 |
| **Daily usage** (8 hrs, auto-stop) | ~2 hrs actual | RTX 4090 | $1.58 |

**Tips to minimize costs:**
- ✅ Enable auto-stop after 5 min idle
- ✅ Use Community Cloud (30-50% cheaper)
- ✅ Stop Pod manually when done for the day
- ✅ Share one Pod across team (not per-user)

---

## 🐛 Troubleshooting

### Issue: Build Fails

**Error:** `ERROR: failed to solve: process "/bin/sh -c pip3 install..." did not complete successfully`

**Solutions:**
- ✅ Check internet connection
- ✅ Verify `requirements.txt` exists
- ✅ Try: `docker build --no-cache ...`

### Issue: Push Fails

**Error:** `denied: requested access to the resource is denied`

**Solutions:**
- ✅ Run `docker login` again
- ✅ Verify username in image tag matches Docker Hub username
- ✅ Check repository is public (or enable private access in RunPod)

### Issue: Pod Fails to Start

**Error:** Pod status "Error" or "Exited"

**Solutions:**
- ✅ Check logs: Pod → Logs tab
- ✅ Verify image name is correct
- ✅ Ensure image is public on Docker Hub
- ✅ Test locally first: `docker run -p 8080:8080 ...`

### Issue: Dashboard Not Loading

**Error:** "Connect" button missing or 404

**Solutions:**
- ✅ Check logs for startup errors
- ✅ Verify port 8080 is exposed in template
- ✅ Wait 60s for server to fully start
- ✅ Try: `curl https://xxx-8080.proxy.runpod.net/status`

### Issue: CuteCaption Connection Failed

**Error:** "❌ Connection failed"

**Solutions:**
- ✅ Verify Pod is "Running" (not Stopped)
- ✅ Check URL format: `https://xxx-8080.proxy.runpod.net` (no trailing `/`)
- ✅ Verify API key matches (case-sensitive, no spaces)
- ✅ Test manually: `curl -H "X-API-Key: YOUR_KEY" https://xxx-8080.proxy.runpod.net/status`

---

## 📝 Notes

### Image Tags
- Use version tags for production: `cutecaption-gpu:v1.0`
- Avoid `:latest` for stability (harder to rollback)

### Registry Options
- **Docker Hub:** Free for public images, easiest
- **GitHub Container Registry:** Free for private, use `ghcr.io/user/repo:tag`
- **RunPod Registry:** Coming soon

### Model Pre-caching
- Default: Models download on first request (5-10 min)
- With volume: Models persist across restarts (30s startup)
- Pre-baked: Models in image (instant startup, but 30GB+ image)

**To pre-bake models, uncomment in Dockerfile:**
```dockerfile
RUN python3 -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', \
    cache_dir='/workspace/models')"
```

---

## 🔄 Updating Deployment

### When code changes:

1. **Rebuild image:**
   ```bash
   docker build --platform linux/amd64 \
     -t your-dockerhub-username/cutecaption-gpu:v1.1 \
     -f remote-gpu/Dockerfile .
   ```

2. **Push new version:**
   ```bash
   docker push your-dockerhub-username/cutecaption-gpu:v1.1
   ```

3. **Update RunPod template:**
   - Go to: Templates → Edit → Change image to `v1.1`

4. **Restart Pod:**
   - Stop existing Pod
   - Deploy new Pod (pulls v1.1)

---

## ✅ Final Checklist

Before considering deployment complete:

- [ ] Docker image builds successfully
- [ ] Image pushed to Docker Hub (public)
- [ ] RunPod template created
- [ ] Test Pod deploys successfully
- [ ] Dashboard accessible and shows GPU
- [ ] CuteCaption connects and tests connection
- [ ] Test inference works (VLM classification)
- [ ] Auto-stop enabled (cost management)
- [ ] Documentation shared with users
- [ ] Template URL shared (or published to Community)

---

**🎉 Congratulations! Your CuteCaption GPU server is live on RunPod!**

**Next steps:**
- Share template URL with users
- Monitor usage and costs
- Update image as needed
- Consider publishing to RunPod Community Templates

**Support:**
- Technical: `remote-gpu/README.md`
- User guide: `remote-gpu/USER_GUIDE.md`
- Full guide: `RUNPOD_QUICKSTART_GUIDE.md`



