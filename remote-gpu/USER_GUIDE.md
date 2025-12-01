# Remote GPU Settings - User Guide

## 📍 Where to Find Remote GPU Settings

### Step 1: Open AI Dataset Intelligence Workspace
- Load a dataset in CuteCaption
- The AI Dataset Intelligence Workspace appears at the bottom
- Look for the **⚙️ Settings** button in the top-right

### Step 2: Click Settings ⚙️
- Settings panel opens as an overlay
- Scrollable list of setting groups

### Step 3: Scroll to "Remote GPU (RunPod)"
The settings are organized in groups:
```
┌─────────────────────────────────────────────┐
│ ⚙️ AI Intelligence Settings                 │
├─────────────────────────────────────────────┤
│                                              │
│ Expert Services                              │
│ ├─ Semantic Search Service        [✓]       │
│ ├─ VLM Service (Qwen3-VL)         [✓]       │
│ └─ ...                                       │
│                                              │
│ ★ Remote GPU (RunPod) ★                     │
│ ├─ Use Remote GPU (RunPod)       [ ]        │
│ ├─ Server URL              [____________]   │
│ ├─ API Key                 [************]   │
│ └─ [🔌 Test Connection]                     │
│                                              │
│ Quality & Curation                           │
│ └─ ...                                       │
│                                              │
└─────────────────────────────────────────────┘
```

---

## ⚙️ Configuration Options

### 1. **Use Remote GPU (RunPod)**
- **Type**: Toggle (checkbox)
- **Default**: OFF
- **Description**: "Offload GPU tasks to remote server (VLM, Semantic, Pose)"
- **When ON**: All GPU inference runs on remote server
- **When OFF**: All GPU inference runs locally

### 2. **Remote GPU Server URL**
- **Type**: Text input
- **Default**: Empty
- **Placeholder**: `http://your-runpod-url:8080`
- **Example**: `https://abc123-8080.proxy.runpod.net`
- **Description**: "e.g., http://your-runpod-url:8080"

### 3. **API Key**
- **Type**: Password input (masked)
- **Default**: Empty
- **Placeholder**: `Enter API key`
- **Description**: "API key from RunPod environment variable"
- **Note**: Matches `CUTECAPTION_API_KEY` in RunPod

### 4. **Test Connection Button** 🔌
- **Validates**: URL and API key before saving
- **Shows**: GPU info on success
- **Example Success**:
  ```
  ✅ Connection successful!
  GPU: NVIDIA GeForce RTX 4090
  VRAM: 24.0 GB
  ```
- **Example Error**:
  ```
  ❌ Connection failed: timeout
  ```

---

## 🎯 Quick Setup Guide

### For Local Testing (No RunPod):
1. ☑️ Enable: **OFF** (use local GPU)
2. Leave URL and API Key empty
3. Continue using CuteCaption normally

### For Remote GPU (RunPod):

**Step 1: Deploy RunPod Pod**
```bash
# See: remote-gpu/RUNPOD_DEPLOYMENT.md
docker build -t your-username/cutecaption-remote-gpu:latest -f remote-gpu/Dockerfile .
docker push your-username/cutecaption-remote-gpu:latest

# Create RunPod template and deploy
# Copy Pod URL (e.g., https://abc123-8080.proxy.runpod.net)
```

**Step 2: Configure CuteCaption**
1. Open Settings in AI Dataset Intelligence Workspace
2. Scroll to **"Remote GPU (RunPod)"**
3. ☑️ **Enable**: ON
4. **Server URL**: Paste Pod URL
5. **API Key**: Enter your secure key
6. Click **🔌 Test Connection**
7. Wait for ✅ success message
8. Click **Done** to save

**Step 3: Verify**
- Load a dataset
- Click **Analyze Dataset**
- VLM classification runs on remote GPU
- Ask a question
- VLM synthesis runs on remote GPU
- Monitor in RunPod dashboard (open Server URL in browser)

---

## 💡 Tips

### Testing Connection
- **Always test** before saving
- Ensures URL is reachable
- Validates API key
- Shows GPU info (confirms GPU is available)

### Troubleshooting
- **"⚠️ Please enter Server URL and API Key first"**
  → Fill in both fields before testing
  
- **"❌ Connection failed: timeout"**
  → Check RunPod Pod is running
  → Verify URL includes `http://` or `https://`
  → Ensure port 8080 is exposed
  
- **"❌ Invalid API key"**
  → Check API key matches RunPod env var `CUTECAPTION_API_KEY`
  → No extra spaces or newlines

### Performance
- First request takes 30-60s (model loading on remote GPU)
- Subsequent requests: 1-5s (pure inference + network latency)
- Enable remote GPU when:
  - Local RAM < 32GB
  - Testing VLM features extensively
  - Want faster inference (RTX 4090/5090)

### Cost Management
- RunPod charges by the hour (~$0.79-1.49/hr)
- Enable auto-stop after 5 min idle
- Only run Pod when actively testing
- Stop Pod when done for the day

---

## 🔒 Security

### API Key Safety
- **Never share** your API key
- Generate strong keys: `openssl rand -base64 32`
- Rotate keys regularly
- Use RunPod Secrets (not plain env vars) in production

### HTTPS
- RunPod provides HTTPS proxies automatically
- Always use `https://` URLs (not `http://`)

### IP Whitelisting
- In RunPod dashboard, restrict access to your IP
- Prevents unauthorized usage

---

## 📊 When to Use Remote GPU

### ✅ Use Remote GPU When:
- Local machine has low RAM (<32GB)
- Testing VLM features extensively (many captions/questions)
- Want faster inference (RTX 4090/5090 vs local GPU)
- Running out of local VRAM
- Need consistent performance (no system freezing)

### ❌ Use Local GPU When:
- Single dataset analysis
- Occasional questions
- Privacy concerns (data stays local)
- No internet connection
- Want zero latency
- Free tier usage (no RunPod costs)

---

## 🎨 UI Reference

### Settings Location
```
CuteCaption App
└─ Load Dataset
   └─ AI Dataset Intelligence Workspace (bottom panel)
      └─ ⚙️ Settings button (top-right)
         └─ Settings Panel
            └─ Scroll to "Remote GPU (RunPod)" section
```

### Visual Indicators
- **Toggle ON**: Purple checkmark
- **Toggle OFF**: Gray
- **Text Input**: Dark background, light text
- **Password Input**: Masked dots (•••)
- **Test Button**: Purple glow on hover
- **Success**: Green background, ✅ icon
- **Error**: Red background, ❌ icon

---

**Questions?** See full documentation:
- Technical: `remote-gpu/README.md`
- Deployment: `remote-gpu/RUNPOD_DEPLOYMENT.md`
- Implementation: `REMOTE_GPU_IMPLEMENTATION_COMPLETE.md`

