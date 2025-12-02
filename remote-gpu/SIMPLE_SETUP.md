# Simple RunPod Setup (No Docker)

## Step 1: Deploy a RunPod Pod

1. Go to https://www.runpod.io/console/pods
2. Click "Deploy" 
3. Choose any GPU (RTX 4090 recommended)
4. **Template**: Use "RunPod PyTorch" or any Python template
5. **Container Disk**: 20GB minimum
6. **Volume**: 50GB → Mount at `/workspace` (for model cache)
7. Deploy

## Step 2: Copy Files to RunPod

Once your pod is running, click "Connect" → "SSH" and copy the SSH command.

Then from your local machine, run:

```bash
# From your local machine (in the working_8_30 directory)
scp -r remote-gpu python/intelligence runpod-username@your-pod-ip:/workspace/
```

Or use RunPod's file browser:
1. Click "Connect" → "File Browser"
2. Upload the `remote-gpu` folder to `/workspace/`
3. Upload the `python/intelligence` folder to `/workspace/`

## Step 3: Run Setup Script

SSH into your pod, then:

```bash
cd /workspace
chmod +x remote-gpu/simple-setup.sh
./remote-gpu/simple-setup.sh
```

This will:
- Install PyTorch
- Install all dependencies
- Start the server on port 8080

## Step 4: Get Connection URL

1. In RunPod, click "Connect" on your pod
2. Find "HTTP Service [8080]" 
3. Copy that URL (e.g., `https://abc123-8080.proxy.runpod.net`)

## Step 5: Configure CuteCaption

1. Open CuteCaption
2. Settings → AI Intelligence → Remote GPU
3. Enable "Use Remote GPU"
4. Server URL: Paste the RunPod HTTP URL
5. API Key: Set `CUTECAPTION_API_KEY` environment variable in RunPod (or use default `dev-key`)
6. Test Connection → Save

## Done!

The server will keep running as long as your pod is active. To stop it, just stop the pod in RunPod.

