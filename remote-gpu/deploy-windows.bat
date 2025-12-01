@echo off
REM ============================================================================
REM CuteCaption RunPod Deployment Script (Windows)
REM Automates Docker build, push, and provides RunPod template configuration
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo    CuteCaption RunPod Deployment
echo ============================================================================
echo.

REM ============================================================================
REM Configuration
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "DOCKERFILE_PATH=remote-gpu\Dockerfile"

echo [Step 1/6] Configuration
echo.

REM Get Docker Hub username
set /p DOCKERHUB_USERNAME="Enter your Docker Hub username: "
if "%DOCKERHUB_USERNAME%"=="" (
    echo ERROR: Docker Hub username is required
    goto :error
)

REM Get image version
set /p IMAGE_VERSION="Enter image version (default: v1.0): "
if "%IMAGE_VERSION%"=="" set "IMAGE_VERSION=v1.0"

set "IMAGE_NAME=cutecaption-gpu"
set "FULL_IMAGE_TAG=%DOCKERHUB_USERNAME%/%IMAGE_NAME%:%IMAGE_VERSION%"

echo.
echo Configuration:
echo   - Docker Hub User: %DOCKERHUB_USERNAME%
echo   - Image Name: %IMAGE_NAME%
echo   - Version: %IMAGE_VERSION%
echo   - Full Tag: %FULL_IMAGE_TAG%
echo.

pause

REM ============================================================================
REM Check Docker
REM ============================================================================

echo.
echo [Step 2/6] Checking Docker
echo.

docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH
    echo.
    echo Please install Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop
    goto :error
)

docker ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running
    echo.
    echo Please start Docker Desktop and try again
    goto :error
)

echo ✓ Docker is running
echo.

REM ============================================================================
REM Docker Login
REM ============================================================================

echo.
echo [Step 3/6] Docker Hub Login
echo.

docker login
if errorlevel 1 (
    echo ERROR: Docker login failed
    goto :error
)

echo ✓ Logged in to Docker Hub
echo.

REM ============================================================================
REM Build Image
REM ============================================================================

echo.
echo [Step 4/6] Building Docker Image
echo.
echo This may take 10-15 minutes depending on internet speed...
echo.

cd /d "%PROJECT_ROOT%"

docker build --platform linux/amd64 ^
    -t %FULL_IMAGE_TAG% ^
    -f %DOCKERFILE_PATH% ^
    .

if errorlevel 1 (
    echo ERROR: Docker build failed
    goto :error
)

echo.
echo ✓ Image built successfully: %FULL_IMAGE_TAG%
echo.

REM ============================================================================
REM Test Image Locally (Optional)
REM ============================================================================

echo.
echo [Step 5/6] Test Image Locally (Optional)
echo.

set /p TEST_LOCAL="Test image locally before pushing? (y/n, default: n): "
if /i "%TEST_LOCAL%"=="y" (
    echo.
    echo Starting test server on http://localhost:8080
    echo Press Ctrl+C to stop when done testing
    echo.
    
    docker run -p 8080:8080 ^
        -e CUTECAPTION_API_KEY=test123 ^
        %FULL_IMAGE_TAG%
    
    echo.
    echo Test completed
    echo.
)

REM ============================================================================
REM Push Image
REM ============================================================================

echo.
echo [Step 6/6] Pushing to Docker Hub
echo.
echo This may take 10-15 minutes depending on internet speed...
echo.

docker push %FULL_IMAGE_TAG%

if errorlevel 1 (
    echo ERROR: Docker push failed
    goto :error
)

echo.
echo ✓ Image pushed successfully to Docker Hub
echo.

REM ============================================================================
REM Generate API Key
REM ============================================================================

echo.
echo ============================================================================
echo   Deployment Successful!
echo ============================================================================
echo.

REM Generate a simple API key using PowerShell
for /f "delims=" %%i in ('powershell -command "[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes([guid]::NewGuid().ToString()))"') do set "GENERATED_API_KEY=%%i"

echo.
echo 📦 Image Details:
echo   - Docker Hub: https://hub.docker.com/r/%DOCKERHUB_USERNAME%/%IMAGE_NAME%
echo   - Image Tag: %FULL_IMAGE_TAG%
echo.
echo 🔑 Generated API Key (save this!):
echo   %GENERATED_API_KEY%
echo.

REM ============================================================================
REM RunPod Template Instructions
REM ============================================================================

echo.
echo 📋 Next Steps: Create RunPod Template
echo ============================================================================
echo.
echo 1. Go to: https://www.runpod.io/console/user/templates
echo.
echo 2. Click "New Template"
echo.
echo 3. Fill in these values:
echo.
echo    Basic Info:
echo      - Template Name: CuteCaption GPU Server
echo      - Container Image: %FULL_IMAGE_TAG%
echo      - Docker Command: (leave blank)
echo.
echo    Container:
echo      - Container Disk: 20 GB
echo      - Expose HTTP Ports: 8080
echo      - Expose TCP Ports: (leave blank)
echo.
echo    Volume (Important!):
echo      - Volume Disk: 50 GB
echo      - Volume Mount Path: /workspace/models
echo.
echo    Environment Variables (click "+ Add" for each):
echo.
echo      Variable 1:
echo        Key: CUTECAPTION_API_KEY
echo        Default Value: (leave empty)
echo        Description: API key for authentication
echo        ☑ User Configurable: YES (checked)
echo.
echo      Variable 2:
echo        Key: CUTECAPTION_MODEL_CACHE
echo        Default Value: /workspace/models
echo        ☐ User Configurable: NO
echo.
echo      Variable 3:
echo        Key: CUTECAPTION_PORT
echo        Default Value: 8080
echo        ☐ User Configurable: NO
echo.
echo      Variable 4:
echo        Key: CUTECAPTION_HOST
echo        Default Value: 0.0.0.0
echo        ☐ User Configurable: NO
echo.
echo 4. Click "Save Template"
echo.
echo 5. Deploy a test Pod:
echo    - Go to: https://www.runpod.io/console/pods
echo    - Click "Deploy" on your template
echo    - Select GPU: RTX 4090 (recommended)
echo    - Enter API Key: %GENERATED_API_KEY%
echo    - Click "Deploy On-Demand"
echo.
echo 6. Once running, click "Connect" → "Connect to HTTP Service [8080]"
echo.
echo 7. Verify dashboard shows GPU info
echo.
echo ============================================================================
echo.

REM ============================================================================
REM Save Configuration
REM ============================================================================

echo.
echo 💾 Saving deployment configuration...
echo.

set "CONFIG_FILE=%SCRIPT_DIR%deployment-config.txt"

(
    echo CuteCaption RunPod Deployment Configuration
    echo Generated: %date% %time%
    echo.
    echo Docker Hub Username: %DOCKERHUB_USERNAME%
    echo Image Name: %IMAGE_NAME%
    echo Image Version: %IMAGE_VERSION%
    echo Full Image Tag: %FULL_IMAGE_TAG%
    echo.
    echo Generated API Key: %GENERATED_API_KEY%
    echo.
    echo Docker Hub URL: https://hub.docker.com/r/%DOCKERHUB_USERNAME%/%IMAGE_NAME%
    echo RunPod Templates: https://www.runpod.io/console/user/templates
    echo RunPod Pods: https://www.runpod.io/console/pods
    echo.
    echo Next Steps:
    echo 1. Create RunPod template (see instructions above)
    echo 2. Deploy test Pod
    echo 3. Connect to dashboard
    echo 4. Configure CuteCaption app with Pod URL and API key
) > "%CONFIG_FILE%"

echo ✓ Configuration saved to: %CONFIG_FILE%
echo.

REM ============================================================================
REM Open Browser
REM ============================================================================

set /p OPEN_BROWSER="Open RunPod templates page in browser? (y/n, default: y): "
if /i not "%OPEN_BROWSER%"=="n" (
    start https://www.runpod.io/console/user/templates
)

echo.
echo ============================================================================
echo   Deployment Complete!
echo ============================================================================
echo.
echo For detailed instructions, see:
echo   - Deployment Checklist: %PROJECT_ROOT%\remote-gpu\DEPLOYMENT_CHECKLIST.md
echo   - Quickstart Guide: %PROJECT_ROOT%\RUNPOD_QUICKSTART_GUIDE.md
echo   - User Guide: %PROJECT_ROOT%\remote-gpu\USER_GUIDE.md
echo.

pause
goto :end

REM ============================================================================
REM Error Handler
REM ============================================================================

:error
echo.
echo ============================================================================
echo   Deployment Failed
echo ============================================================================
echo.
echo Please check the error messages above and try again.
echo.
echo Common issues:
echo   - Docker not installed: https://www.docker.com/products/docker-desktop
echo   - Docker not running: Start Docker Desktop
echo   - Network issues: Check internet connection
echo   - Build errors: Check Dockerfile and dependencies
echo.
echo For help, see troubleshooting guide:
echo   %PROJECT_ROOT%\RUNPOD_QUICKSTART_GUIDE.md
echo.
pause
exit /b 1

:end
endlocal



