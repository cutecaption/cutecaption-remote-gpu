"""
CuteCaption Remote GPU Server
FastAPI service for offloading GPU-intensive inference to RunPod

ARCHITECTURE:
- FastAPI for HTTP/WebSocket endpoints
- Persistent model loading (load once, keep in VRAM)
- Request batching for efficiency
- WebSocket for real-time progress updates
- Graceful shutdown with model cleanup

ENDPOINTS:
- GET /status - GPU info, loaded models, memory usage
- POST /api/v1/vlm/classify - VLM image classification
- POST /api/v1/vlm/caption - VLM image captioning
- POST /api/v1/vlm/verify - VLM candidate verification
- POST /api/v1/vlm/synthesize - VLM answer synthesis
- POST /api/v1/semantic/search - CLIP semantic search
- POST /api/v1/semantic/embed - CLIP embedding generation
- POST /api/v1/pose/detect - MediaPipe pose detection
- WS /ws - WebSocket for live progress updates
"""

import os
import sys
import asyncio
import logging
import base64
import io
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import psutil
import GPUtil

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.getenv('CUTECAPTION_API_KEY', 'dev-key-change-in-production')
HOST = os.getenv('CUTECAPTION_HOST', '0.0.0.0')
PORT = int(os.getenv('CUTECAPTION_PORT', '8080'))
MODEL_CACHE = os.getenv('CUTECAPTION_MODEL_CACHE', '/workspace/models')

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="CuteCaption Remote GPU Server",
    description="High-performance GPU inference service for CuteCaption",
    version="1.0.0"
)

# CORS - allow CuteCaption Electron app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, lock this down to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SERVICE REGISTRY (Lazy loading)
# ============================================================================

class ServiceRegistry:
    """Manages inference services (VLM, Semantic, Pose, JoyCaption, Qwen Caption)"""
    
    def __init__(self):
        self.vlm_service = None
        self.semantic_service = None
        self.pose_service = None
        self.joycaption_service = None
        self.qwen_caption_service = None
        self.services_initialized = {
            'vlm': False,
            'semantic': False,
            'pose': False,
            'joycaption': False,
            'qwen_caption': False
        }
        
    async def get_vlm_service(self):
        """Lazy load VLM service"""
        if self.vlm_service is None:
            logger.info("[ServiceRegistry] Loading VLM service...")
            start = time.time()
            
            # Add python intelligence path
            # In Docker: /workspace/server/main.py -> /workspace/python/intelligence
            # Path structure: /workspace/server/../python/intelligence = /workspace/python/intelligence
            python_path = Path(__file__).parent.parent / 'python' / 'intelligence'
            sys.path.insert(0, str(python_path))
            logger.info(f"[ServiceRegistry] Python intelligence path: {python_path}")
            
            from vlm_engine import VLMEngine
            self.vlm_service = VLMEngine()
            
            # Initialize (loads model)
            await self.vlm_service.initialize()
            
            elapsed = time.time() - start
            logger.info(f"[ServiceRegistry] VLM service loaded in {elapsed:.2f}s")
            self.services_initialized['vlm'] = True
            
        return self.vlm_service
    
    async def get_semantic_service(self):
        """Lazy load Semantic service"""
        if self.semantic_service is None:
            logger.info("[ServiceRegistry] Loading Semantic service...")
            start = time.time()
            
            python_path = Path(__file__).parent.parent / 'python' / 'intelligence'
            sys.path.insert(0, str(python_path))
            logger.info(f"[ServiceRegistry] Python intelligence path: {python_path}")
            
            from semantic_engine import SemanticEngine
            self.semantic_service = SemanticEngine()
            await self.semantic_service.initialize()
            
            elapsed = time.time() - start
            logger.info(f"[ServiceRegistry] Semantic service loaded in {elapsed:.2f}s")
            self.services_initialized['semantic'] = True
            
        return self.semantic_service
    
    async def get_pose_service(self):
        """Lazy load Pose service"""
        if self.pose_service is None:
            logger.info("[ServiceRegistry] Loading Pose service...")
            start = time.time()
            
            python_path = Path(__file__).parent.parent / 'python' / 'intelligence'
            sys.path.insert(0, str(python_path))
            logger.info(f"[ServiceRegistry] Python intelligence path: {python_path}")
            
            from pose_engine import PoseEngine
            self.pose_service = PoseEngine()
            await self.pose_service.initialize()
            
            elapsed = time.time() - start
            logger.info(f"[ServiceRegistry] Pose service loaded in {elapsed:.2f}s")
            self.services_initialized['pose'] = True
            
        return self.pose_service
    
    async def get_joycaption_service(self):
        """Lazy load JoyCaption service"""
        if self.joycaption_service is None:
            logger.info("[ServiceRegistry] Loading JoyCaption service...")
            start = time.time()
            
            python_path = Path(__file__).parent.parent / 'python' / 'intelligence'
            sys.path.insert(0, str(python_path))
            logger.info(f"[ServiceRegistry] Python intelligence path: {python_path}")
            
            from joycaption_engine import JoyCaptionEngine
            self.joycaption_service = JoyCaptionEngine()
            await self.joycaption_service.initialize()
            
            elapsed = time.time() - start
            logger.info(f"[ServiceRegistry] JoyCaption service loaded in {elapsed:.2f}s")
            self.services_initialized['joycaption'] = True
            
        return self.joycaption_service
    
    async def get_qwen_caption_service(self):
        """Lazy load Qwen Caption service (reuses VLM engine with caption mode)"""
        if self.qwen_caption_service is None:
            logger.info("[ServiceRegistry] Loading Qwen Caption service...")
            start = time.time()
            
            # Qwen Caption reuses the VLM service but in caption mode
            # If VLM is already loaded, just reference it
            if self.vlm_service is not None:
                self.qwen_caption_service = self.vlm_service
                logger.info("[ServiceRegistry] Qwen Caption using existing VLM service")
            else:
                # Load VLM if not already loaded
                self.qwen_caption_service = await self.get_vlm_service()
            
            elapsed = time.time() - start
            logger.info(f"[ServiceRegistry] Qwen Caption service loaded in {elapsed:.2f}s")
            self.services_initialized['qwen_caption'] = True
            
        return self.qwen_caption_service
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all services"""
        return {
            'vlm': {
                'loaded': self.services_initialized['vlm'],
                'ready': self.vlm_service is not None
            },
            'semantic': {
                'loaded': self.services_initialized['semantic'],
                'ready': self.semantic_service is not None
            },
            'pose': {
                'loaded': self.services_initialized['pose'],
                'ready': self.pose_service is not None
            },
            'joycaption': {
                'loaded': self.services_initialized['joycaption'],
                'ready': self.joycaption_service is not None
            },
            'qwen_caption': {
                'loaded': self.services_initialized['qwen_caption'],
                'ready': self.qwen_caption_service is not None
            }
        }

# Global service registry
registry = ServiceRegistry()

# ============================================================================
# AUTHENTICATION
# ============================================================================

async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key from header"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class VLMClassifyRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to image file")
    max_resolution: int = Field(1024, description="Max resolution for inference")

class VLMCaptionRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to image file")
    max_resolution: int = Field(1024, description="Max resolution for inference")
    prompt: Optional[str] = Field(None, description="Optional caption prompt")

class VLMVerifyRequest(BaseModel):
    candidates: List[str] = Field(..., description="List of image paths to verify")
    query: str = Field(..., description="Query to verify against")
    max_resolution: int = Field(1024, description="Max resolution for inference")

class VLMSynthesizeRequest(BaseModel):
    question: str = Field(..., description="User's question")
    evidence: Dict[str, Any] = Field(..., description="Evidence from search/analysis")
    persona: str = Field('professional', description="Response persona")
    representative_images: Optional[List[str]] = Field(None, description="Sample images")
    confidence_metrics: Optional[Dict[str, Any]] = Field(None, description="Confidence data")

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Text query")
    query_type: str = Field('auto', description="Query type: person/object/scene/auto")
    top_k: int = Field(50, description="Number of results")
    use_multi_level: bool = Field(False, description="Use multi-level indexes")

class SemanticEmbedRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Image path to embed")
    text: Optional[str] = Field(None, description="Text to embed")

class PoseDetectRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to image file")

# JoyCaption Request Models
class JoyCaptionRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to image file")
    prompt: Optional[str] = Field(None, description="Custom caption prompt")
    temperature: float = Field(0.7, description="Temperature for generation")
    top_p: float = Field(0.9, description="Top-p sampling")
    max_tokens: int = Field(512, description="Max tokens to generate")

class JoyCaptionBatchRequest(BaseModel):
    image_paths: List[str] = Field(..., description="List of image paths to caption")
    prompt: Optional[str] = Field(None, description="Custom caption prompt")
    temperature: float = Field(0.7, description="Temperature for generation")
    top_p: float = Field(0.9, description="Top-p sampling")
    max_tokens: int = Field(512, description="Max tokens to generate")

# Qwen Caption Request Models
class QwenCaptionRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to image file")
    prompt: Optional[str] = Field(None, description="Custom caption prompt")
    max_resolution: int = Field(1024, description="Max resolution for inference")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Max tokens to generate")

class QwenVideoCaptionRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to video file")
    prompt: Optional[str] = Field(None, description="Custom caption prompt")
    max_frames: int = Field(64, description="Max frames to extract")
    fps: int = Field(6, description="Frames per second for extraction")
    max_resolution: int = Field(1024, description="Max resolution for inference")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Max tokens to generate")

# ============================================================================
# SYSTEM INFO HELPERS
# ============================================================================

def _safe_float(value, default=0.0):
    """Convert value to JSON-safe float (handles inf, nan)"""
    import math
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default

def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information"""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {'available': False, 'reason': 'No GPUs detected'}
        
        gpu = gpus[0]  # Primary GPU
        return {
            'available': True,
            'name': gpu.name or 'Unknown GPU',
            'driver': gpu.driver or 'Unknown',
            'memory_total_mb': _safe_float(gpu.memoryTotal),
            'memory_used_mb': _safe_float(gpu.memoryUsed),
            'memory_free_mb': _safe_float(gpu.memoryFree),
            'memory_utilization_percent': round(_safe_float(gpu.memoryUtil) * 100, 1),
            'gpu_utilization_percent': round(_safe_float(gpu.load) * 100, 1),
            'temperature_c': _safe_float(gpu.temperature, default=None)
        }
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return {'available': False, 'error': str(e)}

def get_system_info() -> Dict[str, Any]:
    """Get system resource info"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'memory_used_gb': round(memory.used / (1024**3), 2),
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'memory_percent': memory.percent,
            'disk_total_gb': round(disk.total / (1024**3), 2),
            'disk_used_gb': round(disk.used / (1024**3), 2),
            'disk_free_gb': round(disk.free / (1024**3), 2),
            'disk_percent': disk.percent
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {'error': str(e)}

# ============================================================================
# STATUS ENDPOINT (Public, no auth required for monitoring)
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - serve dashboard"""
    dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'index.html'
    
    if dashboard_path.exists():
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return {"message": "CuteCaption Remote GPU Server", "dashboard": "/dashboard"}

@app.get("/dashboard")
async def dashboard():
    """Dashboard page"""
    dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'index.html'
    
    if dashboard_path.exists():
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return {"error": "Dashboard not found"}

@app.get("/status")
async def get_status():
    """Get comprehensive server status"""
    gpu_info = get_gpu_info()
    system_info = get_system_info()
    services = registry.get_status()
    
    return {
        'server': 'CuteCaption Remote GPU',
        'version': '1.0.0',
        'uptime_seconds': time.time() - app.state.start_time,
        'gpu': gpu_info,
        'system': system_info,
        'services': services,
        'ready': any(services[s]['loaded'] for s in services)
    }

# ============================================================================
# MODEL LOADING ENDPOINTS
# ============================================================================

@app.post("/api/v1/load", dependencies=[Depends(verify_api_key)])
async def load_model(service: str = "vlm"):
    """
    Explicitly load a model into VRAM.
    Services: vlm, semantic, pose, joycaption, qwen_caption
    """
    try:
        if service == "vlm":
            logger.info("[API] Loading VLM service...")
            await registry.get_vlm_service()
            return {"success": True, "service": "vlm", "message": "VLM model loaded successfully"}
        elif service == "semantic":
            logger.info("[API] Loading Semantic service...")
            await registry.get_semantic_service()
            return {"success": True, "service": "semantic", "message": "Semantic model loaded successfully"}
        elif service == "pose":
            logger.info("[API] Loading Pose service...")
            await registry.get_pose_service()
            return {"success": True, "service": "pose", "message": "Pose model loaded successfully"}
        elif service == "joycaption":
            logger.info("[API] Loading JoyCaption service...")
            await registry.get_joycaption_service()
            return {"success": True, "service": "joycaption", "message": "JoyCaption model loaded successfully"}
        elif service == "qwen_caption":
            logger.info("[API] Loading Qwen Caption service...")
            await registry.get_qwen_caption_service()
            return {"success": True, "service": "qwen_caption", "message": "Qwen Caption model loaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown service: {service}. Use: vlm, semantic, pose, joycaption, qwen_caption")
    except Exception as e:
        logger.error(f"Failed to load {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/unload", dependencies=[Depends(verify_api_key)])
async def unload_model(service: str = "vlm"):
    """
    Unload a model from VRAM.
    Note: Full unload requires server restart for complete memory release.
    """
    try:
        if service == "vlm":
            if registry.vlm_service:
                registry.vlm_service = None
                registry.services_initialized['vlm'] = False
                logger.info("[API] VLM service unloaded")
                return {"success": True, "service": "vlm", "message": "VLM model unloaded"}
            return {"success": True, "service": "vlm", "message": "VLM was not loaded"}
        elif service == "semantic":
            if registry.semantic_service:
                registry.semantic_service = None
                registry.services_initialized['semantic'] = False
                logger.info("[API] Semantic service unloaded")
                return {"success": True, "service": "semantic", "message": "Semantic model unloaded"}
            return {"success": True, "service": "semantic", "message": "Semantic was not loaded"}
        elif service == "pose":
            if registry.pose_service:
                registry.pose_service = None
                registry.services_initialized['pose'] = False
                logger.info("[API] Pose service unloaded")
                return {"success": True, "service": "pose", "message": "Pose model unloaded"}
            return {"success": True, "service": "pose", "message": "Pose was not loaded"}
        elif service == "joycaption":
            if registry.joycaption_service:
                registry.joycaption_service = None
                registry.services_initialized['joycaption'] = False
                logger.info("[API] JoyCaption service unloaded")
                return {"success": True, "service": "joycaption", "message": "JoyCaption model unloaded"}
            return {"success": True, "service": "joycaption", "message": "JoyCaption was not loaded"}
        elif service == "qwen_caption":
            if registry.qwen_caption_service:
                registry.qwen_caption_service = None
                registry.services_initialized['qwen_caption'] = False
                logger.info("[API] Qwen Caption service unloaded")
                return {"success": True, "service": "qwen_caption", "message": "Qwen Caption model unloaded"}
            return {"success": True, "service": "qwen_caption", "message": "Qwen Caption was not loaded"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown service: {service}")
    except Exception as e:
        logger.error(f"Failed to unload {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# VLM ENDPOINTS
# ============================================================================

@app.post("/api/v1/vlm/classify", dependencies=[Depends(verify_api_key)])
async def vlm_classify(request: VLMClassifyRequest):
    """VLM image classification"""
    try:
        service = await registry.get_vlm_service()
        
        result = await service.process({
            'action': 'classify_image',
            'image_path': request.image_path,
            'max_resolution': request.max_resolution
        })
        
        return result
        
    except Exception as e:
        logger.error(f"VLM classify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/vlm/caption", dependencies=[Depends(verify_api_key)])
async def vlm_caption(request: VLMCaptionRequest):
    """VLM image captioning"""
    try:
        service = await registry.get_vlm_service()
        
        result = await service.process({
            'action': 'caption_image',
            'image_path': request.image_path,
            'max_resolution': request.max_resolution,
            'prompt': request.prompt
        })
        
        return result
        
    except Exception as e:
        logger.error(f"VLM caption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/vlm/verify", dependencies=[Depends(verify_api_key)])
async def vlm_verify(request: VLMVerifyRequest):
    """VLM candidate verification"""
    try:
        service = await registry.get_vlm_service()
        
        result = await service.process({
            'action': 'verify_candidates',
            'candidates': request.candidates,
            'query': request.query,
            'max_resolution': request.max_resolution
        })
        
        return result
        
    except Exception as e:
        logger.error(f"VLM verify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/vlm/synthesize", dependencies=[Depends(verify_api_key)])
async def vlm_synthesize(request: VLMSynthesizeRequest):
    """VLM answer synthesis"""
    try:
        service = await registry.get_vlm_service()
        
        result = await service.process({
            'action': 'synthesize_answer',
            'question': request.question,
            'evidence': request.evidence,
            'persona': request.persona,
            'representative_images': request.representative_images,
            'confidence_metrics': request.confidence_metrics
        })
        
        return result
        
    except Exception as e:
        logger.error(f"VLM synthesize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SEMANTIC ENDPOINTS
# ============================================================================

@app.post("/api/v1/semantic/search", dependencies=[Depends(verify_api_key)])
async def semantic_search(request: SemanticSearchRequest):
    """CLIP semantic search"""
    try:
        service = await registry.get_semantic_service()
        
        action = 'search_multi_level' if request.use_multi_level else 'search_similar'
        
        result = await service.process({
            'action': action,
            'query': request.query,
            'query_type': request.query_type,
            'top_k': request.top_k
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/semantic/embed", dependencies=[Depends(verify_api_key)])
async def semantic_embed(request: SemanticEmbedRequest):
    """Generate CLIP embeddings"""
    try:
        service = await registry.get_semantic_service()
        
        if request.image_path:
            result = await service.process({
                'action': 'embed_image',
                'image_path': request.image_path
            })
        elif request.text:
            result = await service.process({
                'action': 'embed_text',
                'text': request.text
            })
        else:
            raise HTTPException(status_code=400, detail="Must provide image_path or text")
        
        return result
        
    except Exception as e:
        logger.error(f"Semantic embed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SemanticProjectRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    method: str = Field('tsne', description="Projection method: tsne or umap")
    perplexity: int = Field(30, description="t-SNE perplexity parameter")

@app.post("/api/v1/semantic/project", dependencies=[Depends(verify_api_key)])
async def semantic_project(request: SemanticProjectRequest):
    """Project embeddings to 2D for visualization (t-SNE/UMAP)"""
    try:
        service = await registry.get_semantic_service()
        
        result = await service.process({
            'action': 'project_embeddings',
            'embeddings': request.embeddings,
            'method': request.method,
            'perplexity': request.perplexity
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Semantic project error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# POSE ENDPOINT
# ============================================================================

@app.post("/api/v1/pose/detect", dependencies=[Depends(verify_api_key)])
async def pose_detect(request: PoseDetectRequest):
    """MediaPipe pose detection"""
    try:
        service = await registry.get_pose_service()
        
        result = await service.process({
            'action': 'detect_pose',
            'image_path': request.image_path
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Pose detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# JOYCAPTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/joycaption/caption", dependencies=[Depends(verify_api_key)])
async def joycaption_caption(request: JoyCaptionRequest):
    """Generate caption using JoyCaption model"""
    try:
        service = await registry.get_joycaption_service()
        
        result = await service.process({
            'action': 'caption_image',
            'image_path': request.image_path,
            'prompt': request.prompt,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'max_tokens': request.max_tokens
        })
        
        return result
        
    except Exception as e:
        logger.error(f"JoyCaption caption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/joycaption/batch", dependencies=[Depends(verify_api_key)])
async def joycaption_batch(request: JoyCaptionBatchRequest):
    """Batch caption using JoyCaption model"""
    try:
        service = await registry.get_joycaption_service()
        
        results = []
        for image_path in request.image_paths:
            result = await service.process({
                'action': 'caption_image',
                'image_path': image_path,
                'prompt': request.prompt,
                'temperature': request.temperature,
                'top_p': request.top_p,
                'max_tokens': request.max_tokens
            })
            results.append({
                'image_path': image_path,
                'caption': result.get('caption', ''),
                'success': result.get('success', True)
            })
        
        return {
            'success': True,
            'results': results,
            'total': len(results)
        }
        
    except Exception as e:
        logger.error(f"JoyCaption batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# QWEN CAPTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/qwen-caption/caption", dependencies=[Depends(verify_api_key)])
async def qwen_caption(request: QwenCaptionRequest):
    """Generate caption using Qwen3-VL model"""
    try:
        service = await registry.get_qwen_caption_service()
        
        # Use VLM service with caption-specific prompt
        caption_prompt = request.prompt or "Describe this image in detail."
        
        result = await service.process({
            'action': 'caption_image',
            'image_path': request.image_path,
            'prompt': caption_prompt,
            'max_resolution': request.max_resolution,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Qwen caption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/qwen-caption/video", dependencies=[Depends(verify_api_key)])
async def qwen_caption_video(request: QwenVideoCaptionRequest):
    """Generate video caption using Qwen3-VL model"""
    try:
        service = await registry.get_qwen_caption_service()
        
        # Use VLM service with video caption mode
        caption_prompt = request.prompt or "Describe what happens in this video in detail."
        
        result = await service.process({
            'action': 'caption_video',
            'video_path': request.video_path,
            'prompt': caption_prompt,
            'max_frames': request.max_frames,
            'fps': request.fps,
            'max_resolution': request.max_resolution,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Qwen video caption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to WebSocket: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Server startup"""
    app.state.start_time = time.time()
    logger.info("=" * 80)
    logger.info("CuteCaption Remote GPU Server Starting")
    logger.info("=" * 80)
    logger.info(f"Host: {HOST}:{PORT}")
    logger.info(f"Model Cache: {MODEL_CACHE}")
    logger.info(f"API Key: {'*' * (len(API_KEY) - 4)}{API_KEY[-4:]}")
    
    # Log GPU info
    gpu_info = get_gpu_info()
    if gpu_info.get('available'):
        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"VRAM: {gpu_info['memory_total_mb']} MB")
    else:
        logger.warning("No GPU detected!")
    
    logger.info("=" * 80)
    logger.info("Server ready for connections")
    logger.info("Services will be loaded on first request (lazy loading)")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown - cleanup models"""
    logger.info("Shutting down server...")
    
    # Cleanup services
    if registry.vlm_service:
        logger.info("Unloading VLM service...")
        # Add cleanup if needed
    
    if registry.semantic_service:
        logger.info("Unloading Semantic service...")
        # Add cleanup if needed
    
    if registry.pose_service:
        logger.info("Unloading Pose service...")
        # Add cleanup if needed
    
    logger.info("Shutdown complete")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )

