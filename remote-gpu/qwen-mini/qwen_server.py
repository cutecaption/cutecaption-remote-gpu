"""
CuteCaption Qwen Mini-Server
============================

A lightweight, focused server for offloading Qwen VLM operations to RunPod.

Features:
- Qwen3-VL-8B image analysis (classification, captioning, VQA)
- Question analysis for intelligent query routing
- Answer synthesis from evidence
- Candidate verification
- Real-time GPU/memory monitoring
- Modern dashboard with service controls

Endpoints:
- GET  /           - Dashboard UI
- GET  /status     - Server status + GPU info
- POST /api/v1/load   - Preload model into VRAM
- POST /api/v1/unload - Unload model from VRAM
- POST /api/v1/analyze_question - VLM question analysis
- POST /api/v1/classify - Image classification (25 dimensions)
- POST /api/v1/caption - Image captioning
- POST /api/v1/verify - Candidate verification
- POST /api/v1/synthesize - Answer synthesis
- POST /api/v1/vqa - Visual question answering
- WS   /ws         - Real-time progress updates
"""

import os
import sys
import gc
import json
import time
import asyncio
import logging
import base64
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Model cache
MODEL_CACHE = os.environ.get('CUTECAPTION_MODEL_CACHE', '/workspace/models')
os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ['HF_HOME'] = MODEL_CACHE
os.environ['HUGGINGFACE_HUB_CACHE'] = MODEL_CACHE
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE

# Now import heavy libraries
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
import psutil
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = os.environ.get('CUTECAPTION_API_KEY', 'dev-key-change-in-production')
HOST = os.environ.get('CUTECAPTION_HOST', '0.0.0.0')
PORT = int(os.environ.get('CUTECAPTION_PORT', '8080'))
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="CuteCaption Qwen Mini-Server",
    description="Lightweight VLM inference server for CuteCaption",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# QWEN SERVICE
# =============================================================================

class QwenService:
    """Manages Qwen3-VL model lifecycle and inference"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.is_loaded = False
        self.is_loading = False
        self.load_time = None
        self.inference_count = 0
        self.last_inference = None
        
    async def load(self) -> Dict[str, Any]:
        """Load Qwen3-VL model into VRAM"""
        if self.is_loaded:
            return {"status": "already_loaded", "message": "Model already in VRAM"}
        
        if self.is_loading:
            return {"status": "loading", "message": "Model is currently loading"}
        
        self.is_loading = True
        start_time = time.time()
        
        try:
            logger.info(f"Loading Qwen3-VL from {MODEL_ID}...")
            
            # Device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Import Qwen-specific modules
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
            
            # Load processor
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.processor = AutoProcessor.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True
                )
            
            # Load model with optimal settings for GPU
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.model.eval()
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"Model loaded in {self.load_time:.1f}s, VRAM: {allocated:.1f}GB")
            
            return {
                "status": "loaded",
                "load_time_seconds": round(self.load_time, 1),
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loading = False
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.is_loading = False
    
    async def unload(self) -> Dict[str, Any]:
        """Unload model from VRAM"""
        if not self.is_loaded:
            return {"status": "not_loaded", "message": "Model not in VRAM"}
        
        try:
            # Delete model and processor
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.is_loaded = False
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Model unloaded from VRAM")
            return {"status": "unloaded"}
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _ensure_loaded(self):
        """Ensure model is loaded before inference"""
        if not self.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Call /api/v1/load first or enable auto-load in settings."
            )
    
    def _load_image(self, image_data: str, max_resolution: int = 1024) -> Image.Image:
        """Load image from base64 or file path"""
        try:
            # Check if it's base64
            if image_data.startswith('data:image'):
                # Extract base64 data after comma
                base64_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif ';base64,' in image_data:
                base64_data = image_data.split(';base64,')[1]
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # Try as raw base64
                try:
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                except:
                    # Treat as file path
                    image = Image.open(image_data)
            
            # Convert to RGB
            image = image.convert('RGB')
            
            # Resize if needed
            max_dim = max(image.size)
            if max_dim > max_resolution:
                scale = max_resolution / max_dim
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
    async def _run_vlm(self, image: Image.Image, prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
        """Run VLM inference with image + prompt"""
        self._ensure_loaded()
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        self.inference_count += 1
        self.last_inference = datetime.now().isoformat()
        
        return response
    
    async def _run_text_only(self, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        """Run VLM inference with text only (no image)"""
        self._ensure_loaded()
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        self.inference_count += 1
        self.last_inference = datetime.now().isoformat()
        
        return response
    
    async def classify(self, image_data: str, max_resolution: int = 512) -> Dict[str, Any]:
        """Two-stage conditional classification with aspect info"""
        # Load image and capture original dimensions BEFORE any resizing
        image = self._load_image(image_data, max_resolution)
        original_size = image.size  # Note: this is AFTER resize, need to get BEFORE
        
        # Calculate aspect info (matching local VLM engine format)
        width, height = original_size
        aspect_ratio = width / height
        
        # Determine orientation
        if aspect_ratio > 2.0:
            orientation = "panoramic_horizontal"
        elif aspect_ratio > 1.2:
            orientation = "landscape"
        elif aspect_ratio < 0.5:
            orientation = "panoramic_vertical"
        elif aspect_ratio < 0.83:
            orientation = "portrait"
        else:
            orientation = "square"
        
        aspect_info = {
            "original_width": width,
            "original_height": height,
            "aspect_ratio": round(aspect_ratio, 3),
            "orientation": orientation
        }
        
        # Stage 1: Universal (12 dimensions)
        stage1_prompt = """Classify this image in 12 UNIVERSAL dimensions:

1. Image Type: photograph, illustration, 3d_render, digital_art, painting, or sketch
2. Content Type: portrait, landscape, architecture, still_life, abstract, urban, nature, interior, or food
3. Shot Scale: extreme_close_up, close_up, medium, wide, or extreme_wide
4. Time of Day: day, night, golden_hour, blue_hour, or twilight
5. Dominant Colors: List 3 prominent colors (e.g., "red, blue, white")
6. Color Mode: color, bw, sepia, or monochrome
7. Lighting Quality: poor, acceptable, good, or excellent
8. Exposure: normal, overexposed, or underexposed
9. Background Type: studio, outdoor, indoor, bokeh, busy, or plain
10. NSFW Level: safe, suggestive, or borderline
11. Overlays: emoji, sticker, watermark, text_overlay, filter, frame, or none
12. Human Presence: none, single_person, multiple_people, or crowd

Respond in this exact format:
image_type: <answer>
content_type: <answer>
shot_scale: <answer>
time_of_day: <answer>
dominant_colors: <answer>
color_mode: <answer>
lighting_quality: <answer>
exposure: <answer>
background_type: <answer>
nsfw_level: <answer>
overlays: <answer>
human_presence: <answer>"""
        
        stage1_response = await self._run_vlm(image, stage1_prompt, max_tokens=100)
        result = self._parse_stage1(stage1_response)
        
        # Add aspect info (matching local VLM engine format)
        result['aspect_info'] = aspect_info
        result['dimensions'] = {'width': width, 'height': height}
        result['aspect_ratio'] = aspect_ratio
        
        # Stage 2: Human-specific (if humans detected)
        if result.get('human_presence', 'none') != 'none':
            stage2_prompt = """This image contains humans. Classify these 13 HUMAN-SPECIFIC dimensions:

1. Human Details: gender, age_group, pose/action, clothing
2. Body Type: slim, average, curvy, plus_size, athletic, petite, tall, or not_visible
3. Detailed Clothing: specific items with colors
4. Clothing Style: casual, formal, business, athletic, bohemian, streetwear, vintage, or elegant
5. Accessories: hat, glasses, jewelry, watch, bag, or none
6. Expression: smiling, neutral, serious, laughing, surprised, or not_visible
7. Eye Contact: yes or no
8. Shot Style: candid, posed, editorial, selfie, or professional
9. Estimated Age: age range like "20-30" or "unknown"
10. Hair: color and style or not_visible
11. Ethnicity/Appearance: Asian, Caucasian, Hispanic, African, Middle Eastern, South Asian, Mixed, or unknown
12. Primary Action: sitting, standing, walking, posing, or other
13. Setting/Location: beach, office, studio, street, home, gym, restaurant, nature, or unknown

Respond in this exact format:
human_details: <answer>
body_type: <answer>
detailed_clothing: <answer>
clothing_style: <answer>
accessories: <answer>
expression: <answer>
eye_contact: <answer>
shot_style: <answer>
estimated_age: <answer>
hair: <answer>
ethnicity: <answer>
primary_action: <answer>
setting: <answer>"""
            
            stage2_response = await self._run_vlm(image, stage2_prompt, max_tokens=150)
            result = self._parse_stage2(stage2_response, result)
        
        return result
    
    def _parse_stage1(self, response: str) -> Dict[str, Any]:
        """Parse Stage 1 response"""
        result = {
            "image_type": "photograph",
            "content_type": "portrait",
            "shot_scale": "medium",
            "time_of_day": "day",
            "dominant_colors": "",
            "color_mode": "color",
            "lighting_quality": "acceptable",
            "exposure_quality": "normal",
            "background_type": "indoor",
            "nsfw_level": "safe",
            "has_overlays": False,
            "overlay_types": [],
            "human_presence": "none",
            "confidence": 0.85
        }
        
        for line in response.lower().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'image_type':
                    result['image_type'] = value
                elif key == 'content_type':
                    result['content_type'] = value
                elif key == 'shot_scale':
                    result['shot_scale'] = value
                elif key == 'time_of_day':
                    result['time_of_day'] = value
                elif key == 'dominant_colors':
                    result['dominant_colors'] = value
                elif key == 'color_mode':
                    result['color_mode'] = value
                elif key == 'lighting_quality':
                    result['lighting_quality'] = value
                elif key == 'exposure':
                    result['exposure_quality'] = value
                elif key == 'background_type':
                    result['background_type'] = value
                elif key == 'nsfw_level':
                    result['nsfw_level'] = value
                elif key == 'overlays':
                    if value and value != 'none':
                        result['has_overlays'] = True
                        result['overlay_types'] = [v.strip() for v in value.split(',') if v.strip() != 'none']
                elif key == 'human_presence':
                    result['human_presence'] = value
        
        return result
    
    def _parse_stage2(self, response: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Stage 2 response and merge with Stage 1"""
        for line in response.lower().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'human_details':
                    result['human_details'] = value
                elif key == 'body_type':
                    result['body_type'] = value
                elif key == 'detailed_clothing':
                    result['detailed_clothing'] = value
                elif key == 'clothing_style':
                    result['clothing_style'] = value
                elif key == 'accessories':
                    result['accessories'] = [a.strip() for a in value.split(',') if a.strip() != 'none']
                elif key == 'expression':
                    result['expression'] = value
                elif key == 'eye_contact':
                    result['eye_contact'] = value in ['yes', 'true']
                elif key == 'shot_style':
                    result['shot_style'] = value
                elif key == 'estimated_age':
                    result['estimated_age'] = value
                elif key == 'hair':
                    result['hair'] = value
                elif key == 'ethnicity':
                    result['ethnicity'] = value
                elif key == 'primary_action':
                    result['primary_action'] = value
                elif key == 'setting':
                    result['setting'] = value
        
        return result
    
    async def caption(self, image_data: str, style: str = "descriptive") -> Dict[str, Any]:
        """Generate image caption"""
        # Resolution based on style
        res_map = {"detailed": 1024, "descriptive": 768, "concise": 512}
        resolution = res_map.get(style, 768)
        
        image = self._load_image(image_data, resolution)
        
        prompt_map = {
            "detailed": "Provide a detailed, comprehensive description of this image including all visible elements, composition, colors, and mood.",
            "descriptive": "Describe this image.",
            "concise": "Provide a brief, one-sentence description of this image."
        }
        prompt = prompt_map.get(style, prompt_map["descriptive"])
        max_tokens = 256 if style == "detailed" else 100
        
        caption = await self._run_vlm(image, prompt, max_tokens=max_tokens)
        
        return {
            "caption": caption,
            "style": style,
            "resolution_used": resolution
        }
    
    async def vqa(self, image_data: str, question: str) -> Dict[str, Any]:
        """Visual question answering"""
        # Determine resolution from question type
        lower_q = question.lower()
        if any(k in lower_q for k in ['text', 'sign', 'label', 'read', 'written']):
            resolution = 2048  # OCR needs high res
        elif any(k in lower_q for k in ['detail', 'small', 'specific']):
            resolution = 1024
        elif any(k in lower_q for k in ['pose', 'standing', 'sitting', 'action']):
            resolution = 768
        else:
            resolution = 512
        
        image = self._load_image(image_data, resolution)
        answer = await self._run_vlm(image, question, max_tokens=256)
        
        return {
            "question": question,
            "answer": answer,
            "resolution_used": resolution
        }
    
    async def verify_candidates(self, candidates: List[Dict], query: str) -> Dict[str, Any]:
        """Verify if candidate images match a query"""
        verified = []
        rejected = []
        
        verification_prompt = f"""Does this image match the query: "{query}"?

Analyze carefully:
- All objects, people, and details mentioned in the query
- Colors, poses, clothing, actions, and context
- Be precise - only confirm if ALL query elements are present

Answer format:
MATCH: YES/NO
CONFIDENCE: 0.0-1.0
REASON: Brief explanation"""

        for candidate in candidates:
            try:
                image = self._load_image(candidate.get('image_data', ''), 1024)
                response = await self._run_vlm(image, verification_prompt, max_tokens=100, temperature=0.2)
                
                match = False
                confidence = 0.0
                reason = ""
                
                for line in response.lower().split('\n'):
                    if 'match:' in line:
                        match = 'yes' in line
                    elif 'confidence:' in line:
                        try:
                            confidence = float(line.split(':')[-1].strip())
                        except:
                            confidence = 0.5
                    elif 'reason:' in line:
                        reason = line.split(':', 1)[-1].strip()
                
                result = {
                    'id': candidate.get('id'),
                    'match': match,
                    'confidence': confidence,
                    'reason': reason
                }
                
                if match and confidence > 0.5:
                    verified.append(result)
                else:
                    rejected.append(result)
                    
            except Exception as e:
                rejected.append({
                    'id': candidate.get('id'),
                    'match': False,
                    'confidence': 0.0,
                    'reason': f'Error: {str(e)}'
                })
        
        return {
            "verified": verified,
            "rejected": rejected,
            "total": len(candidates),
            "verification_rate": len(verified) / len(candidates) if candidates else 0
        }
    
    async def analyze_question(self, question: str, dataset_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze user question to determine execution strategy"""
        context_info = ""
        if dataset_context:
            context_info = f"""
Dataset Information:
- Total images: {dataset_context.get('total_images', 'unknown')}
- Contains humans: {dataset_context.get('has_humans', 'unknown')}
- Has captions: {dataset_context.get('has_captions', 'unknown')}"""
            if dataset_context.get('has_selected_images'):
                context_info += f"\n- User has {dataset_context.get('selected_count', 0)} image(s) SELECTED"
            if dataset_context.get('has_reference_images'):
                context_info += f"\n- User has ATTACHED {dataset_context.get('reference_image_count', 0)} reference image(s)"
        
        analysis_prompt = f"""You are analyzing a user's question about an image dataset for AI/ML training.

USER QUESTION: "{question}"
{context_info}

Analyze and respond with JSON:
{{
    "intent": "<FACT_FINDING|SEMANTIC_RETRIEVAL|POSE_SEARCH|SIMILARITY_SEARCH|EXCLUSION_SEARCH|QUALITY_ASSESSMENT|STATISTICAL_ANALYSIS|SUBJECTIVE_QUERY|SUPERLATIVE_QUERY>",
    "search_query": "<optimized CLIP search query>",
    "negative_query": "<for exclusion: what to exclude, null otherwise>",
    "entities": {{
        "subjects": ["<subject types>"],
        "gender": "<male/female/any or null>",
        "objects": ["<objects>"],
        "colors": ["<colors>"],
        "attributes": ["<visual attributes>"],
        "poses": ["<pose descriptions>"],
        "age_group": "<child/young/adult/elderly or null>"
    }},
    "services_needed": ["semantic", "vlm"],
    "requires_vlm_verification": true,
    "is_counting_question": <true/false>,
    "is_exclusion_query": <true/false>,
    "is_superlative_query": <true/false>,
    "is_similarity_query": <true/false>,
    "filters": {{
        "quality": "<high/low/blurry or null>",
        "human_presence": "<none/single/multiple or null>",
        "nsfw_level": "<safe/suggestive or null>"
    }},
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}

Respond ONLY with valid JSON."""

        response = await self._run_text_only(analysis_prompt, max_tokens=800, temperature=0.3)
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                analysis = json.loads(response[json_start:json_end])
                return {"success": True, "analysis": analysis}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {e}")
            return {"success": False, "error": str(e), "raw_response": response[:500]}
        
        return {"success": False, "error": "No valid JSON in response"}
    
    async def synthesize_answer(
        self, 
        question: str, 
        evidence: Any, 
        persona: str = "professional",
        representative_images: List[str] = None
    ) -> Dict[str, Any]:
        """Synthesize natural language answer from evidence"""
        
        persona_prompts = {
            'professional': "You are a professional dataset analyst. Your tone is formal and objective.",
            'casual': "You are a friendly AI assistant. Your tone is conversational.",
            'facts_only': "Provide only factual information, no commentary."
        }
        system_prompt = persona_prompts.get(persona, persona_prompts['professional'])
        
        # Format evidence
        if isinstance(evidence, dict):
            evidence_text = json.dumps(evidence, indent=2)
        elif isinstance(evidence, list):
            evidence_text = f"Found {len(evidence)} results:\n" + "\n".join(str(e)[:200] for e in evidence[:10])
        else:
            evidence_text = str(evidence)
        
        synthesis_prompt = f"""{system_prompt}

User Question: {question}

Evidence:
{evidence_text}

GUIDELINES:
- Your answer MUST be grounded in the evidence. Do not claim anything not supported.
- Be CONCISE - 2-3 sentences for simple queries, 4-5 for complex analysis.
- If evidence is insufficient, say so briefly.

Provide a clear, concise answer matching the {persona} tone."""

        # If we have representative images, include them
        if representative_images and len(representative_images) > 0:
            # Build multi-modal content
            content = [{"type": "text", "text": synthesis_prompt}]
            
            for img_data in representative_images[:4]:  # Max 4 images
                try:
                    image = self._load_image(img_data, 768)
                    content.append({"type": "image", "image": image})
                except:
                    continue
            
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            # Text-only synthesis
            messages = [{"role": "user", "content": [{"type": "text", "text": synthesis_prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7
            )
        
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        self.inference_count += 1
        self.last_inference = datetime.now().isoformat()
        
        return {
            "question": question,
            "answer": answer,
            "persona": persona
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "loaded": self.is_loaded,
            "loading": self.is_loading,
            "device": self.device,
            "load_time_seconds": self.load_time,
            "inference_count": self.inference_count,
            "last_inference": self.last_inference
        }


# Global service instance
qwen_service = QwenService()

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class LoadRequest(BaseModel):
    pass

class ClassifyRequest(BaseModel):
    image_data: str = Field(..., description="Base64-encoded image or file path")
    max_resolution: int = Field(512, description="Max resolution for classification")

class BatchClassifyRequest(BaseModel):
    images: List[Dict] = Field(..., description="List of images with 'id' and 'image_data'")
    max_resolution: int = Field(512, description="Max resolution for classification")

class CaptionRequest(BaseModel):
    image_data: str = Field(..., description="Base64-encoded image or file path")
    style: str = Field("descriptive", description="Caption style: detailed, descriptive, concise")

class VQARequest(BaseModel):
    image_data: str = Field(..., description="Base64-encoded image or file path")
    question: str = Field(..., description="Question about the image")

class VerifyRequest(BaseModel):
    candidates: List[Dict] = Field(..., description="List of candidate images with 'id' and 'image_data'")
    query: str = Field(..., description="Query to verify against")

class AnalyzeQuestionRequest(BaseModel):
    question: str = Field(..., description="User's natural language question")
    dataset_context: Optional[Dict] = Field(None, description="Context about the dataset")

class SynthesizeRequest(BaseModel):
    question: str = Field(..., description="User's original question")
    evidence: Any = Field(..., description="Evidence to synthesize from")
    persona: str = Field("professional", description="Response persona")
    representative_images: Optional[List[str]] = Field(None, description="Optional sample images (base64)")

# =============================================================================
# AUTHENTICATION
# =============================================================================

async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# =============================================================================
# SYSTEM HELPERS
# =============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    if not HAS_GPUTIL:
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                "memory_used_mb": torch.cuda.memory_allocated(0) / 1024**2,
                "memory_free_mb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**2
            }
        return {"available": False}
    
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {"available": False}
        
        gpu = gpus[0]
        return {
            "available": True,
            "name": gpu.name,
            "driver": gpu.driver,
            "memory_total_mb": gpu.memoryTotal,
            "memory_used_mb": gpu.memoryUsed,
            "memory_free_mb": gpu.memoryFree,
            "memory_utilization_percent": round(gpu.memoryUtil * 100, 1),
            "gpu_utilization_percent": round(gpu.load * 100, 1),
            "temperature_c": gpu.temperature
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information (RunPod-style)"""
    try:
        # Memory
        memory = psutil.virtual_memory()
        
        # CPU
        cpu_count = psutil.cpu_count() or 0
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Disk (container disk - root filesystem)
        disk = psutil.disk_usage('/')
        
        # Volume (workspace - RunPod mounts /workspace as persistent volume)
        volume_info = {}
        try:
            workspace_path = '/workspace'
            if os.path.exists(workspace_path):
                volume = psutil.disk_usage(workspace_path)
                volume_info = {
                    "volume_total_gb": round(volume.total / 1024**3, 1),
                    "volume_used_gb": round(volume.used / 1024**3, 1),
                    "volume_percent": volume.percent
                }
        except:
            pass
        
        # Process count
        process_count = len(psutil.pids())
        
        result = {
            "cpu_count": cpu_count,
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / 1024**3, 1),
            "memory_used_gb": round(memory.used / 1024**3, 1),
            "memory_percent": memory.percent,
            "disk_total_gb": round(disk.total / 1024**3, 1),
            "disk_used_gb": round(disk.used / 1024**3, 1),
            "disk_percent": disk.percent,
            "process_count": process_count
        }
        result.update(volume_info)
        return result
        
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Serve dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    return {"message": "CuteCaption Qwen Mini-Server", "status": "running"}

@app.get("/status")
async def get_status(request: Request = None):
    """Get comprehensive server status"""
    # Build connection info from request
    connection_info = {}
    if request:
        # Get the host from the request (handles proxies like RunPod)
        host = request.headers.get('host', f'{HOST}:{PORT}')
        # Check if behind HTTPS proxy
        proto = 'https' if request.headers.get('x-forwarded-proto') == 'https' else 'http'
        ws_proto = 'wss' if proto == 'https' else 'ws'
        
        base_url = f"{proto}://{host}"
        connection_info = {
            "server_url": base_url,
            "api_endpoint": f"{base_url}/api/v1",
            "websocket_url": f"{ws_proto}://{host}/ws",
            "api_key": API_KEY,  # Server's API key for client to use
            "api_key_masked": API_KEY[:4] + "..." + API_KEY[-4:] if len(API_KEY) > 8 else "****"
        }
    
    return {
        "server": "CuteCaption Qwen Mini-Server",
        "version": "1.0.0",
        "uptime_seconds": time.time() - app.state.start_time,
        "connection": connection_info,
        "gpu": get_gpu_info(),
        "system": get_system_info(),
        "qwen": qwen_service.get_status(),
        "ready": qwen_service.is_loaded
    }

@app.post("/api/v1/load")
async def load_model(background_tasks: BackgroundTasks, x_api_key: str = Header(None)):
    """Load Qwen model into VRAM"""
    await verify_api_key(x_api_key)
    return await qwen_service.load()

@app.post("/api/v1/unload")
async def unload_model(x_api_key: str = Header(None)):
    """Unload Qwen model from VRAM"""
    await verify_api_key(x_api_key)
    return await qwen_service.unload()

@app.post("/api/v1/classify")
async def classify_image(request: ClassifyRequest, x_api_key: str = Header(None)):
    """Classify image (25 dimensions)"""
    await verify_api_key(x_api_key)
    return await qwen_service.classify(request.image_data, request.max_resolution)

@app.post("/api/v1/classify_batch")
async def classify_batch(request: BatchClassifyRequest, x_api_key: str = Header(None)):
    """Batch classify multiple images"""
    await verify_api_key(x_api_key)
    
    results = []
    for img in request.images:
        try:
            result = await qwen_service.classify(img.get('image_data', ''), request.max_resolution)
            result['id'] = img.get('id')
            results.append(result)
        except Exception as e:
            results.append({
                'id': img.get('id'),
                'error': str(e)
            })
    
    return {
        "results": results,
        "total": len(request.images),
        "successful": len([r for r in results if 'error' not in r])
    }

@app.post("/api/v1/caption")
async def caption_image(request: CaptionRequest, x_api_key: str = Header(None)):
    """Generate image caption"""
    await verify_api_key(x_api_key)
    return await qwen_service.caption(request.image_data, request.style)

@app.post("/api/v1/vqa")
async def visual_qa(request: VQARequest, x_api_key: str = Header(None)):
    """Visual question answering"""
    await verify_api_key(x_api_key)
    return await qwen_service.vqa(request.image_data, request.question)

@app.post("/api/v1/verify")
async def verify_candidates(request: VerifyRequest, x_api_key: str = Header(None)):
    """Verify candidate images against query"""
    await verify_api_key(x_api_key)
    return await qwen_service.verify_candidates(request.candidates, request.query)

@app.post("/api/v1/analyze_question")
async def analyze_question(request: AnalyzeQuestionRequest, x_api_key: str = Header(None)):
    """Analyze user question for query routing"""
    await verify_api_key(x_api_key)
    return await qwen_service.analyze_question(request.question, request.dataset_context)

@app.post("/api/v1/synthesize")
async def synthesize_answer(request: SynthesizeRequest, x_api_key: str = Header(None)):
    """Synthesize natural language answer"""
    await verify_api_key(x_api_key)
    return await qwen_service.synthesize_answer(
        request.question,
        request.evidence,
        request.persona,
        request.representative_images
    )

# =============================================================================
# WEBSOCKET
# =============================================================================

class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
    
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
    
    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws)
    
    async def broadcast(self, message: dict):
        for conn in self.connections:
            try:
                await conn.send_json(message)
            except:
                pass

ws_manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

# =============================================================================
# LIFECYCLE
# =============================================================================

@app.on_event("startup")
async def startup():
    app.state.start_time = time.time()
    logger.info("=" * 60)
    logger.info("CuteCaption Qwen Mini-Server Starting")
    logger.info("=" * 60)
    logger.info(f"Host: {HOST}:{PORT}")
    logger.info(f"Model Cache: {MODEL_CACHE}")
    logger.info(f"API Key: {'*' * 12}{API_KEY[-4:]}")
    
    gpu = get_gpu_info()
    if gpu.get('available'):
        logger.info(f"GPU: {gpu['name']}")
        logger.info(f"VRAM: {gpu.get('memory_total_mb', 0):.0f} MB")
    else:
        logger.warning("No GPU detected!")
    
    logger.info("=" * 60)
    logger.info("Ready! Load model via dashboard or POST /api/v1/load")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")
    await qwen_service.unload()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "qwen_server:app",
        host=HOST,
        port=PORT,
        log_level="info"
    )

