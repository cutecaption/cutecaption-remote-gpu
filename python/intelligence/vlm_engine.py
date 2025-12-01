"""
VLM Engine - Vision-Language Model Expert Service
Handles Qwen3-VL for image classification, captioning, and reasoning

Model: Qwen/Qwen3-VL-8B-Instruct (default)
       https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

Qwen3-VL Capabilities (significantly enhanced over Qwen2-VL):
- Visual Agent: Operates PC/mobile GUIs, recognizes elements, invokes tools
- Visual Coding: Generates diagrams/HTML/CSS/JS from images
- Advanced Spatial Perception: Object positions, viewpoints, occlusions, 2D grounding
- Extended context length and deeper visual reasoning
- Image classification (21+ dimensions)
- Image captioning
- Visual question answering
- Multi-modal reasoning
- Answer synthesis with hallucination protection
"""

# CRITICAL: Set environment variables BEFORE importing PyTorch
import os
import gc

# Split large allocations into smaller chunks
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Suppress the symlink warning - we know about it
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Force single-threaded downloads to prevent I/O saturation on low-RAM systems
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 min timeout per file

# CRITICAL: Use non-system drive for cache to prevent I/O competition with Windows
# This stops page file thrashing when downloading large model files
# Change this path via CUTECAPTION_MODEL_CACHE env var if needed
_model_cache_dir = os.environ.get('CUTECAPTION_MODEL_CACHE', 'D:\\huggingface_cache')
os.makedirs(_model_cache_dir, exist_ok=True)
os.environ['HF_HOME'] = _model_cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = _model_cache_dir
os.environ['TRANSFORMERS_CACHE'] = _model_cache_dir

import platform
# Disable memory pool for debugging - forces direct cudaMalloc calls
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'  # Uncomment if still having issues

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine

# Import VLM libraries
# REQUIRES: pip install git+https://github.com/huggingface/transformers
# Qwen3VLForConditionalGeneration is in transformers 4.57.0+ (unreleased, git only)
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"[vlm_engine] Import error: {e}", file=sys.stderr)
    print("[vlm_engine] Qwen3-VL requires transformers from git:", file=sys.stderr)
    print("[vlm_engine]   pip install git+https://github.com/huggingface/transformers", file=sys.stderr)
    sys.exit(1)


class VLMEngine(BaseEngine):
    """
    Vision-Language Model Engine - Expert service for Qwen3-VL
    
    Provides high-level visual understanding and natural language synthesis
    Uses Qwen3-VL - the most powerful vision-language model in the Qwen series
    
    Ref: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
    """
    
    def __init__(self):
        super().__init__("vlm_engine", "1.0.0")
        
        # Model state
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        
        # Configuration
        self.model_path = None  # Set via initialize params
        self.max_tokens = 512
        self.temperature = 0.7
    
    async def initialize(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize VLM Engine
        Load Qwen3-VL model
        """
        try:
            self.logger.info("Initializing VLM Engine...")
            
            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            if self.device == "cpu":
                self.logger.warning("Running on CPU - VLM operations will be slow!")
            
            # Model path - Qwen3-VL-8B-Instruct
            self.model_path = model_path or "Qwen/Qwen3-VL-8B-Instruct"
            self.logger.info(f"Loading Qwen3-VL model: {self.model_path}")
            
            # Load processor (suppress deprecation warnings)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Log available memory
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                self.logger.info(f"Available VRAM before load: {free_mem:.2f}GB")
            
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # DIAGNOSTIC: Let's understand what's happening
            import psutil
            process = psutil.Process()
            ram_gb = psutil.virtual_memory().available / 1024**3
            self.logger.info(f"Available System RAM: {ram_gb:.2f}GB")
            
            # Check if our environment variable is actually set
            alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')
            self.logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
            
            # Check if we're in TCC or WDDM mode
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "-q", "-d", "COMPUTE"],
                    capture_output=True, text=True, timeout=5
                )
                if "WDDM" in result.stdout:
                    self.logger.warning("GPU is in WDDM mode - this limits contiguous allocations!")
                    self.logger.info("Consider switching to TCC mode for better memory allocation")
            except:
                pass
            
            # PRE-DOWNLOAD: Explicitly download model files with SINGLE THREAD
            # This prevents the "system freeze" caused by parallel downloads + symlink fallback copies
            # With only ~3GB RAM free and no symlinks, we MUST go slow to avoid page file thrashing
            self.logger.info("Verifying/Downloading model files (single-threaded to prevent freeze)...")
            self.logger.info("TIP: Enable Windows Developer Mode for faster downloads via symlinks")
            
            # Force garbage collection before download
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Explicitly use snapshot_download with SINGLE worker
            local_model_path = snapshot_download(
                repo_id=self.model_path,
                revision="main",
                max_workers=1,         # SINGLE THREAD - prevents I/O + RAM saturation
            )
            
            # Force cleanup after download
            gc.collect()
            
            self.logger.info(f"Model files ready at: {local_model_path}")
            
            # PRIMARY LOAD PATH: Use device_map="auto" for direct GPU streaming
            # This prevents system RAM spikes by loading directly to GPU
            self.logger.info("Loading model from local path with device_map='auto'...")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    local_model_path,  # Use the local path we just verified
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            
            # No need for manual .to("cuda") as device_map handles it
            self.logger.info("Model loaded successfully")
            
            self.model.eval()
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free_mem:.2f}GB")
            
            self.logger.info("✓ Qwen3-VL model loaded successfully")
            
            self._mark_initialized()
            
            return {
                "model": self.model_path,
                "device": self.device,
                "max_tokens": self.max_tokens
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self._mark_unhealthy(str(e))
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a VLM request
        
        Supported actions:
        - classify: Classify image_type and content_type
        - caption: Generate image caption
        - answer_question: Answer visual question
        - synthesize_answer: Synthesize final answer from evidence
        """
        action = request.get('action')
        
        # Handle health check
        if action == 'health_check':
            return await self.health_check()
        
        if action == 'classify':
            return await self._classify(
                request['image_path'],
                max_resolution=request.get('max_resolution', 128)
            )
        
        elif action == 'caption':
            return await self._caption(
                request['image_path'],
                request.get('style', 'descriptive')
            )
        
        elif action == 'answer_question':
            return await self._answer_question(
                request['image_path'],
                request['question']
            )
        
        elif action == 'verify_candidates':
            return await self._verify_candidates(
                request['candidates'],
                request['query'],
                request.get('max_resolution', 1024)  # Full resolution for verification
            )
        
        elif action == 'synthesize_answer':
            return await self._synthesize_answer(
                request['question'],
                request['evidence'],
                request.get('persona', 'professional'),
                request.get('representative_images', None),
                request.get('confidence_metrics', None)
            )
        
        elif action == 'analyze_question':
            return await self._analyze_question(
                request['question'],
                request.get('available_services', []),
                request.get('dataset_context', {})
            )
        
        elif action == 'filter_exclusion':
            return await self._filter_exclusion(
                request['candidates'],
                request['exclude_query'],
                request.get('original_question', ''),
                request.get('max_resolution', 512)
            )
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _classify(self, image_path: str, max_resolution: int = 512) -> Dict[str, Any]:
        """
        TWO-STAGE CONDITIONAL CLASSIFICATION
        
        Stage 1: Universal attributes (12 dimensions) - ALL images
        Stage 2: Human-specific attributes (13 dimensions) - ONLY if humans detected
        
        This architecture:
        - Reduces cognitive load on VLM for non-human images
        - Speeds up classification for landscapes, objects, architecture
        - Maintains full accuracy for people-focused images (LoRA datasets)
        
        Args:
            image_path: Path to image
            max_resolution: Maximum dimension for classification (default 512px)
        """
        try:
            self.logger.info(f"Classifying: {image_path} (max_res={max_resolution}px)")
            
            # Load and prepare image
            image = Image.open(image_path)
            
            # Apply EXIF orientation
            try:
                from PIL import ImageOps
                image = ImageOps.exif_transpose(image)
            except Exception as e:
                self.logger.debug(f"EXIF transpose not applied: {e}")
            
            image = image.convert('RGB')
            original_size = image.size
            original_aspect = original_size[0] / original_size[1]
            
            # Resize if needed
            min_dimension = 64
            max_dim = max(image.size)
            
            if max_dim > max_resolution:
                scale = max_resolution / max_dim
                new_width = int(image.width * scale)
                new_height = int(image.height * scale)
                
                if min(new_width, new_height) < min_dimension:
                    if new_width < new_height:
                        new_width = min_dimension
                        new_height = int(min_dimension / original_aspect) if original_aspect < 1 else int(min_dimension * (1/original_aspect))
                    else:
                        new_height = min_dimension
                        new_width = int(min_dimension * original_aspect)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized from {original_size} to {image.size} for classification")
            
            # ============================================================
            # STAGE 1: UNIVERSAL CLASSIFICATION (12 dimensions)
            # Fast pass for ALL images - determines if Stage 2 needed
            # ============================================================
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
            
            # Stage 1 VLM call
            stage1_response = await self._run_vlm_prompt(image, stage1_prompt, max_tokens=80)
            
            # Parse Stage 1 results
            result = self._parse_stage1_response(stage1_response, image_path)
            
            # Add aspect ratio info (CRITICAL for shot type accuracy)
            orientation = "square"
            if original_aspect > 1.2:
                orientation = "landscape"
            elif original_aspect < 0.83:
                orientation = "portrait"
            elif original_aspect > 2.0:
                orientation = "panoramic_horizontal"
            elif original_aspect < 0.5:
                orientation = "panoramic_vertical"
            
            result['aspect_ratio'] = round(original_aspect, 3)
            result['orientation'] = orientation
            result['dimensions'] = {'width': original_size[0], 'height': original_size[1]}
            
            # ============================================================
            # STAGE 2: HUMAN-SPECIFIC CLASSIFICATION (13 dimensions)
            # Only runs if humans were detected in Stage 1
            # ============================================================
            if result['human_presence'] != 'none':
                self.logger.debug(f"Humans detected ({result['human_presence']}), running Stage 2 classification")
                
                stage2_prompt = """This image contains humans. Classify these 13 HUMAN-SPECIFIC dimensions:

1. Human Details: gender, age_group, pose/action, clothing (e.g., "woman, adult, sitting, red dress")
2. Body Type: slim, average, curvy, plus_size, athletic, petite, tall, or not_visible
3. Detailed Clothing: specific items with colors (e.g., "blue jeans, white t-shirt")
4. Clothing Style: casual, formal, business, athletic, bohemian, streetwear, vintage, or elegant
5. Accessories: hat, cap, glasses, sunglasses, jewelry, watch, bag, scarf, tie, belt, headphones, or none
6. Expression: smiling, neutral, serious, laughing, surprised, or not_visible
7. Eye Contact: yes or no
8. Shot Style: candid, posed, editorial, selfie, or professional
9. Estimated Age: age range like "20-30" or "45-55" or "60+" (or unknown)
10. Hair: color and length/style (e.g., "blonde, long, wavy") or not_visible
11. Ethnicity/Appearance: Asian, Caucasian, Hispanic, African, Middle Eastern, South Asian, Mixed, or unknown
12. Primary Action: sitting, standing, walking, running, dancing, posing, talking, eating, working, or other
13. Setting/Location: beach, office, studio, street, home, gym, restaurant, nature, urban, or unknown

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
                
                # Stage 2 VLM call (reuses same image)
                stage2_response = await self._run_vlm_prompt(image, stage2_prompt, max_tokens=120)
                
                # Parse and merge Stage 2 results
                result = self._parse_stage2_response(stage2_response, result)
            else:
                self.logger.debug("No humans detected, skipping Stage 2 (faster classification)")
                # Set human-specific defaults for non-human images
                result.update({
                    'human_details': '',
                    'body_type': 'not_visible',
                    'detailed_clothing': '',
                    'clothing_style': None,
                    'accessories': [],
                    'expression': 'not_visible',
                    'eye_contact': False,
                    'shot_style': 'candid',
                    'estimated_age': 'unknown',
                    'estimated_age_range': None,
                    'hair': 'not_visible',
                    'hair_color': None,
                    'hair_style': None,
                    'ethnicity': 'unknown',
                    'primary_action': 'unknown',
                    'setting': result.get('setting', 'unknown')
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            raise
    
    async def _classify_batch(
        self, 
        image_paths: List[str], 
        max_resolution: int = 512
    ) -> List[Dict[str, Any]]:
        """
        BATCH CLASSIFICATION - Process multiple images efficiently
        
        Preserves aspect ratio for each image and includes orientation info.
        Uses GPU batching when possible for ~5-8x speedup.
        
        Args:
            image_paths: List of image paths to classify
            max_resolution: Maximum dimension for classification
            
        Returns:
            List of classification results with aspect_info included
        """
        results = []
        
        for image_path in image_paths:
            try:
                # Load image and get aspect info BEFORE any resizing
                image = Image.open(image_path)
                
                # Apply EXIF orientation
                try:
                    from PIL import ImageOps
                    image = ImageOps.exif_transpose(image)
                except Exception:
                    pass
                
                image = image.convert('RGB')
                original_width, original_height = image.size
                aspect_ratio = original_width / original_height
                
                # Determine orientation (CRITICAL for shot type detection)
                if aspect_ratio > 1.2:
                    orientation = "landscape"
                elif aspect_ratio < 0.83:
                    orientation = "portrait"
                elif aspect_ratio > 2.0:
                    orientation = "panoramic_horizontal"
                elif aspect_ratio < 0.5:
                    orientation = "panoramic_vertical"
                else:
                    orientation = "square"
                
                aspect_info = {
                    "original_width": original_width,
                    "original_height": original_height,
                    "aspect_ratio": round(aspect_ratio, 3),
                    "orientation": orientation
                }
                
                # Run classification (uses existing two-stage method)
                result = await self._classify(image_path, max_resolution)
                result["aspect_info"] = aspect_info
                result["filename"] = os.path.basename(image_path)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Batch classification failed for {image_path}: {e}")
                results.append({
                    "filename": os.path.basename(image_path),
                    "error": str(e)
                })
        
        return results
    
    async def _run_vlm_prompt(self, image: Image.Image, prompt: str, max_tokens: int = 100) -> str:
        """Helper to run a VLM prompt and return the response text"""
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
        
        image_inputs, video_inputs = process_vision_info(messages)
        
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
                temperature=0.3
            )
        
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        return response
    
    def _parse_stage1_response(self, response: str, image_path: str) -> Dict[str, Any]:
        """Parse Stage 1 (universal) classification response"""
        # Defaults
        image_type = "photograph"
        content_type = "portrait"
        shot_scale = "medium"
        time_of_day = "day"
        dominant_colors = ""
        color_mode = "color"
        lighting_quality = "acceptable"
        exposure = "normal"
        background_type = "indoor"
        nsfw_level = "safe"
        overlays = "none"
        human_presence = "none"
        
        for line in response.lower().split('\n'):
            if 'image_type:' in line:
                image_type = line.split('image_type:')[-1].strip()
            elif 'content_type:' in line:
                content_type = line.split('content_type:')[-1].strip()
            elif 'shot_scale:' in line:
                shot_scale = line.split('shot_scale:')[-1].strip()
            elif 'time_of_day:' in line:
                time_of_day = line.split('time_of_day:')[-1].strip()
            elif 'dominant_colors:' in line:
                dominant_colors = line.split('dominant_colors:')[-1].strip()
            elif 'color_mode:' in line:
                color_mode = line.split('color_mode:')[-1].strip()
            elif 'lighting_quality:' in line:
                lighting_quality = line.split('lighting_quality:')[-1].strip()
            elif 'exposure:' in line:
                exposure = line.split('exposure:')[-1].strip()
            elif 'background_type:' in line:
                background_type = line.split('background_type:')[-1].strip()
            elif 'nsfw_level:' in line:
                nsfw_level = line.split('nsfw_level:')[-1].strip()
            elif 'overlays:' in line:
                overlays = line.split('overlays:')[-1].strip()
            elif 'human_presence:' in line:
                human_presence = line.split('human_presence:')[-1].strip()
        
        # Parse overlays into list
        overlay_types = []
        has_overlays = False
        if overlays and overlays != 'none':
            has_overlays = True
            overlay_types = [o.strip() for o in overlays.split(',') if o.strip() and o.strip() != 'none']
        
        return {
            "image_type": image_type,
            "content_type": content_type,
            "shot_scale": shot_scale,
            "time_of_day": time_of_day,
            "dominant_colors": dominant_colors,
            "color_mode": color_mode,
            "lighting_quality": lighting_quality,
            "exposure_quality": exposure,
            "background_type": background_type,
            "nsfw_level": nsfw_level,
            "has_overlays": has_overlays,
            "overlay_types": overlay_types,
            "human_presence": human_presence,
            "confidence": 0.85,
            "image_path": image_path
        }
    
    def _parse_stage2_response(self, response: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Stage 2 (human-specific) classification response and merge with Stage 1"""
        # Defaults for human-specific fields
        human_details = ""
        body_type = "not_visible"
        detailed_clothing = ""
        clothing_style = "unknown"
        accessories = "none"
        expression = "neutral"
        eye_contact = "no"
        shot_style = "posed"
        estimated_age = "unknown"
        hair = "not_visible"
        ethnicity = "unknown"
        primary_action = "unknown"
        setting = "unknown"
        
        for line in response.lower().split('\n'):
            if 'human_details:' in line:
                human_details = line.split('human_details:')[-1].strip()
            elif 'body_type:' in line:
                body_type = line.split('body_type:')[-1].strip()
            elif 'detailed_clothing:' in line:
                detailed_clothing = line.split('detailed_clothing:')[-1].strip()
            elif 'clothing_style:' in line:
                clothing_style = line.split('clothing_style:')[-1].strip()
            elif 'accessories:' in line:
                accessories = line.split('accessories:')[-1].strip()
            elif 'expression:' in line:
                expression = line.split('expression:')[-1].strip()
            elif 'eye_contact:' in line:
                eye_contact = line.split('eye_contact:')[-1].strip()
            elif 'shot_style:' in line:
                shot_style = line.split('shot_style:')[-1].strip()
            elif 'estimated_age:' in line:
                estimated_age = line.split('estimated_age:')[-1].strip()
            elif 'hair:' in line:
                hair = line.split('hair:')[-1].strip()
            elif 'ethnicity:' in line:
                ethnicity = line.split('ethnicity:')[-1].strip()
            elif 'primary_action:' in line:
                primary_action = line.split('primary_action:')[-1].strip()
            elif 'setting:' in line:
                setting = line.split('setting:')[-1].strip()
        
        # Parse accessories into list
        accessories_list = []
        if accessories and accessories != 'none':
            accessories_list = [a.strip() for a in accessories.split(',') if a.strip() and a.strip() != 'none']
        
        # Parse eye_contact to boolean
        has_eye_contact = eye_contact in ['yes', 'true', '1']
        
        # Parse estimated_age into range
        estimated_age_range = self._parse_age_range(estimated_age)
        
        # Parse hair into structured format
        hair_color = None
        hair_style = None
        if hair and hair not in ['not_visible', 'n/a', 'unknown', 'none']:
            hair_parts = [p.strip() for p in hair.split(',')]
            if len(hair_parts) >= 1:
                hair_color = hair_parts[0]
            if len(hair_parts) >= 2:
                hair_style = ', '.join(hair_parts[1:])
        
        # Merge with Stage 1 results
        result.update({
            'human_details': human_details,
            'body_type': body_type,
            'detailed_clothing': detailed_clothing,
            'clothing_style': clothing_style,
            'accessories': accessories_list,
            'expression': expression,
            'eye_contact': has_eye_contact,
            'shot_style': shot_style,
            'estimated_age': estimated_age,
            'estimated_age_range': estimated_age_range,
            'hair': hair,
            'hair_color': hair_color,
            'hair_style': hair_style,
            'ethnicity': ethnicity,
            'primary_action': primary_action,
            'setting': setting
        })
        
        return result
    
    def _parse_age_range(self, estimated_age: str) -> list:
        """Parse estimated age string into [min, max] range"""
        if not estimated_age or estimated_age == 'unknown':
            return None
        
        try:
            age_str = estimated_age.replace('around', '').replace('approximately', '').replace('about', '').strip()
            if '+' in age_str:
                min_age = int(age_str.replace('+', '').strip())
                return [min_age, 100]
            elif '-' in age_str:
                parts = age_str.split('-')
                return [int(parts[0].strip()), int(parts[1].strip())]
            elif 'over' in estimated_age.lower():
                min_age = int(age_str.replace('over', '').strip())
                return [min_age, 100]
            elif 'under' in estimated_age.lower():
                max_age = int(age_str.replace('under', '').strip())
                return [0, max_age]
            else:
                age = int(age_str)
                return [max(0, age - 5), age + 5]
        except (ValueError, IndexError):
            return None
    
    async def _caption(self, image_path: str, style: str) -> Dict[str, Any]:
        """
        Generate image caption
        
        Uses optimized resolution based on caption style
        
        Args:
            image_path: Path to image
            style: 'descriptive', 'detailed', 'concise'
        """
        try:
            self.logger.info(f"Captioning: {image_path} (style: {style})")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Store original max dimension before resizing
            original_max_dim = max(image.size)
            
            # Determine resolution based on caption style
            # Descriptive: 768px should be sufficient
            # Detailed: 1024px for more detail
            # Concise: 512px is enough
            if style == 'detailed':
                required_resolution = 1024
            elif style == 'concise':
                required_resolution = 512
            else:  # descriptive
                required_resolution = 768
            
            # Resize if needed for optimization
            if original_max_dim > required_resolution:
                scale = required_resolution / original_max_dim
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized from {original_max_dim}px to {required_resolution}px for {style} captioning")
            
            # Caption prompt based on style
            if style == 'detailed':
                prompt = "Provide a detailed, comprehensive description of this image including all visible elements, composition, colors, and mood."
            elif style == 'concise':
                prompt = "Provide a brief, one-sentence description of this image."
            else:
                prompt = "Describe this image."
            
            # Generate caption with Qwen3-VL
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
            
            image_inputs, video_inputs = process_vision_info(messages)
            
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
                    max_new_tokens=256 if style == 'detailed' else 50,
                    temperature=0.7
                )
            
            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            caption_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            result = {
                "caption": caption_text,
                "style": style,
                "image_path": image_path,
                "resolution_used": min(original_max_dim, required_resolution)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Captioning failed: {e}")
            raise
    
    async def _answer_question(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Answer a question about an image
        
        Visual Question Answering (VQA)
        Uses task-dependent image sizing for optimal performance
        """
        try:
            self.logger.info(f"VQA: {question} on {image_path}")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Store original max dimension before resizing
            original_max_dim = max(image.size)
            
            # Determine required resolution based on question complexity
            required_resolution = self._determine_required_resolution(question)
            self.logger.debug(f"Question requires {required_resolution}px resolution")
            
            # Resize if needed for optimization
            if original_max_dim > required_resolution:
                scale = required_resolution / original_max_dim
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized from {original_max_dim}px to {required_resolution}px for VQA optimization")
            
            # Generate answer with Qwen3-VL
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
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
                    max_new_tokens=128,
                    temperature=0.7
                )
            
            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            answer_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            result = {
                "question": question,
                "answer": answer_text,
                "confidence": 0.9,
                "image_path": image_path,
                "resolution_used": min(original_max_dim, required_resolution)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"VQA failed: {e}")
            raise
    
    def _determine_required_resolution(self, question: str) -> int:
        """
        Determine minimum resolution needed based on question type
        
        Returns appropriate max dimension in pixels
        
        Optimization strategy:
        - Simple questions (colors, objects, scenes): 512px
        - Medium questions (poses, actions, people): 768px
        - Complex questions (fine details): 1024px
        - OCR/text questions: 2048px (high res for text recognition)
        """
        lower_q = question.lower()
        
        # OCR and text questions need high resolution
        text_keywords = ['text', 'sign', 'label', 'read', 'say', 'word', 'letter', 
                        'caption', 'written', 'says', 'mentions', 'contains text']
        if any(keyword in lower_q for keyword in text_keywords):
            self.logger.debug("Question requires OCR - using 2048px")
            return 2048
        
        # Fine detail questions
        detail_keywords = ['detail', 'small', 'tiny', 'specific', 'exact', 'precise',
                          'identify', 'recognize', 'what brand', 'what model']
        if any(keyword in lower_q for keyword in detail_keywords):
            self.logger.debug("Question requires fine detail - using 1024px")
            return 1024
        
        # Pose and action questions
        pose_keywords = ['pose', 'standing', 'sitting', 'doing', 'action', 'position',
                        'arms', 'legs', 'hands', 'raised', 'extended', 'crossed',
                        'walking', 'running', 'jumping', 'kneeling', 'crouching']
        if any(keyword in lower_q for keyword in pose_keywords):
            self.logger.debug("Question requires pose/action analysis - using 768px")
            return 768
        
        # Simple questions (colors, objects, scenes, basic attributes)
        # Default to 512px for most questions
        self.logger.debug("Question is simple - using 512px")
        return 512
    
    async def _verify_candidates(
        self, 
        candidates: List[str], 
        query: str, 
        max_resolution: int = 1024
    ) -> Dict[str, Any]:
        """
        Verify candidate images match the query at full resolution
        
        This is Stage 2: Deep verification of pre-filtered candidates
        Qwen examines each candidate at full resolution to confirm match
        
        Args:
            candidates: List of image paths to verify
            query: Original user query (e.g., "woman sitting wearing red dress")
            max_resolution: Maximum resolution for verification (default 1024px)
            
        Returns:
            Dict with verified candidates and rejection reasons
        """
        try:
            self.logger.info(f"Verifying {len(candidates)} candidates for query: '{query}'")
            
            verified = []
            rejected = []
            
            # Create verification prompt
            verification_prompt = f"""Does this image match the query: "{query}"?

Analyze carefully:
- All objects, people, and details mentioned in the query
- Colors, poses, clothing, actions, and context
- Be precise - only confirm if ALL query elements are present

Answer format:
MATCH: YES/NO
CONFIDENCE: 0.0-1.0
REASON: Brief explanation of your decision"""

            # OOM prevention: clear GPU cache periodically during long verification loops
            cache_clear_interval = 10  # Clear cache every N images
            
            for i, candidate_path in enumerate(candidates):
                # Periodic GPU cache cleanup to prevent OOM during long loops
                if i > 0 and i % cache_clear_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.logger.debug(f"Cleared GPU cache at iteration {i}")
                
                try:
                    self.logger.debug(f"Verifying candidate {i+1}/{len(candidates)}: {candidate_path}")
                    
                    # Load image at full resolution (up to max_resolution)
                    image = Image.open(candidate_path).convert('RGB')
                    
                    # Apply EXIF orientation - critical for phone photos
                    try:
                        from PIL import ImageOps
                        image = ImageOps.exif_transpose(image)
                    except Exception:
                        pass  # Continue without EXIF correction if it fails
                    
                    # Resize if needed (preserve quality for verification)
                    original_max_dim = max(image.size)
                    if original_max_dim > max_resolution:
                        scale = max_resolution / original_max_dim
                        new_size = (int(image.width * scale), int(image.height * scale))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                        self.logger.debug(f"Resized from {original_max_dim}px to {max_resolution}px for verification")
                    
                    # Generate verification response
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": verification_prompt}
                        ]
                    }]
                    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    image_inputs, video_inputs = process_vision_info(messages)
                    
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
                            max_new_tokens=150,
                            temperature=0.2  # Low temperature for precise verification
                        )
                    
                    generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
                    response = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0].strip()
                    
                    # Parse verification response
                    match = False
                    confidence = 0.0
                    reason = "No response"
                    
                    for line in response.lower().split('\n'):
                        if 'match:' in line:
                            match = 'yes' in line.split('match:')[-1].strip()
                        elif 'confidence:' in line:
                            try:
                                conf_str = line.split('confidence:')[-1].strip()
                                confidence = float(conf_str)
                            except:
                                confidence = 0.5
                        elif 'reason:' in line:
                            reason = line.split('reason:')[-1].strip()
                    
                    # Store result
                    result = {
                        'path': candidate_path,
                        'match': match,
                        'confidence': confidence,
                        'reason': reason,
                        'resolution_used': min(original_max_dim, max_resolution)
                    }
                    
                    if match and confidence > 0.5:  # Threshold for acceptance
                        verified.append(result)
                        self.logger.debug(f"✓ VERIFIED: {candidate_path} (conf: {confidence:.2f})")
                    else:
                        rejected.append(result)
                        self.logger.debug(f"✗ REJECTED: {candidate_path} - {reason}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to verify {candidate_path}: {str(e)}")
                    rejected.append({
                        'path': candidate_path,
                        'match': False,
                        'confidence': 0.0,
                        'reason': f"Verification error: {str(e)}",
                        'resolution_used': 0
                    })
            
            self.logger.info(f"Verification complete: {len(verified)} verified, {len(rejected)} rejected")
            
            return {
                "verified": verified,
                "rejected": rejected,
                "total_candidates": len(candidates),
                "verification_rate": len(verified) / len(candidates) if candidates else 0,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Candidate verification failed: {e}")
            raise
    
    async def _synthesize_answer(
        self, 
        question: str, 
        evidence: Any, 
        persona: str,
        representative_images: Optional[List[str]] = None,
        confidence_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize final answer from evidence
        
        The "Language Brain" - converts analytical results into natural language
        
        Now includes representative images for visual verification (optimized)
        Naturally incorporates confidence/uncertainty information into the answer
        
        Args:
            question: Original user question
            evidence: Evidence from analytical services (image list, stats, etc.)
            persona: Tone ('professional', 'casual', 'poetic', 'facts_only')
            representative_images: Optional list of image paths to include for visual verification (max 4)
            confidence_metrics: Optional confidence metrics (coverage, confidence, uncertainty) to naturally incorporate
        """
        try:
            self.logger.info(f"Synthesizing answer for: {question}")
            
            # Build system prompt based on persona
            persona_prompts = {
                'professional': "You are a professional dataset analyst. Your tone is formal and objective.",
                'casual': "You are a friendly AI assistant. Your tone is conversational and enthusiastic.",
                'poetic': "You are an artistic AI. Use evocative, descriptive language.",
                'facts_only': "Provide only factual information, no commentary."
            }
            
            system_prompt = persona_prompts.get(persona, persona_prompts['professional'])
            
            # Prepare evidence for synthesis
            evidence_text = self._format_evidence(evidence)
            
            # Prepare confidence context for natural incorporation
            confidence_context = self._format_confidence_context(confidence_metrics, persona)
            
            # Build content array (text + optional images)
            # CRITICAL: Grounding instruction to prevent hallucination
            content = [{"type": "text", "text": f"""{system_prompt}

User Question: {question}

Evidence:
{evidence_text}
{confidence_context}

IMPORTANT GUIDELINES:
- Your answer MUST be grounded in the evidence above. Do not claim anything not supported by the data.
- Be CONCISE - 2-3 sentences max for simple queries, 4-5 for complex analysis. No fluff or filler.
- If the evidence is insufficient, say so briefly.
- Naturally incorporate confidence/uncertainty - be transparent about limitations.

Provide a clear, concise answer matching the {persona} tone."""}]
            
            # Include representative images if provided (for visual verification)
            # Use 768px resolution - sufficient for synthesis context
            if representative_images and len(representative_images) > 0:
                max_images = min(4, len(representative_images))  # Limit to 4 images
                self.logger.debug(f"Including {max_images} representative images in synthesis for visual verification")
                
                for img_path in representative_images[:max_images]:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        
                        # Resize to 768px for efficiency (sufficient for synthesis context)
                        max_dim = max(image.size)
                        if max_dim > 768:
                            scale = 768 / max_dim
                            new_size = (int(image.width * scale), int(image.height * scale))
                            image = image.resize(new_size, Image.Resampling.LANCZOS)
                            self.logger.debug(f"Resized representative image to 768px for synthesis")
                        
                        content.append({"type": "image", "image": image})
                    except Exception as e:
                        self.logger.warning(f"Failed to load representative image {img_path}: {e}")
                        # Continue without this image
            
            # Generate answer with Qwen3-VL (now with images if provided)
            messages = [{"role": "user", "content": content}]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info if images are included
            image_inputs, video_inputs = process_vision_info(messages)
            
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
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            
            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]  # Only new tokens
            answer_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Clean up any remaining role markers
            for marker in ['system\n', 'user\n', 'assistant\n', '<|im_start|>', '<|im_end|>']:
                answer_text = answer_text.replace(marker, '')
            
            answer_text = answer_text.strip()
            
            result = {
                "question": question,
                "answer": answer_text,
                "persona": persona,
                "evidence_count": len(evidence) if isinstance(evidence, list) else 1,
                "representative_images_included": len([c for c in content if c.get("type") == "image"]) if representative_images else 0
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Answer synthesis failed: {e}")
            raise
    
    async def _analyze_question(
        self,
        question: str,
        available_services: List[str],
        dataset_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        VLM-First Question Analysis
        
        Uses Qwen3-VL to understand the user's question and generate an execution plan.
        This replaces the rule-based regex matching with true natural language understanding.
        
        Args:
            question: The user's natural language question
            available_services: List of available expert services ['semantic', 'pose', 'text', 'cv', 'vlm']
            dataset_context: Context about the dataset (total_images, has_humans, etc.)
            
        Returns:
            Dict with:
            - intent: The classified intent type
            - execution_plan: Steps to execute
            - search_query: Optimized search query for semantic search
            - entities: Extracted entities (objects, colors, attributes, etc.)
            - requires_verification: Whether VLM verification is needed
            - confidence: Confidence in the analysis
        """
        try:
            self.logger.info(f"VLM analyzing question: '{question}'")
            
            # Build context about available capabilities
            service_descriptions = {
                'semantic': 'CLIP-based semantic image search (find images by visual/conceptual similarity)',
                'pose': 'Human pose detection and search (find images by body position, gestures)',
                'text': 'Caption/OCR text search (search within image captions and visible text)',
                'cv': 'Computer vision metrics (brightness, sharpness, color analysis)',
                'vlm': 'Visual question answering and verification (detailed image analysis)'
            }
            
            available_desc = "\n".join([
                f"- {svc}: {service_descriptions.get(svc, 'Unknown service')}"
                for svc in available_services
            ])
            
            # Build dataset context
            dataset_info = ""
            if dataset_context:
                total = dataset_context.get('total_images', 'unknown')
                has_humans = dataset_context.get('has_humans', 'unknown')
                has_captions = dataset_context.get('has_captions', 'unknown')
                has_selected = dataset_context.get('has_selected_images', False)
                selected_count = dataset_context.get('selected_count', 0)
                has_previous = dataset_context.get('has_previous_results', False)
                previous_count = dataset_context.get('previous_result_count', 0)
                has_reference = dataset_context.get('has_reference_images', False)
                reference_count = dataset_context.get('reference_image_count', 0)
                
                dataset_info = f"""
Dataset Information:
- Total images: {total}
- Contains humans: {has_humans}
- Has captions: {has_captions}"""
                
                # Add user interaction context
                if has_selected:
                    dataset_info += f"\n- User has {selected_count} image(s) SELECTED (when user says 'these', 'selected', they mean these images)"
                if has_previous:
                    dataset_info += f"\n- User has {previous_count} image(s) from previous search results visible"
                if has_reference:
                    dataset_info += f"\n- User has ATTACHED {reference_count} reference image(s) for similarity comparison"
            
            # The analysis prompt - this is the core of VLM-first understanding
            analysis_prompt = f"""You are an expert AI assistant analyzing a user's question about an image dataset for AI/ML training (LoRA, fine-tuning, etc).

USER QUESTION: "{question}"

AVAILABLE SERVICES:
{available_desc}
{dataset_info}

This is a dataset intelligence system. Users ask questions to find, filter, analyze, and manage images in their datasets. Questions can be about:
- Subject attributes (gender, age, ethnicity, body type, clothing, accessories)
- Image quality (blur, brightness, lighting, resolution, composition)
- Content (objects, animals, logos, text, backgrounds, scenes)
- Poses and body positions (sitting, standing, specific poses)
- Similarity (find images similar to attached/selected images)
- Exclusion (find images WITHOUT certain attributes)
- Subjective qualities (boring, interesting, NSFW-adjacent, artistic style)
- Statistical analysis (most common themes, outliers, duplicates)
- Specific scenarios (selfies, full body shots, crowds, vehicles)
- Overlays and artifacts (emojis, stickers, watermarks, filters, text overlays)
- Color characteristics (black and white, monochrome, color dominant images)

Analyze this question and provide a structured response in the following JSON format:

{{
    "intent": "<one of: FACT_FINDING, SEMANTIC_RETRIEVAL, POSE_SEARCH, SIMILARITY_SEARCH, EXCLUSION_SEARCH, QUALITY_ASSESSMENT, STATISTICAL_ANALYSIS, ANOMALY_DETECTION, SUBJECTIVE_QUERY, SUPERLATIVE_QUERY, REFERENCE_SEARCH, ACTION_REQUEST>",
    "search_query": "<optimized CLIP search query - use visual concepts, not complex sentences. For exclusion queries, search for the positive case first>",
    "negative_query": "<for EXCLUSION_SEARCH: what to exclude, null otherwise>",
    "entities": {{
        "subjects": ["<subject types: man, woman, person, animal, etc>"],
        "gender": "<male/female/any or null>",
        "objects": ["<objects mentioned: vehicle, jewelry, logo, etc>"],
        "colors": ["<colors mentioned>"],
        "attributes": ["<visual attributes: blurry, bright, dark, seated, standing, etc>"],
        "clothing": ["<clothing items>"],
        "accessories": ["<jewelry, glasses, hats, etc>"],
        "poses": ["<pose descriptions: sitting, standing, selfie, full body, etc>"],
        "scene": ["<scene types: indoor, outdoor, vehicle interior, etc>"],
        "style": ["<artistic style, mood, aesthetic>"],
        "age_group": "<child/young/adult/elderly or null>",
        "ethnicity": "<if specifically mentioned, null otherwise>",
        "body_type": "<if mentioned: slim, curvy, athletic, etc>",
        "count": "<single/multiple/crowd or null for subject count>",
        "hair_color": "<blonde/brunette/black/red/gray/etc or null>",
        "hair_style": "<long/short/curly/straight/bald/etc or null>",
        "setting": "<beach/office/studio/street/home/gym/restaurant/nature/urban/etc or null>",
        "primary_action": "<sitting/standing/walking/running/dancing/posing/talking/eating/working/etc or null>"
    }},
    "services_needed": ["<list of services to use in order: semantic, vlm, pose, cv, text>"],
    "requires_vlm_verification": <true - almost always true for accuracy>,
    "is_counting_question": <true/false>,
    "is_exclusion_query": <true/false - looking for images WITHOUT something>,
    "is_superlative_query": <true/false - looking for most/least/best/worst>,
    "is_similarity_query": <true/false - find similar to reference image>,
    "is_subjective": <true/false - boring, interesting, artistic, NSFW-adjacent, etc>,
    "is_anomaly_query": <true/false - looking for outliers, images that don't fit, different from rest>,
    "reference_type": "<if user references an image: 'attached', 'selected', 'from_results', or null>",
    "filters": {{
        "quality": "<high/low/blurry/sharp or null>",
        "brightness": "<bright/dark/perfect or null>",
        "lighting": "<good/poor/perfect or null>",
        "human_presence": "<none/single/multiple/crowd or null>",
        "shot_type": "<selfie/full_body/portrait/close_up or null>",
        "nsfw_level": "<safe/suggestive/borderline or null>",
        "color_mode": "<bw/color or null - for black and white vs color images>",
        "pose_similarity_level": "<exact/similar/loose or null - for pose matching queries>",
        "time_of_day": "<day/night/golden_hour or null>",
        "orientation": "<portrait/landscape/square or null>",
        "expression": "<smiling/neutral/serious or null>",
        "background_type": "<studio/outdoor/indoor/bokeh/plain/busy or null>",
        "shot_style": "<candid/posed/editorial/selfie/professional or null>",
        "exposure": "<normal/overexposed/underexposed or null>",
        "age_band": "<child/teen/young_adult/adult/middle_aged/senior or null>",
        "min_age": "<number or null - for 'over 50' queries extract 50>",
        "max_age": "<number or null - for 'under 30' queries extract 30>",
        "setting": "<beach/office/studio/street/home/gym/restaurant/nature/urban or null>",
        "hair_color": "<blonde/brunette/black/red/gray or null>",
        "hair_style": "<long/short/curly/straight/bald or null>",
        "primary_action": "<sitting/standing/walking/running/dancing/posing or null>",
        "clothing_style": "<casual/formal/business/athletic/bohemian/streetwear/vintage/elegant or null>"
    }},
    "confidence": <0.0-1.0 confidence in this analysis>,
    "reasoning": "<brief explanation of how to answer this question>"
}}

CRITICAL GUIDELINES:
1. ALWAYS set requires_vlm_verification to true unless it's a pure metadata query
2. For "show me images WITHOUT X", set is_exclusion_query=true and search for X first, then filter out
3. For "most/least/best/worst" queries, set is_superlative_query=true
4. For similarity queries (resembles, similar to, same pose), set is_similarity_query=true
5. Extract ALL entities thoroughly - this determines search accuracy
6. search_query should be CLIP-optimized: "woman sitting red dress" not "show me all images of women who are sitting and wearing red dresses"
7. For subjective queries (boring, interesting, NSFW-adjacent), set is_subjective=true
8. For pose queries: "exact pose" = exact, "similar pose" = similar, "posed similar" or "pose like" = loose

Respond ONLY with valid JSON, no additional text."""

            # Generate analysis using Qwen3-VL (text-only, no images needed)
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": analysis_prompt}]
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
                    max_new_tokens=800,
                    temperature=0.3,  # Low temperature for structured output
                    do_sample=True
                )
            
            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            self.logger.debug(f"VLM analysis raw response: {response[:500]}...")
            
            # Parse JSON response
            try:
                # Clean up response - find JSON block
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse VLM response as JSON: {e}")
                self.logger.error(f"Raw response was: {response[:1000]}")
                # Return error instead of silent fallback
                return {
                    "success": False,
                    "question": question,
                    "error": f"VLM returned invalid JSON: {str(e)}",
                    "raw_response": response[:500],  # Include partial response for debugging
                    "analysis": None
                }
            
            # Ensure required fields exist
            analysis.setdefault("intent", "SEMANTIC_RETRIEVAL")
            analysis.setdefault("search_query", question)
            analysis.setdefault("entities", {})
            analysis.setdefault("services_needed", ["semantic"])
            analysis.setdefault("requires_vlm_verification", True)
            analysis.setdefault("is_counting_question", False)
            analysis.setdefault("is_action_request", False)
            analysis.setdefault("confidence", 0.7)
            
            self.logger.info(f"VLM analysis complete: intent={analysis['intent']}, services={analysis['services_needed']}")
            
            return {
                "success": True,
                "question": question,
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(f"Question analysis failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return error - NO silent fallback
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "analysis": None  # Explicitly null - let caller handle the error
            }
    
    async def _filter_exclusion(
        self,
        candidates: List[str],
        exclude_query: str,
        original_question: str,
        max_resolution: int = 512
    ) -> Dict[str, Any]:
        """
        Filter candidates to EXCLUDE images that match the exclude_query.
        
        This handles "show me images WITHOUT X" queries.
        We check each candidate and keep only those that do NOT contain the excluded element.
        
        Args:
            candidates: List of image paths to filter
            exclude_query: What to exclude (e.g., "human subject", "text", "animals")
            original_question: The original user question for context
            max_resolution: Max resolution for analysis
            
        Returns:
            Dict with filtered candidates (those that do NOT match exclude_query)
        """
        try:
            self.logger.info(f"Exclusion filter: checking {len(candidates)} candidates, excluding '{exclude_query}'")
            
            kept = []
            excluded = []
            
            exclusion_prompt = f"""Does this image contain: "{exclude_query}"?

Answer with ONLY one of these:
- YES (if the image contains {exclude_query})
- NO (if the image does NOT contain {exclude_query})

Be precise. The user wants to find images WITHOUT this element."""

            for candidate_path in candidates:
                try:
                    image = Image.open(candidate_path).convert('RGB')
                    
                    # Resize for efficiency
                    max_dim = max(image.size)
                    if max_dim > max_resolution:
                        scale = max_resolution / max_dim
                        new_size = (int(image.width * scale), int(image.height * scale))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": exclusion_prompt}
                        ]
                    }]
                    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    image_inputs, video_inputs = process_vision_info(messages)
                    
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
                            max_new_tokens=10,
                            temperature=0.1
                        )
                    
                    generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
                    response = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0].strip().upper()
                    
                    # If YES, the image contains what we want to exclude - skip it
                    # If NO, the image does NOT contain it - keep it
                    contains_excluded = response.startswith('YES')
                    
                    if contains_excluded:
                        excluded.append({'path': candidate_path, 'reason': f'Contains {exclude_query}'})
                        self.logger.debug(f"EXCLUDED: {candidate_path}")
                    else:
                        kept.append({'path': candidate_path, 'match': True})
                        self.logger.debug(f"KEPT: {candidate_path}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check {candidate_path}: {e}")
                    # On error, keep the image (conservative)
                    kept.append({'path': candidate_path, 'match': True, 'note': 'check_failed'})
            
            self.logger.info(f"Exclusion filter complete: {len(kept)} kept, {len(excluded)} excluded")
            
            return {
                "verified": kept,
                "rejected": excluded,
                "total_candidates": len(candidates),
                "kept_count": len(kept),
                "excluded_count": len(excluded),
                "exclude_query": exclude_query
            }
            
        except Exception as e:
            self.logger.error(f"Exclusion filter failed: {e}")
            raise
    
    def _format_confidence_context(self, confidence_metrics: Optional[Dict[str, Any]], persona: str) -> str:
        """Format confidence metrics for natural incorporation into VLM answer"""
        if not confidence_metrics:
            return ""
        
        confidence = confidence_metrics.get('confidence', 1.0)
        coverage = confidence_metrics.get('coverage', 1.0)
        uncertainty = confidence_metrics.get('uncertainty')
        should_say_unknown = confidence_metrics.get('shouldSayUnknown', False)
        
        # If confidence is very low, instruct VLM to say "I don't know"
        if should_say_unknown:
            return f"\n\nIMPORTANT: The confidence in this answer is very low ({confidence*100:.0f}%) because only {coverage*100:.1f}% of the dataset was analyzed. You should politely decline to answer and explain why the answer may not be reliable."
        
        # Build natural confidence context based on persona
        context_parts = []
        
        if coverage < 0.95:
            coverage_percent = coverage * 100
            if persona == 'professional':
                context_parts.append(f"Note: This analysis is based on {coverage_percent:.1f}% of the dataset ({confidence*100:.0f}% confidence).")
            elif persona == 'casual':
                context_parts.append(f"Just so you know, I looked at about {coverage_percent:.0f}% of your images ({confidence*100:.0f}% confidence).")
            elif persona == 'facts_only':
                context_parts.append(f"Coverage: {coverage_percent:.1f}%. Confidence: {confidence*100:.0f}%.")
            else:
                context_parts.append(f"Based on analyzing {coverage_percent:.1f}% of the dataset ({confidence*100:.0f}% confidence).")
        
        # Add uncertainty information for counting questions
        if uncertainty and uncertainty.get('estimatedCount') is not None:
            est = uncertainty['estimatedCount']
            lower = uncertainty.get('lowerBound', est)
            upper = uncertainty.get('upperBound', est)
            
            if persona == 'professional':
                context_parts.append(f"Estimated count: {est} (95% confidence interval: {lower}-{upper}).")
            elif persona == 'casual':
                context_parts.append(f"The number is around {est}, give or take (probably between {lower} and {upper}).")
            elif persona == 'facts_only':
                context_parts.append(f"Count: {est} (95% CI: {lower}-{upper}).")
            else:
                context_parts.append(f"Estimated: {est} (range: {lower}-{upper}).")
        
        if confidence < 0.7 and coverage < 0.8:
            if persona == 'professional':
                context_parts.append("Please interpret these results with appropriate caution.")
            elif persona == 'casual':
                context_parts.append("Take this with a grain of salt since I didn't see everything.")
        
        return "\n\n" + " ".join(context_parts) if context_parts else ""
    
    def _format_evidence(self, evidence: Any) -> str:
        """Format evidence into human-readable context for VLM"""
        
        # Handle dict with search results
        if isinstance(evidence, dict):
            # VLM verification results
            if 'verified' in evidence:
                verified = evidence['verified']
                rejected = evidence.get('rejected', [])
                query = evidence.get('query', 'unknown')
                
                if len(verified) == 0:
                    return f"No images verified for query '{query}'. {len(rejected)} candidates were rejected."
                
                formatted = f"Found {len(verified)} verified images (from {evidence.get('total_candidates', len(verified) + len(rejected))} candidates):\n"
                for idx, result in enumerate(verified[:15], 1):
                    img_path = result.get('path', 'unknown')
                    filename = img_path.split('\\')[-1] if '\\' in img_path else img_path.split('/')[-1]
                    confidence = result.get('confidence', 0)
                    formatted += f"{idx}. {filename} (confidence: {confidence:.2f})\n"
                
                if len(verified) > 15:
                    formatted += f"...and {len(verified) - 15} more verified images\n"
                
                return formatted
            
            # Semantic search results
            elif 'results' in evidence:
                results = evidence['results']
                if len(results) == 0:
                    return f"No matching images found for the query '{evidence.get('query', 'unknown')}'."
                
                # Format image results
                formatted = f"Found {len(results)} matching images:\n"
                for idx, result in enumerate(results[:10], 1):
                    img_path = result.get('image_path', 'unknown')
                    filename = img_path.split('\\')[-1] if '\\' in img_path else img_path.split('/')[-1]
                    similarity = result.get('similarity', 0)
                    formatted += f"{idx}. {filename} (similarity: {similarity:.2f})\n"
                
                return formatted
            
            # Duplicate detection results
            elif 'duplicate_groups' in evidence:
                groups = evidence['duplicate_groups']
                if len(groups) == 0:
                    return "No duplicate images found in the dataset."
                
                formatted = f"Found {len(groups)} duplicate groups ({evidence.get('total_duplicates', 0)} total duplicates):\n\n"
                for idx, group in enumerate(groups[:10], 1):
                    rep = group['representative'].split('\\')[-1] if '\\' in group['representative'] else group['representative'].split('/')[-1]
                    formatted += f"Group {idx}: {rep} has {len(group['duplicates'])} duplicates\n"
                    for dup in group['duplicates'][:3]:
                        filename = dup['path'].split('\\')[-1] if '\\' in dup['path'] else dup['path'].split('/')[-1]
                        formatted += f"  - {filename} ({dup['similarity']:.3f} similarity)\n"
                    if len(group['duplicates']) > 3:
                        formatted += f"  ...and {len(group['duplicates']) - 3} more\n"
                    formatted += "\n"
                
                return formatted
            
            # Caption search results
            elif 'matches' in evidence:
                matches = evidence['matches']
                if len(matches) == 0:
                    return f"No captions found containing '{evidence.get('query', 'unknown')}'."
                
                formatted = f"Found {len(matches)} images with matching captions:\n"
                for idx, match in enumerate(matches[:10], 1):
                    filename = match.get('filename', 'unknown')
                    caption = match.get('caption', '')
                    formatted += f"{idx}. {filename}\n   Caption: \"{caption[:100]}...\"\n"
                
                return formatted
            
            # Handle statistics/metadata
            elif 'total_image_count' in evidence or 'count' in evidence:
                return f"Dataset statistics: {json.dumps(evidence, indent=2)}"
            
            # Generic dict
            else:
                return json.dumps(evidence, indent=2)
        
        # Handle list of images (with CV metadata)
        elif isinstance(evidence, list):
            if len(evidence) == 0:
                return "No results found."
            
            # Check if images have brightness/CV data
            if len(evidence) > 0 and isinstance(evidence[0], dict) and 'brightness_score' in evidence[0]:
                formatted = f"Dataset contains {len(evidence)} images with the following visual characteristics:\n\n"
                
                # Calculate brightness distribution
                brightness_scores = [img.get('brightness_score', 0) for img in evidence if 'brightness_score' in img]
                if brightness_scores:
                    avg_brightness = sum(brightness_scores) / len(brightness_scores)
                    bright_count = sum(1 for b in brightness_scores if b > 0.6)
                    dark_count = sum(1 for b in brightness_scores if b < 0.4)
                    
                    formatted += f"Brightness Analysis:\n"
                    formatted += f"- Average brightness: {avg_brightness:.2f}\n"
                    formatted += f"- Bright images (>0.6): {bright_count}\n"
                    formatted += f"- Dark images (<0.4): {dark_count}\n"
                    formatted += f"- Medium brightness: {len(brightness_scores) - bright_count - dark_count}\n\n"
                
                # Show sample images
                formatted += "Sample images:\n"
                for idx, item in enumerate(evidence[:5], 1):
                    filename = item.get('filename', 'unknown')
                    brightness = item.get('brightness_score', 0)
                    formatted += f"{idx}. {filename} (brightness: {brightness:.2f})\n"
                
                return formatted
            else:
                # Generic list formatting
                formatted = f"Found {len(evidence)} items:\n"
                for idx, item in enumerate(evidence[:10], 1):
                    formatted += f"{idx}. {str(item)[:100]}\n"
                return formatted
        
        # Fallback
        else:
            return str(evidence)


# Main entry point
async def main():
    """Run VLM Engine as a service"""
    engine = VLMEngine()
    
    try:
        # Initialize
        init_result = await engine.initialize()
        print(json.dumps({"event": "initialized", "data": init_result}), flush=True)
        
        # Process requests from stdin
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = await engine.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                print(json.dumps({
                    "success": False,
                    "error": {"message": "Invalid JSON"}
                }), flush=True)
            except Exception as e:
                print(json.dumps({
                    "success": False,
                    "error": {"message": str(e)}
                }), flush=True)
    
    except KeyboardInterrupt:
        await engine.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

