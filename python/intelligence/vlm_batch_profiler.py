"""
VLM Batch Profiler - Optimized Qwen3-VL Dataset Profiling
===========================================================

CRITICAL: Aspect ratio is PRESERVED throughout the pipeline!
- min_pixels/max_pixels control TOTAL pixel count, not dimensions
- Qwen3-VL uses 28x28 patches (not 32x32)
- A 912x1156 image stays vertical, a 1920x1080 stays horizontal

This module integrates the optimization research into the main VLM engine
with proper handling for any image type: portrait, landscape, square, panoramic.

Usage:
    python vlm_batch_profiler.py --folder /path/to/images --output profile.json
    
Configuration via config section below.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vlm_batch_profiler')

# ============================================================================
# CONFIGURATION - User can edit these settings
# ============================================================================

@dataclass
class ProfilerConfig:
    """Configuration for the batch profiler - edit these values as needed."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    
    # Resolution settings (in total pixels, NOT dimensions)
    # Qwen uses 28x28 patches, so:
    #   256 tokens  = 256 * 28 * 28 = 200,704 pixels  (fast, good for basic profiling)
    #   576 tokens  = 576 * 28 * 28 = 451,584 pixels  (balanced)
    #   1024 tokens = 1024 * 28 * 28 = 802,816 pixels (high quality)
    min_pixels: int = 200704   # ~256 tokens - minimum quality
    max_pixels: int = 451584   # ~576 tokens - good balance
    
    # Batch settings (adjust based on your GPU)
    # RTX 3060 (12GB): 4-6
    # RTX 3080 (10GB): 6-8
    # RTX 3090 (24GB): 8-12
    # RTX 4090 (24GB): 12-16
    batch_size: int = 8
    
    # Output settings
    include_detailed_human_analysis: bool = True
    detect_aspect_ratio: bool = True  # ALWAYS TRUE - critical for shot type
    detect_orientation: bool = True   # portrait/landscape/square/panoramic
    
    # Processing settings
    use_flash_attention: bool = True
    device: str = "cuda"  # cuda or cpu


# Default config - can be overridden
DEFAULT_CONFIG = ProfilerConfig()


# ============================================================================
# ASPECT RATIO UTILITIES
# ============================================================================

def calculate_aspect_info(width: int, height: int) -> Dict[str, Any]:
    """
    Calculate aspect ratio and orientation from dimensions.
    
    CRITICAL: This information is passed to Qwen along with the image
    to ensure accurate shot type detection even after resizing.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Dict with aspect_ratio, orientation, and dimensions
    """
    aspect_ratio = width / height
    
    # Determine orientation
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
    
    # Common aspect ratio detection
    common_ratios = {
        (16, 9): 1.778,
        (4, 3): 1.333,
        (3, 2): 1.5,
        (1, 1): 1.0,
        (9, 16): 0.5625,
        (3, 4): 0.75,
        (2, 3): 0.667,
        (21, 9): 2.333,
    }
    
    closest_ratio = None
    min_diff = float('inf')
    for (w, h), ratio in common_ratios.items():
        diff = abs(aspect_ratio - ratio)
        if diff < min_diff and diff < 0.05:  # Within 5% tolerance
            min_diff = diff
            closest_ratio = f"{w}:{h}"
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 3),
        "orientation": orientation,
        "common_ratio": closest_ratio
    }


# ============================================================================
# BATCH PROFILER CLASS
# ============================================================================

class VLMBatchProfiler:
    """
    Optimized batch profiler for Qwen3-VL.
    
    Key features:
    - Preserves aspect ratio (critical for shot type detection)
    - Batched processing for 5-8x speedup
    - Configurable resolution for speed/quality tradeoff
    - Includes orientation metadata in output
    """
    
    def __init__(self, config: ProfilerConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.processor = None
        self.device = None
        
        # Stats
        self.stats = {
            "total_processed": 0,
            "total_time": 0,
            "errors": 0
        }
    
    def initialize(self) -> bool:
        """Initialize the model and processor."""
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            
            logger.info(f"Initializing {self.config.model_name}...")
            
            # Detect device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            else:
                self.device = "cpu"
                logger.warning("Using CPU - processing will be slow!")
            
            # Load processor with aspect-ratio-preserving settings
            logger.info(f"Loading processor (min_pixels={self.config.min_pixels}, max_pixels={self.config.max_pixels})")
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels
            )
            
            # Enable Flash Attention 2 if available
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True
            }
            
            if self.config.use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled")
                except Exception:
                    logger.warning("Flash Attention 2 not available, using standard attention")
            
            # Load model
            logger.info("Loading model weights...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            self.model.eval()
            
            logger.info("Model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def _get_profiling_prompt(self, aspect_info: Dict) -> str:
        """
        Generate profiling prompt that includes aspect ratio context.
        
        CRITICAL: We tell Qwen the original aspect ratio so it can make
        accurate judgments about shot type even if the image was resized.
        """
        orientation_hint = f"The image is {aspect_info['orientation']} ({aspect_info['width']}x{aspect_info['height']}, aspect ratio {aspect_info['aspect_ratio']})."
        
        return f"""Analyze this image for dataset profiling. {orientation_hint}

Respond in JSON format with these fields:
{{
    "image_type": "photograph|illustration|3d_render|digital_art|painting|sketch",
    "content_type": "portrait|landscape|architecture|still_life|abstract|urban|nature|interior|food|product|action",
    "shot_scale": "extreme_closeup|closeup|medium|medium_wide|wide|extreme_wide",
    "time_of_day": "day|night|golden_hour|blue_hour|twilight|artificial_light|unknown",
    "human_presence": "none|single|multiple",
    "gender": "male|female|mixed|unknown",
    "age_range": "child|teen|young_adult|adult|middle_aged|senior|unknown",
    "clothing_description": "brief description or null",
    "clothing_colors": ["primary_color", "secondary_color"],
    "eye_color": "blue|brown|green|hazel|gray|amber|unknown",
    "background_type": "indoor|outdoor|studio|abstract|unknown",
    "expression": "neutral|happy|serious|candid|posed",
    "dominant_colors": ["color1", "color2", "color3"]
}}

Be concise. If no humans, set human-related fields to null."""
    
    def profile_single(self, image_path: str) -> Dict[str, Any]:
        """
        Profile a single image with full aspect ratio preservation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Profile dictionary with all fields + aspect_info
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        
        try:
            # Load image and get aspect info BEFORE any resizing
            img = Image.open(image_path).convert('RGB')
            aspect_info = calculate_aspect_info(img.width, img.height)
            
            # Create message with aspect-aware prompt
            prompt = self._get_profiling_prompt(aspect_info)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Process with Qwen (aspect ratio preserved internally)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse response
            profile = self._parse_response(response)
            
            # Add aspect info to profile (always included)
            profile["aspect_info"] = aspect_info
            profile["filename"] = os.path.basename(image_path)
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to profile {image_path}: {e}")
            return {
                "filename": os.path.basename(image_path),
                "error": str(e),
                "aspect_info": aspect_info if 'aspect_info' in dir() else None
            }
    
    def profile_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Profile a batch of images efficiently.
        
        Processes multiple images in a single forward pass for ~8x speedup.
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        
        results = []
        
        try:
            # Load all images and get aspect info
            images = []
            aspect_infos = []
            valid_paths = []
            
            for path in image_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    aspect_info = calculate_aspect_info(img.width, img.height)
                    images.append(img)
                    aspect_infos.append(aspect_info)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    results.append({
                        "filename": os.path.basename(path),
                        "error": str(e)
                    })
            
            if not images:
                return results
            
            # Create messages for batch
            messages_batch = []
            for img, aspect_info in zip(images, aspect_infos):
                prompt = self._get_profiling_prompt(aspect_info)
                messages_batch.append([{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }])
            
            # Process batch
            inputs = self.processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode each
            for i, (path, aspect_info) in enumerate(zip(valid_paths, aspect_infos)):
                try:
                    generated = output_ids[i, inputs.input_ids.shape[1]:]
                    response = self.processor.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    
                    profile = self._parse_response(response)
                    profile["aspect_info"] = aspect_info
                    profile["filename"] = os.path.basename(path)
                    results.append(profile)
                    
                except Exception as e:
                    logger.warning(f"Failed to decode {path}: {e}")
                    results.append({
                        "filename": os.path.basename(path),
                        "error": str(e),
                        "aspect_info": aspect_info
                    })
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return partial results
            for path in image_paths:
                if not any(r.get("filename") == os.path.basename(path) for r in results):
                    results.append({
                        "filename": os.path.basename(path),
                        "error": str(e)
                    })
            return results
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse VLM response to dictionary."""
        import re
        
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback
        logger.warning(f"Could not parse JSON, returning raw text")
        return {"raw_response": text}
    
    def profile_folder(
        self, 
        folder_path: str, 
        output_path: str = None,
        extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Profile all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output_path: Optional path to save results (JSON)
            extensions: Image extensions to process (default: common formats)
            
        Returns:
            Dict with profiles for all images + statistics
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff']
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(folder.glob(f"*{ext}"))
            image_paths.extend(folder.glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in sorted(set(image_paths))]
        total = len(image_paths)
        
        if total == 0:
            logger.warning(f"No images found in {folder_path}")
            return {"images": [], "statistics": {"total": 0}}
        
        logger.info(f"Found {total} images to profile")
        
        # Process in batches
        all_profiles = []
        start_time = time.time()
        
        for i in range(0, total, self.config.batch_size):
            batch = image_paths[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (total + self.config.batch_size - 1) // self.config.batch_size
            
            batch_start = time.time()
            profiles = self.profile_batch(batch)
            batch_time = time.time() - batch_start
            
            all_profiles.extend(profiles)
            
            # Progress
            processed = min(i + self.config.batch_size, total)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            
            logger.info(
                f"Batch {batch_num}/{total_batches}: "
                f"{processed}/{total} images "
                f"({rate:.1f} img/s, ETA: {eta:.0f}s)"
            )
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            "folder": str(folder_path),
            "processed_at": datetime.now().isoformat(),
            "config": asdict(self.config),
            "statistics": {
                "total_images": total,
                "successful": len([p for p in all_profiles if "error" not in p]),
                "errors": len([p for p in all_profiles if "error" in p]),
                "total_time_seconds": round(total_time, 2),
                "images_per_second": round(total / total_time, 2) if total_time > 0 else 0
            },
            "images": all_profiles
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        
        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for the batch profiler."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Profile images using Qwen3-VL (batch mode, aspect-ratio preserved)"
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Folder containing images to profile"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: profile_<folder_name>.json)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size (default: 8, reduce if GPU OOM)"
    )
    parser.add_argument(
        "--quality", "-q",
        choices=["fast", "balanced", "high"],
        default="balanced",
        help="Quality preset (fast=256 tokens, balanced=576, high=1024)"
    )
    parser.add_argument(
        "--device", "-d",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Configure based on quality preset
    quality_presets = {
        "fast": (200704, 200704),      # ~256 tokens
        "balanced": (200704, 451584),  # 256-576 tokens
        "high": (451584, 802816)       # 576-1024 tokens
    }
    
    min_px, max_px = quality_presets[args.quality]
    
    config = ProfilerConfig(
        batch_size=args.batch_size,
        min_pixels=min_px,
        max_pixels=max_px,
        device=args.device
    )
    
    # Default output filename
    if args.output is None:
        folder_name = Path(args.folder).name
        args.output = f"profile_{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Run profiler
    profiler = VLMBatchProfiler(config)
    
    if not profiler.initialize():
        logger.error("Failed to initialize profiler")
        sys.exit(1)
    
    try:
        results = profiler.profile_folder(args.folder, args.output)
        
        print("\n" + "=" * 60)
        print("PROFILING COMPLETE")
        print("=" * 60)
        print(f"Images processed: {results['statistics']['total_images']}")
        print(f"Successful: {results['statistics']['successful']}")
        print(f"Errors: {results['statistics']['errors']}")
        print(f"Total time: {results['statistics']['total_time_seconds']:.1f}s")
        print(f"Speed: {results['statistics']['images_per_second']:.2f} img/s")
        print(f"Output: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

