#!/usr/bin/env python3
"""
VLM Profiling Resolution Test
==============================

Tests Qwen3-VL classification accuracy at different resolutions.
Drop this in a folder with images and run it to find the optimal
resolution for your use case.

USAGE:
    python test_profiling_resolution.py --folder /path/to/test/images

The script will test each resolution and report accuracy/speed tradeoffs.

EDIT THE CONFIG BELOW TO CUSTOMIZE:
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ============================================================================
# ⚙️ USER CONFIGURATION - EDIT THESE VALUES ⚙️
# ============================================================================

@dataclass
class TestConfig:
    """
    EDIT THESE VALUES TO CUSTOMIZE THE TEST
    ----------------------------------------
    """
    
    # Model to use
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    
    # Resolution presets to test (in pixels for longest edge)
    # Qwen3-VL uses 28x28 patches, so optimal values are multiples of 28
    # Common test points: 64, 112, 168, 224, 280, 336, 392, 448, 504, 560
    # Or simpler: 64, 128, 192, 256, 320, 384, 448, 512
    resolutions_to_test: List[int] = field(default_factory=lambda: [
        64,    # Minimum - very fast but low quality
        128,   # Very low - basic detection only
        192,   # Low - simple classification
        256,   # Standard - good speed/quality balance
        320,   # Medium-low
        384,   # Medium - better for details
        448,   # Medium-high
        512,   # High - current default
        640,   # Very high - for fine details
        768,   # Maximum practical
    ])
    
    # How many images to test per resolution (set to None for all)
    max_images_per_resolution: Optional[int] = 10
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Output file for results
    output_file: str = "resolution_test_results.json"
    
    # Verbose logging
    verbose: bool = True


# Create default config
CONFIG = TestConfig()

# ============================================================================
# RESOLUTION CONVERSION UTILITIES
# ============================================================================

def pixels_to_resolution(longest_edge: int) -> int:
    """
    Convert longest edge to approximate min_pixels for Qwen processor.
    
    Qwen3-VL uses 28x28 patches. The processor uses min_pixels/max_pixels
    to control the total pixel count while preserving aspect ratio.
    
    For a square image at 256px: 256 * 256 = 65,536 pixels
    For a 16:9 image at 256px longest: 256 * 144 = 36,864 pixels
    
    We use the formula: min_pixels ≈ (longest_edge * 0.75) ^ 2
    This accounts for typical aspect ratios.
    """
    # Approximate for typical aspect ratios (16:9 to 4:3)
    avg_short_edge = int(longest_edge * 0.7)  # Average short edge
    return longest_edge * avg_short_edge


def resolution_to_tokens(longest_edge: int) -> int:
    """
    Estimate token count for a given resolution.
    
    Qwen3-VL uses 28x28 patches.
    tokens ≈ (pixels) / (28 * 28)
    """
    pixels = pixels_to_resolution(longest_edge)
    return pixels // (28 * 28)


# ============================================================================
# TEST RUNNER
# ============================================================================

class ResolutionTester:
    """Tests VLM classification at different resolutions."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('resolution_test')
    
    def initialize(self) -> bool:
        """Load the model."""
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            
            self.logger.info(f"Loading {self.config.model_name}...")
            
            # Device
            if self.config.device == "cuda" and torch.cuda.is_available():
                device = "cuda"
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                self.logger.warning("Using CPU - tests will be slow")
            
            # Load model once (we'll adjust resolution per-test)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    def _get_processor(self, resolution: int):
        """Get processor configured for specific resolution."""
        from transformers import AutoProcessor
        
        min_pixels = pixels_to_resolution(resolution)
        max_pixels = min_pixels * 2  # Allow some flexibility
        
        processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        return processor
    
    def _classify_image(
        self, 
        image_path: str, 
        resolution: int
    ) -> Dict[str, Any]:
        """Classify a single image at a specific resolution."""
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        
        processor = self._get_processor(resolution)
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        aspect_ratio = original_size[0] / original_size[1]
        
        # Determine orientation
        if aspect_ratio > 1.2:
            orientation = "landscape"
        elif aspect_ratio < 0.83:
            orientation = "portrait"
        else:
            orientation = "square"
        
        # Build prompt with aspect info
        prompt = f"""Analyze this image ({orientation}, {original_size[0]}x{original_size[1]}).

Respond in JSON:
{{
    "image_type": "photograph|illustration|3d_render|digital_art",
    "content_type": "portrait|landscape|architecture|still_life|object|action",
    "shot_scale": "extreme_closeup|closeup|medium|wide|extreme_wide",
    "time_of_day": "day|night|golden_hour|twilight|unknown",
    "human_presence": "none|single|multiple",
    "confidence": "high|medium|low"
}}

Be concise."""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Process
        start_time = time.time()
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        
        elapsed = time.time() - start_time
        
        # Decode
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(
            generated, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse
        result = self._parse_response(response)
        result['_meta'] = {
            'filename': os.path.basename(image_path),
            'resolution': resolution,
            'inference_time': round(elapsed, 3),
            'original_size': original_size,
            'orientation': orientation
        }
        
        return result
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response."""
        import re
        
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {"raw": text, "parse_error": True}
    
    def test_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Test all resolutions on images in folder.
        
        Returns results with timing and quality comparisons.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find images
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(folder.glob(f"*{ext}"))
            image_paths.extend(folder.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(set(str(p) for p in image_paths))
        
        if self.config.max_images_per_resolution:
            image_paths = image_paths[:self.config.max_images_per_resolution]
        
        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")
        
        self.logger.info(f"Testing {len(image_paths)} images across {len(self.config.resolutions_to_test)} resolutions")
        
        # Test each resolution
        all_results = {}
        
        for resolution in self.config.resolutions_to_test:
            tokens = resolution_to_tokens(resolution)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing {resolution}px (~{tokens} tokens)")
            self.logger.info(f"{'='*60}")
            
            resolution_results = []
            total_time = 0
            
            for i, image_path in enumerate(image_paths):
                try:
                    result = self._classify_image(image_path, resolution)
                    resolution_results.append(result)
                    total_time += result['_meta']['inference_time']
                    
                    self.logger.debug(
                        f"  [{i+1}/{len(image_paths)}] {result['_meta']['filename']}: "
                        f"{result.get('content_type', '?')} / {result.get('shot_scale', '?')} "
                        f"({result['_meta']['inference_time']:.2f}s)"
                    )
                    
                except Exception as e:
                    self.logger.error(f"  Failed {image_path}: {e}")
                    resolution_results.append({
                        '_meta': {
                            'filename': os.path.basename(image_path),
                            'resolution': resolution,
                            'error': str(e)
                        }
                    })
            
            avg_time = total_time / len(resolution_results) if resolution_results else 0
            
            all_results[f"{resolution}px"] = {
                'resolution': resolution,
                'approx_tokens': tokens,
                'images_tested': len(resolution_results),
                'total_time': round(total_time, 2),
                'avg_time_per_image': round(avg_time, 3),
                'images_per_second': round(1/avg_time, 2) if avg_time > 0 else 0,
                'results': resolution_results
            }
            
            self.logger.info(
                f"  Summary: {len(resolution_results)} images in {total_time:.1f}s "
                f"(avg {avg_time:.2f}s/img, {1/avg_time:.1f} img/s)"
            )
        
        # Compile final report
        report = {
            'test_date': datetime.now().isoformat(),
            'folder': str(folder_path),
            'model': self.config.model_name,
            'images_tested': len(image_paths),
            'resolutions_tested': self.config.resolutions_to_test,
            'summary': self._generate_summary(all_results),
            'detailed_results': all_results
        }
        
        # Save results
        output_path = Path(folder_path) / self.config.output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\nResults saved to: {output_path}")
        
        return report
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary comparison across resolutions."""
        summary = []
        
        for key, data in results.items():
            summary.append({
                'resolution': f"{data['resolution']}px",
                'tokens': data['approx_tokens'],
                'avg_time': data['avg_time_per_image'],
                'speed': f"{data['images_per_second']} img/s",
                'errors': sum(1 for r in data['results'] if '_meta' in r and 'error' in r['_meta'])
            })
        
        return summary


# ============================================================================
# CLI
# ============================================================================

def print_resolution_guide():
    """Print resolution/token reference guide."""
    print("\n" + "="*70)
    print("RESOLUTION / TOKEN REFERENCE GUIDE")
    print("="*70)
    print(f"{'Resolution':<12} {'Est. Tokens':<12} {'Speed':<15} {'Quality':<20}")
    print("-"*70)
    
    resolutions = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768]
    speeds = ["Very Fast", "Very Fast", "Fast", "Fast", "Medium", "Medium", "Slow", "Slow", "Very Slow", "Very Slow"]
    qualities = ["Minimal", "Basic", "Low", "Good", "Good+", "Very Good", "High", "Very High", "Excellent", "Maximum"]
    
    for res, speed, quality in zip(resolutions, speeds, qualities):
        tokens = resolution_to_tokens(res)
        print(f"{res}px{'':<8} ~{tokens:<11} {speed:<15} {quality:<20}")
    
    print("-"*70)
    print("\nRecommended:")
    print("  • 256px: Best speed/quality balance for basic profiling")
    print("  • 384px: Good for clothing colors, expressions")
    print("  • 512px: Current default, best for fine details")
    print("  • 64-128px: Only for quick human/no-human detection")
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Qwen3-VL profiling accuracy at different resolutions"
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Folder containing test images"
    )
    parser.add_argument(
        "--resolutions", "-r",
        nargs="+",
        type=int,
        default=None,
        help="Specific resolutions to test (e.g., --resolutions 256 384 512)"
    )
    parser.add_argument(
        "--max-images", "-n",
        type=int,
        default=10,
        help="Max images to test per resolution (default: 10)"
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Print resolution/token guide and exit"
    )
    
    args = parser.parse_args()
    
    if args.guide:
        print_resolution_guide()
        return
    
    # Update config with CLI args
    if args.resolutions:
        CONFIG.resolutions_to_test = args.resolutions
    CONFIG.max_images_per_resolution = args.max_images
    
    print_resolution_guide()
    
    # Run tests
    tester = ResolutionTester(CONFIG)
    
    if not tester.initialize():
        print("Failed to initialize model")
        sys.exit(1)
    
    try:
        report = tester.test_folder(args.folder)
        
        print("\n" + "="*70)
        print("TEST COMPLETE - SUMMARY")
        print("="*70)
        print(f"{'Resolution':<12} {'Tokens':<10} {'Avg Time':<12} {'Speed':<15}")
        print("-"*70)
        
        for item in report['summary']:
            print(f"{item['resolution']:<12} ~{item['tokens']:<9} {item['avg_time']:.2f}s{'':<7} {item['speed']:<15}")
        
        print("="*70)
        print(f"\nDetailed results saved to: {CONFIG.output_file}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

