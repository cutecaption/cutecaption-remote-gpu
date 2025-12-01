#!/usr/bin/env python3
"""
VLM Resolution Threshold Tester

Tests Qwen3-VL's ability to accurately classify images at different resolutions
to find the optimal balance between speed and accuracy.

Tests:
1. shot_scale classification accuracy at 192px, 256px, 320px, 384px, 512px
2. time_of_day classification accuracy at different resolutions
3. image_type and content_type accuracy

Usage:
    python test_vlm_resolution_threshold.py <image_path> [--resolutions 192,256,384,512]
    
Example:
    python test_vlm_resolution_threshold.py test_images/portrait.jpg
    python test_vlm_resolution_threshold.py test_images/landscape.jpg --resolutions 192,256,320,384,512
    
This will help determine the minimum resolution needed for accurate classification.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Set environment before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['HF_HOME'] = os.environ.get('TRANSFORMERS_CACHE', 'D:\\hfcache')

from vlm_engine import VLMEngine


async def test_resolution_accuracy(image_path: str, resolutions: list):
    """
    Test classification accuracy at different resolutions
    
    Args:
        image_path: Path to test image
        resolutions: List of resolutions to test (e.g. [192, 256, 384, 512])
    """
    print("="*80)
    print(f"Testing VLM Classification Resolution Threshold")
    print(f"Image: {image_path}")
    print("="*80)
    
    # Initialize VLM engine
    print("\nInitializing Qwen3-VL...")
    engine = VLMEngine()
    await engine.initialize()
    print("VLM Engine ready!\n")
    
    # Get baseline (highest resolution)
    baseline_res = max(resolutions)
    print(f"Establishing baseline at {baseline_res}px...")
    baseline = await engine._classify(image_path, max_resolution=baseline_res)
    
    print("\nBaseline Classification:")
    print(f"  Image Type:   {baseline['image_type']}")
    print(f"  Content Type: {baseline['content_type']}")
    print(f"  Shot Scale:   {baseline['shot_scale']}")
    print(f"  Time of Day:  {baseline['time_of_day']}")
    
    print("\n" + "="*80)
    print("Testing Different Resolutions")
    print("="*80)
    
    results = []
    
    for res in sorted(resolutions):
        print(f"\n[{res}px]")
        print("-" * 40)
        
        result = await engine._classify(image_path, max_resolution=res)
        
        # Compare to baseline
        matches = {
            'image_type': result['image_type'] == baseline['image_type'],
            'content_type': result['content_type'] == baseline['content_type'],
            'shot_scale': result['shot_scale'] == baseline['shot_scale'],
            'time_of_day': result['time_of_day'] == baseline['time_of_day']
        }
        
        accuracy = sum(matches.values()) / len(matches) * 100
        
        print(f"  Image Type:   {result['image_type']:<20} {'✓' if matches['image_type'] else '✗ MISMATCH'}")
        print(f"  Content Type: {result['content_type']:<20} {'✓' if matches['content_type'] else '✗ MISMATCH'}")
        print(f"  Shot Scale:   {result['shot_scale']:<20} {'✓' if matches['shot_scale'] else '✗ MISMATCH'}")
        print(f"  Time of Day:  {result['time_of_day']:<20} {'✓' if matches['time_of_day'] else '✗ MISMATCH'}")
        print(f"  Accuracy:     {accuracy:.0f}%")
        
        results.append({
            'resolution': res,
            'accuracy': accuracy,
            'matches': matches,
            'classification': result
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for r in results:
        status = "✓ ACCURATE" if r['accuracy'] == 100 else f"✗ {r['accuracy']:.0f}% accurate"
        print(f"{r['resolution']:>4}px: {status}")
    
    # Find minimum accurate resolution
    accurate_resolutions = [r for r in results if r['accuracy'] == 100]
    if accurate_resolutions:
        min_accurate = min(r['resolution'] for r in accurate_resolutions)
        print(f"\nRECOMMENDATION: Minimum resolution for accurate classification: {min_accurate}px")
    else:
        print(f"\nWARNING: No resolution achieved 100% accuracy. Use highest tested resolution.")
    
    print("\nNOTE: Test on multiple images (portraits, landscapes, night scenes, etc.)")
    print("      to determine the true minimum threshold for your use case.")
    print("="*80)


async def main():
    parser = argparse.ArgumentParser(description='Test VLM classification accuracy at different resolutions')
    parser.add_argument('image_path', help='Path to test image')
    parser.add_argument('--resolutions', default='192,256,320,384,512', 
                       help='Comma-separated list of resolutions to test (default: 192,256,320,384,512)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    resolutions = [int(r.strip()) for r in args.resolutions.split(',')]
    
    await test_resolution_accuracy(args.image_path, resolutions)


if __name__ == "__main__":
    asyncio.run(main())

