#!/usr/bin/env python3
"""
Qwen Resolution Profiling Test Script

Tests Qwen's ability to perform comprehensive 9-field profiling at different resolutions.
Finds the MINIMUM resolution where Qwen can still accurately classify:
- Image type, content type, shot scale, time of day
- Human presence, human details, body type, detailed clothing, dominant colors

Usage:
1. Place 5 test images in the same folder as this script
2. Run: python test_qwen_resolution_profiling.py
3. Review results to find optimal resolution threshold

Author: AI Dataset Intelligence Team
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from PIL import Image
import torch
from typing import Dict, List, Any
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import VLM engine
try:
    from vlm_engine import VLMEngine
except ImportError as e:
    print(f"Error: Could not import VLMEngine: {e}")
    print("Make sure you're running this from the intelligence directory")
    sys.exit(1)

class QwenResolutionTester:
    def __init__(self):
        self.vlm_engine = None
        self.test_images = []
        self.results = {}
        
        # Test resolutions - VERY FINE at low end, coarser at high end
        self.test_resolutions = [
            32, 40, 48, 56, 64,           # Ultra-low (fine increments)
            72, 80, 88, 96,               # Low (fine increments)
            112, 128, 144, 160,           # Medium-low
            192, 224, 256,                # Medium
            320, 384, 448, 512,           # Medium-high
            640, 768, 896, 1024           # High (optimization range)
        ]
        
        print("🔬 Qwen Resolution Profiling Test")
        print("=" * 50)
        print(f"Testing {len(self.test_resolutions)} resolutions: {self.test_resolutions}")
        print("=" * 50)
    
    async def initialize(self):
        """Initialize VLM engine"""
        try:
            print("🚀 Initializing Qwen VLM Engine...")
            self.vlm_engine = VLMEngine()
            
            # Initialize with minimal config
            init_result = await self.vlm_engine.initialize()
            if not init_result:
                raise Exception("VLM Engine initialization failed")
            
            print("✅ VLM Engine initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ VLM Engine initialization failed: {e}")
            return False
    
    def find_test_images(self):
        """Find test images in current directory"""
        current_dir = Path(__file__).parent
        
        # Look for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
        
        for ext in image_extensions:
            for img_file in current_dir.glob(f"*{ext}"):
                if img_file.name != Path(__file__).name:  # Skip this script
                    self.test_images.append(str(img_file))
        
        print(f"📸 Found {len(self.test_images)} test images:")
        for i, img_path in enumerate(self.test_images, 1):
            img_name = Path(img_path).name
            try:
                img = Image.open(img_path)
                print(f"  {i}. {img_name} ({img.width}x{img.height})")
            except:
                print(f"  {i}. {img_name} (could not read dimensions)")
        
        if len(self.test_images) == 0:
            print("❌ No test images found!")
            print("Please place 5 test images in the same folder as this script")
            return False
        
        return True
    
    async def test_resolution(self, image_path: str, resolution: int) -> Dict[str, Any]:
        """Test profiling at specific resolution"""
        try:
            start_time = time.time()
            
            # Use VLM engine's classify method with specific resolution
            result = await self.vlm_engine._classify(image_path, max_resolution=resolution)
            
            elapsed = time.time() - start_time
            
            # Add timing and resolution info
            result['resolution_tested'] = resolution
            result['processing_time'] = elapsed
            result['success'] = True
            
            return result
            
        except Exception as e:
            return {
                'resolution_tested': resolution,
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def display_result(self, image_path: str, resolution: int, result: Dict[str, Any]):
        """Display test result in readable format"""
        img_name = Path(image_path).name
        
        print(f"\n📊 RESOLUTION {resolution}px - {img_name}")
        print("-" * 60)
        
        if not result.get('success', False):
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return
        
        # Show original image info
        try:
            img = Image.open(image_path)
            print(f"🖼️  Original: {img.width}x{img.height}px")
        except:
            print("🖼️  Original: Could not read dimensions")
        
        # Show processing time
        print(f"⏱️  Processing Time: {result.get('processing_time', 0):.3f}s")
        
        # Show all 9 profiling fields
        print(f"📝 PROFILING RESULTS:")
        print(f"   Image Type: {result.get('image_type', 'N/A')}")
        print(f"   Content Type: {result.get('content_type', 'N/A')}")
        print(f"   Shot Scale: {result.get('shot_scale', 'N/A')}")
        print(f"   Time of Day: {result.get('time_of_day', 'N/A')}")
        print(f"   Human Presence: {result.get('human_presence', 'N/A')}")
        print(f"   Human Details: {result.get('human_details', 'N/A')}")
        print(f"   Body Type: {result.get('body_type', 'N/A')}")
        print(f"   Detailed Clothing: {result.get('detailed_clothing', 'N/A')}")
        print(f"   Dominant Colors: {result.get('dominant_colors', 'N/A')}")
        
        # Quality assessment
        filled_fields = sum(1 for field in ['image_type', 'content_type', 'shot_scale', 'time_of_day', 
                                          'human_presence', 'human_details', 'body_type', 
                                          'detailed_clothing', 'dominant_colors'] 
                          if result.get(field) and result.get(field) != 'N/A' and result.get(field).strip())
        
        quality_score = filled_fields / 9.0
        print(f"📈 Quality Score: {quality_score:.1%} ({filled_fields}/9 fields populated)")
        
        if quality_score >= 0.8:
            print("✅ EXCELLENT - All critical fields captured")
        elif quality_score >= 0.6:
            print("⚠️  GOOD - Most fields captured, some degradation")
        elif quality_score >= 0.4:
            print("🔶 FAIR - Significant degradation, basic info only")
        else:
            print("❌ POOR - Major degradation, unreliable results")
    
    def save_results(self):
        """Save all results to JSON file"""
        output_file = Path(__file__).parent / "qwen_resolution_test_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def analyze_threshold(self):
        """Analyze results to find optimal resolution threshold"""
        print("\n" + "=" * 60)
        print("📊 THRESHOLD ANALYSIS")
        print("=" * 60)
        
        # Calculate average quality scores per resolution
        resolution_scores = {}
        
        for image_path, image_results in self.results.items():
            img_name = Path(image_path).name
            
            for resolution, result in image_results.items():
                if result.get('success', False):
                    # Calculate quality score
                    filled_fields = sum(1 for field in ['image_type', 'content_type', 'shot_scale', 'time_of_day', 
                                                      'human_presence', 'human_details', 'body_type', 
                                                      'detailed_clothing', 'dominant_colors'] 
                                      if result.get(field) and result.get(field) != 'N/A' and result.get(field).strip())
                    
                    quality_score = filled_fields / 9.0
                    
                    if resolution not in resolution_scores:
                        resolution_scores[resolution] = []
                    resolution_scores[resolution].append(quality_score)
        
        # Find thresholds
        print("\n🎯 RESOLUTION RECOMMENDATIONS:")
        
        for resolution in sorted(resolution_scores.keys()):
            scores = resolution_scores[resolution]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            
            print(f"{resolution:4d}px: Avg {avg_score:.1%}, Min {min_score:.1%}", end="")
            
            if avg_score >= 0.9 and min_score >= 0.8:
                print(" 🏆 EXCELLENT - Recommended for production")
            elif avg_score >= 0.8 and min_score >= 0.7:
                print(" ✅ VERY GOOD - Safe for most use cases")
            elif avg_score >= 0.7 and min_score >= 0.6:
                print(" ⚠️  ACCEPTABLE - Some quality loss")
            elif avg_score >= 0.5:
                print(" 🔶 MARGINAL - Significant degradation")
            else:
                print(" ❌ POOR - Not recommended")
        
        # Find minimum acceptable resolution
        acceptable_resolutions = [r for r, scores in resolution_scores.items() 
                                if sum(scores) / len(scores) >= 0.8 and min(scores) >= 0.7]
        
        if acceptable_resolutions:
            min_acceptable = min(acceptable_resolutions)
            print(f"\n🎯 MINIMUM RECOMMENDED RESOLUTION: {min_acceptable}px")
            print(f"   (Maintains ≥80% average quality, ≥70% minimum quality)")
        else:
            print(f"\n⚠️  No resolution met quality thresholds - consider higher resolutions")
    
    async def run_comprehensive_test(self):
        """Run the complete resolution test suite"""
        if not await self.initialize():
            return False
        
        if not self.find_test_images():
            return False
        
        print(f"\n🧪 Starting comprehensive test on {len(self.test_images)} images...")
        print(f"Testing {len(self.test_resolutions)} resolutions from {min(self.test_resolutions)}px to {max(self.test_resolutions)}px")
        
        total_tests = len(self.test_images) * len(self.test_resolutions)
        current_test = 0
        
        for image_path in self.test_images:
            img_name = Path(image_path).name
            self.results[image_path] = {}
            
            print(f"\n🖼️  Testing {img_name}...")
            
            for resolution in self.test_resolutions:
                current_test += 1
                progress = (current_test / total_tests) * 100
                
                print(f"   Resolution {resolution:4d}px... ({current_test}/{total_tests} - {progress:.1f}%)", end="", flush=True)
                
                result = await self.test_resolution(image_path, resolution)
                self.results[image_path][resolution] = result
                
                # Quick status
                if result.get('success', False):
                    filled_fields = sum(1 for field in ['image_type', 'content_type', 'shot_scale', 'time_of_day', 
                                                      'human_presence', 'human_details', 'body_type', 
                                                      'detailed_clothing', 'dominant_colors'] 
                                      if result.get(field) and result.get(field) != 'N/A' and result.get(field).strip())
                    quality = filled_fields / 9.0
                    print(f" ✅ {quality:.0%} quality ({result.get('processing_time', 0):.2f}s)")
                else:
                    print(f" ❌ FAILED")
        
        # Show detailed results for each resolution
        print(f"\n" + "=" * 80)
        print("📋 DETAILED RESULTS BY RESOLUTION")
        print("=" * 80)
        
        for resolution in self.test_resolutions:
            print(f"\n🔍 RESOLUTION: {resolution}px")
            print("=" * 40)
            
            for image_path in self.test_images:
                result = self.results[image_path][resolution]
                self.display_result(image_path, resolution, result)
        
        # Analysis
        self.analyze_threshold()
        
        # Save results
        self.save_results()
        
        print(f"\n🎉 Test complete! Tested {total_tests} combinations.")
        print("📊 Check the analysis above to find your optimal resolution threshold.")
        
        return True

def main():
    """Main entry point"""
    print("🔬 Qwen Resolution Profiling Test Script")
    print("=" * 50)
    print("This script will test Qwen's profiling accuracy at different resolutions.")
    print("Place 5 diverse test images in this folder before running.")
    print("=" * 50)
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - this will be very slow!")
    
    # Confirm with user
    response = input("\n▶️  Ready to start testing? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Test cancelled.")
        return
    
    # Run test
    tester = QwenResolutionTester()
    
    try:
        success = asyncio.run(tester.run_comprehensive_test())
        if success:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test failed!")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")

if __name__ == "__main__":
    main()
