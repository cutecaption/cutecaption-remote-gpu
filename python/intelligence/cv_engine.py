"""
CV Engine - Computer Vision Expert Service
Performs fast, non-neural CV analysis for Universal Dataset Profile

Capabilities:
- Brightness analysis (pixel histogram)
- Sharpness analysis (Laplacian variance)
- Visual complexity (Canny edge detection)
- Color palette extraction (k-Means clustering)
- Color vibrancy score (saturation analysis)
- Shot scale detection (saliency + bbox)
- Focal point quadrant (saliency model)

All operations are optimized for batch processing
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine

# Import CV libraries
try:
    import cv2
    from PIL import Image
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"Error importing dependencies: {e}", file=sys.stderr)
    sys.exit(1)


class CVEngine(BaseEngine):
    """
    Computer Vision Engine - Fast classical CV analysis
    
    Generates universal metadata for any image without neural networks
    Designed for speed and reliability
    """
    
    def __init__(self):
        super().__init__("cv_engine", "1.0.0")
        
        # Configuration
        self.color_clusters = 5  # Extract top 5 colors
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
    
    def _load_image_with_exif(self, image_path: str):
        """
        Load image with EXIF orientation correction.
        
        cv2.imread ignores EXIF orientation tags, causing portrait photos
        from phones to be loaded sideways. This uses PIL to apply EXIF
        orientation first, then converts to OpenCV format.
        """
        try:
            from PIL import Image, ImageOps
            
            # Load with PIL (handles more formats than cv2)
            pil_img = Image.open(image_path)
            
            # Apply EXIF orientation
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass  # Continue if EXIF transpose fails
            
            # Convert to RGB if needed (handles RGBA, palette, etc.)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Convert PIL → NumPy → OpenCV BGR format
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
        except Exception as e:
            self.logger.warning(f"EXIF-aware load failed, falling back to cv2.imread: {e}")
            # Fallback to standard cv2.imread
            return cv2.imread(image_path)
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize CV Engine (no models to load)"""
        try:
            self.logger.info("Initializing CV Engine...")
            
            # Just verify OpenCV is working
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            
            self._mark_initialized()
            
            return {
                "opencv_version": cv2.__version__,
                "ready": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self._mark_unhealthy(str(e))
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a CV request
        
        Supported actions:
        - analyze_universal_profile: Generate all universal metadata
        - calculate_brightness: Just brightness
        - calculate_sharpness: Just sharpness
        - calculate_complexity: Just complexity
        - extract_color_palette: Just colors
        - batch_analyze: Process multiple images
        """
        action = request.get('action')
        
        # Handle health check
        if action == 'health_check':
            return await self.health_check()
        
        if action == 'analyze_universal_profile':
            return await self._analyze_universal_profile(request['image_path'])
        
        elif action == 'calculate_brightness':
            return await self._calculate_brightness(request['image_path'])
        
        elif action == 'calculate_sharpness':
            return await self._calculate_sharpness(request['image_path'])
        
        elif action == 'calculate_complexity':
            return await self._calculate_complexity(request['image_path'])
        
        elif action == 'extract_color_palette':
            return await self._extract_color_palette(request['image_path'])
        
        elif action == 'batch_analyze':
            return await self._batch_analyze(
                request['image_paths'],
                request.get('batch_size', 32)
            )
        
        elif action == 'filter_by_metrics':
            return await self._filter_by_metrics(
                request['candidates'],
                request.get('quality'),
                request.get('brightness'),
                request.get('sharpness'),
                request.get('lighting')
            )
        
        elif action == 'calculate_training_quality':
            return await self._calculate_training_quality(
                request.get('image_paths', []),
                request.get('include_sharpness', True),
                request.get('include_exposure', True),
                request.get('include_composition', True)
            )
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _analyze_universal_profile(self, image_path: str) -> Dict[str, Any]:
        """
        Generate complete universal profile for an image
        
        Returns all 9 universal metadata fields:
        - brightness_score
        - sharpness_score
        - visual_complexity_score
        - color_palette (5 colors)
        - color_vibrancy_score
        - shot_scale
        - focal_point_quadrant
        - dominant_color
        - aspect_ratio
        """
        try:
            # Load image with EXIF orientation handling
            img = self._load_image_with_exif(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Resize for faster processing (max 300px on longest side - sufficient for CV metrics)
            height, width = img.shape[:2]
            max_dim = max(height, width)
            if max_dim > 300:
                scale = 300 / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                height, width = new_height, new_width
            
            # Convert to RGB for PIL operations
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Generate all metadata (9 universal + 3 quality fields)
            profile = {
                # Universal fields
                "brightness_score": self._calc_brightness(img),
                "sharpness_score": self._calc_sharpness(img),
                "visual_complexity_score": self._calc_complexity(img),
                "color_palette": self._calc_color_palette(img_rgb),
                "color_vibrancy_score": self._calc_vibrancy(img_rgb),
                "dominant_color": self._calc_dominant_color(img_rgb),
                "aspect_ratio": round(width / height, 3),
                "shot_scale": self._calc_shot_scale(img),
                "focal_point_quadrant": self._calc_focal_point(img),
                "dimensions": {"width": width, "height": height},
                
                # Quality metrics (critical for dataset curation)
                "aesthetic_score": self._calc_aesthetic_quality(img, img_rgb),
                "blur_score": self._calc_blur_quality(img),
                "noise_score": self._calc_noise_level(img)
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error analyzing {image_path}: {e}")
            raise
    
    def _calc_brightness(self, img: np.ndarray) -> float:
        """
        Calculate brightness score (0.0 to 1.0)
        
        Method: Average pixel brightness in grayscale
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray) / 255.0)
    
    def _calc_sharpness(self, img: np.ndarray) -> float:
        """
        Calculate sharpness score (0.0 to 1.0+)
        
        Method: Laplacian variance - higher = sharper
        Normalized to 0-1 range (typical range 0-100)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize: typical sharp images have variance 100-500
        # Blurry images have variance < 100
        normalized = min(variance / 500.0, 1.0)
        
        return float(normalized)
    
    def _calc_complexity(self, img: np.ndarray) -> float:
        """
        Calculate visual complexity score (0.0 to 1.0)
        
        Method: Canny edge detection density
        More edges = more complex
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Normalize to 0-1 (typical range 0.01 to 0.3)
        normalized = min(edge_density / 0.3, 1.0)
        
        return float(normalized)
    
    def _calc_color_palette(self, img_rgb: np.ndarray) -> List[str]:
        """
        Extract top N dominant colors using k-Means clustering
        
        Returns: List of hex color codes
        """
        # Reshape image to list of pixels
        pixels = img_rgb.reshape(-1, 3)
        
        # Sample pixels aggressively for speed (use every 20th pixel instead of 10th)
        sampled_pixels = pixels[::20]
        
        # Limit to max 5000 pixels for k-means
        if len(sampled_pixels) > 5000:
            sampled_pixels = sampled_pixels[:5000]
        
        # Perform k-Means clustering with fewer iterations
        kmeans = KMeans(n_clusters=self.color_clusters, random_state=42, n_init=3, max_iter=100)
        kmeans.fit(sampled_pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        
        return hex_colors
    
    def _calc_vibrancy(self, img_rgb: np.ndarray) -> float:
        """
        Calculate color vibrancy score (0.0 to 1.0)
        
        Method: Average saturation in HSV color space
        """
        # Convert to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Extract saturation channel
        saturation = img_hsv[:, :, 1]
        
        # Calculate average saturation
        avg_saturation = np.mean(saturation) / 255.0
        
        return float(avg_saturation)
    
    def _calc_shot_scale(self, img: np.ndarray) -> str:
        """
        Estimate shot scale based on simple heuristics
        
        Returns: 'extreme_wide', 'wide', 'medium', 'close_up', 'extreme_close_up'
        
        Method: Analyze edge distribution and complexity
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Calculate center region (middle 40%)
        center_h_start = int(height * 0.3)
        center_h_end = int(height * 0.7)
        center_w_start = int(width * 0.3)
        center_w_end = int(width * 0.7)
        
        center_region = gray[center_h_start:center_h_end, center_w_start:center_w_end]
        
        # Calculate edge density in center vs. whole image
        edges_full = cv2.Canny(gray, 50, 150)
        edges_center = cv2.Canny(center_region, 50, 150)
        
        density_full = np.sum(edges_full > 0) / edges_full.size
        density_center = np.sum(edges_center > 0) / edges_center.size
        
        # Ratio indicates subject concentration
        if density_center < 0.01:
            return "extreme_wide"
        
        ratio = density_center / max(density_full, 0.001)
        
        if ratio > 3.0:
            return "extreme_close_up"
        elif ratio > 2.0:
            return "close_up"
        elif ratio > 1.0:
            return "medium"
        elif ratio > 0.5:
            return "wide"
        else:
            return "extreme_wide"
    
    def _calc_focal_point(self, img: np.ndarray) -> str:
        """
        Estimate focal point quadrant
        
        Returns: 'top-left', 'top-center', 'top-right', 
                 'center-left', 'center', 'center-right',
                 'bottom-left', 'bottom-center', 'bottom-right'
        
        Method: Simple saliency approximation using edge detection
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide image into 3x3 grid
        height, width = edges.shape
        h_third = height // 3
        w_third = width // 3
        
        # Calculate edge density in each region
        regions = {}
        region_names = [
            ['top-left', 'top-center', 'top-right'],
            ['center-left', 'center', 'center-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        
        max_density = 0
        focal_region = 'center'
        
        for i in range(3):
            for j in range(3):
                region = edges[i*h_third:(i+1)*h_third, j*w_third:(j+1)*w_third]
                density = np.sum(region > 0) / region.size
                
                if density > max_density:
                    max_density = density
                    focal_region = region_names[i][j]
        
        return focal_region
    
    def _calc_dominant_color(self, img_rgb: np.ndarray) -> str:
        """
        Get single dominant color name
        
        Returns: Color name (e.g., 'red', 'blue', 'green', etc.)
        """
        # Get color palette
        palette = self._calc_color_palette(img_rgb)
        
        # Convert first (most dominant) hex to RGB
        hex_color = palette[0]
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # Simple color name mapping
        max_channel = max(r, g, b)
        
        if max_channel < 50:
            return "black"
        elif min(r, g, b) > 200:
            return "white"
        elif r > g and r > b:
            if g > 100:
                return "orange" if g > b else "brown"
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            if r > 100:
                return "purple"
            return "blue"
        else:
            return "gray"
    
    async def _calculate_brightness(self, image_path: str) -> Dict[str, Any]:
        """Calculate only brightness"""
        img = self._load_image_with_exif(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return {
            "brightness_score": self._calc_brightness(img),
            "image_path": image_path
        }
    
    async def _calculate_sharpness(self, image_path: str) -> Dict[str, Any]:
        """Calculate only sharpness"""
        img = self._load_image_with_exif(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return {
            "sharpness_score": self._calc_sharpness(img),
            "image_path": image_path
        }
    
    async def _calculate_complexity(self, image_path: str) -> Dict[str, Any]:
        """Calculate only complexity"""
        img = self._load_image_with_exif(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return {
            "visual_complexity_score": self._calc_complexity(img),
            "image_path": image_path
        }
    
    async def _extract_color_palette(self, image_path: str) -> Dict[str, Any]:
        """Extract only color palette"""
        img = self._load_image_with_exif(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            "color_palette": self._calc_color_palette(img_rgb),
            "dominant_color": self._calc_dominant_color(img_rgb),
            "color_vibrancy_score": self._calc_vibrancy(img_rgb),
            "image_path": image_path
        }
    
    def _calc_aesthetic_quality(self, img: np.ndarray, img_rgb: np.ndarray) -> float:
        """
        Calculate aesthetic quality score (0.0 to 1.0)
        Based on rule-of-thirds, color harmony, and composition balance
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Rule of thirds: Check if focal points align with grid intersections
        thirds_h = [height // 3, 2 * height // 3]
        thirds_w = [width // 3, 2 * width // 3]
        
        # Find regions of interest using edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Check alignment with rule of thirds intersections
        alignment_score = 0
        for h in thirds_h:
            for w in thirds_w:
                region = edges[max(0, h-20):min(height, h+20), max(0, w-20):min(width, w+20)]
                if region.size > 0:
                    alignment_score += np.sum(region > 0) / region.size
        
        alignment_score = min(alignment_score / 4, 1.0)  # Normalize
        
        # Color harmony: Check color distribution
        palette = self._calc_color_palette(img_rgb)
        color_diversity = min(len(set(palette)) / 5.0, 1.0)
        
        # Composition balance: Check left-right, top-bottom symmetry
        left_half = edges[:, :width//2]
        right_half = edges[:, width//2:]
        # Use float64 to prevent overflow and normalize
        left_sum = np.sum(left_half, dtype=np.float64)
        right_sum = np.sum(right_half, dtype=np.float64)
        total_sum = np.sum(edges, dtype=np.float64)
        
        if total_sum > 0:
            balance_score = 1.0 - min(abs(left_sum - right_sum) / total_sum, 1.0)
        else:
            balance_score = 1.0
        
        # Combined aesthetic score
        aesthetic = (alignment_score * 0.4 + color_diversity * 0.3 + balance_score * 0.3)
        
        return float(aesthetic)
    
    def _calc_blur_quality(self, img: np.ndarray) -> float:
        """
        Calculate blur quality (0.0 = very blurry, 1.0 = sharp)
        Uses Laplacian variance - same as sharpness but inverted scale
        """
        # Sharpness is already a good blur metric
        # High sharpness = low blur
        return self._calc_sharpness(img)
    
    def _calc_noise_level(self, img: np.ndarray) -> float:
        """
        Calculate noise level (0.0 = clean, 1.0 = noisy)
        Based on high-frequency content analysis
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to get smooth version
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate difference (noise is high-frequency)
        noise = cv2.absdiff(gray, blurred)
        
        # Normalize noise level
        noise_level = np.mean(noise) / 255.0
        
        return float(noise_level)
    
    async def _batch_analyze(self, image_paths: List[str], batch_size: int) -> Dict[str, Any]:
        """
        Batch process multiple images
        
        Returns profiles for all images that processed successfully
        """
        results = []
        failed = []
        
        for i, image_path in enumerate(image_paths):
            try:
                profile = await self._analyze_universal_profile(image_path)
                results.append({
                    "image_path": image_path,
                    "profile": profile
                })
                
                # Log progress every batch_size images
                if (i + 1) % batch_size == 0:
                    self.logger.info(f"Processed {i + 1}/{len(image_paths)} images")
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {image_path}: {e}")
                failed.append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        return {
            "results": results,
            "successful": len(results),
            "failed": len(failed),
            "failures": failed
        }
    
    async def _filter_by_metrics(
        self, 
        candidates: List[Dict[str, Any]], 
        quality: str = None,
        brightness: str = None,
        sharpness: str = None,
        lighting: str = None,
        color_mode: str = None
    ) -> Dict[str, Any]:
        """
        Filter candidate images by CV metrics.
        
        Args:
            candidates: List of candidate dicts with 'path' or 'image_path'
            quality: 'high', 'low', or None
            brightness: 'bright', 'dark', 'perfect', or None
            sharpness: 'sharp', 'blurry', or None
            lighting: 'good', 'poor', 'perfect', or None
            color_mode: 'color', 'bw', 'monochrome', or None
            
        Returns:
            Filtered list of candidates that match criteria
        """
        self.logger.info(f"Filtering {len(candidates)} candidates by CV metrics")
        
        filtered = []
        
        for candidate in candidates:
            image_path = candidate.get('path') or candidate.get('image_path')
            if not image_path:
                continue
                
            try:
                # Analyze the image with EXIF orientation handling
                img = self._load_image_with_exif(image_path)
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Calculate metrics
                brightness_score = self._calc_brightness(img)
                sharpness_score = self._calc_sharpness(img)
                quality_score = self._calc_aesthetic_quality(img, img_rgb)
                
                # Apply filters
                passes = True
                
                # Quality filter
                if quality == 'high' and quality_score < 0.6:
                    passes = False
                elif quality == 'low' and quality_score > 0.4:
                    passes = False
                
                # Brightness filter
                if brightness == 'bright' and brightness_score < 0.6:
                    passes = False
                elif brightness == 'dark' and brightness_score > 0.4:
                    passes = False
                elif brightness == 'perfect' and (brightness_score < 0.4 or brightness_score > 0.7):
                    passes = False
                
                # Sharpness filter
                if sharpness == 'sharp' and sharpness_score < 0.5:
                    passes = False
                elif sharpness == 'blurry' and sharpness_score > 0.3:
                    passes = False
                
                # Lighting filter (uses brightness + contrast as proxy)
                if lighting == 'good' and (brightness_score < 0.3 or brightness_score > 0.8):
                    passes = False
                elif lighting == 'poor' and (brightness_score > 0.3 and brightness_score < 0.7):
                    passes = False
                elif lighting == 'perfect' and (brightness_score < 0.4 or brightness_score > 0.65):
                    passes = False
                
                # Color mode filter (black & white vs color)
                if color_mode:
                    vibrancy = self._calc_vibrancy(img_rgb)
                    if color_mode in ['bw', 'monochrome', 'black_and_white', 'grayscale']:
                        # Black and white = very low saturation/vibrancy
                        if vibrancy > 0.15:
                            passes = False
                    elif color_mode == 'color':
                        # Color images = higher saturation
                        if vibrancy < 0.1:
                            passes = False
                
                if passes:
                    # Add metrics to candidate
                    filtered.append({
                        **candidate,
                        'cv_metrics': {
                            'brightness_score': brightness_score,
                            'sharpness_score': sharpness_score,
                            'quality_score': quality_score
                        }
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {image_path}: {e}")
                # On error, include the candidate (conservative)
                filtered.append(candidate)
        
        self.logger.info(f"CV filter: {len(filtered)}/{len(candidates)} passed")
        
        return {
            "results": filtered,
            "filtered_count": len(filtered),
            "original_count": len(candidates)
        }
    
    async def _calculate_training_quality(
        self,
        image_paths: List[str],
        include_sharpness: bool = True,
        include_exposure: bool = True,
        include_composition: bool = True
    ) -> Dict[str, Any]:
        """
        SOTA: Calculate training quality score for each image
        
        Combines multiple metrics into a composite score that indicates
        how suitable an image is for model training:
        - Sharpness: Blurry images harm model quality
        - Exposure: Over/underexposed images introduce bias
        - Composition: Centered, well-framed subjects train better
        - Complexity: Moderate complexity is ideal
        
        Returns scores and recommendations for each image.
        """
        self.logger.info(f"Calculating training quality for {len(image_paths)} images")
        
        results = []
        
        for image_path in image_paths:
            try:
                img = self._load_image_with_exif(image_path)
                if img is None:
                    results.append({
                        "path": image_path,
                        "training_score": 0,
                        "issues": ["Failed to load image"],
                        "recommendation": "remove"
                    })
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                issues = []
                scores = {}
                
                # 1. Sharpness score (0-1)
                if include_sharpness:
                    sharpness = self._calc_sharpness(img)
                    scores['sharpness'] = sharpness
                    if sharpness < 0.3:
                        issues.append("Very blurry - will degrade model quality")
                    elif sharpness < 0.5:
                        issues.append("Slightly blurry")
                
                # 2. Exposure score (0-1, 0.5 is ideal)
                if include_exposure:
                    brightness = self._calc_brightness(img)
                    # Convert to exposure score (distance from ideal 0.5)
                    exposure_score = 1 - abs(brightness - 0.5) * 2
                    scores['exposure'] = exposure_score
                    if brightness > 0.85:
                        issues.append("Overexposed - loss of detail")
                    elif brightness < 0.15:
                        issues.append("Underexposed - loss of detail")
                
                # 3. Composition score (focal point position)
                if include_composition:
                    # Simple composition check: is the focal point centered or rule-of-thirds?
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                    
                    # Find brightest region (likely subject)
                    kernel_size = max(img.shape[0], img.shape[1]) // 10
                    blurred = cv2.GaussianBlur(gray, (kernel_size | 1, kernel_size | 1), 0)
                    _, _, _, max_loc = cv2.minMaxLoc(blurred)
                    
                    # Calculate position relative to image
                    h, w = gray.shape
                    x_rel = max_loc[0] / w  # 0 to 1
                    y_rel = max_loc[1] / h  # 0 to 1
                    
                    # Ideal: center (0.5) or rule-of-thirds (0.33, 0.67)
                    x_score = max(
                        1 - abs(x_rel - 0.5) * 2,  # Center
                        1 - abs(x_rel - 0.33) * 3,  # Left third
                        1 - abs(x_rel - 0.67) * 3   # Right third
                    )
                    y_score = max(
                        1 - abs(y_rel - 0.5) * 2,
                        1 - abs(y_rel - 0.33) * 3,
                        1 - abs(y_rel - 0.67) * 3
                    )
                    composition_score = (x_score + y_score) / 2
                    scores['composition'] = max(0, min(1, composition_score))
                    
                    if composition_score < 0.3:
                        issues.append("Subject position may not be ideal for training")
                
                # 4. Complexity score (moderate is best)
                complexity = self._calc_complexity(img)
                scores['complexity'] = complexity
                if complexity < 0.1:
                    issues.append("Very low complexity - may be too simple")
                elif complexity > 0.9:
                    issues.append("Very high complexity - may be too noisy")
                
                # Calculate composite training score
                weights = {
                    'sharpness': 0.35,
                    'exposure': 0.25,
                    'composition': 0.20,
                    'complexity': 0.20
                }
                
                total_weight = sum(weights[k] for k in scores.keys())
                training_score = sum(scores[k] * weights[k] for k in scores.keys()) / total_weight
                
                # Determine recommendation
                if training_score >= 0.7:
                    recommendation = "keep"
                elif training_score >= 0.4:
                    recommendation = "review"
                else:
                    recommendation = "remove"
                
                results.append({
                    "path": image_path,
                    "training_score": round(training_score, 3),
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                    "issues": issues,
                    "recommendation": recommendation
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate training quality for {image_path}: {e}")
                results.append({
                    "path": image_path,
                    "training_score": 0,
                    "issues": [f"Error: {str(e)}"],
                    "recommendation": "review"
                })
        
        # Summary statistics
        scores_only = [r['training_score'] for r in results if r['training_score'] > 0]
        avg_score = sum(scores_only) / len(scores_only) if scores_only else 0
        
        return {
            "results": results,
            "summary": {
                "total_images": len(results),
                "average_score": round(avg_score, 3),
                "keep_count": len([r for r in results if r['recommendation'] == 'keep']),
                "review_count": len([r for r in results if r['recommendation'] == 'review']),
                "remove_count": len([r for r in results if r['recommendation'] == 'remove'])
            }
        }


# Main entry point
async def main():
    """Run CV Engine as a service"""
    engine = CVEngine()
    
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

