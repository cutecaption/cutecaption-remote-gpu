"""
Text Engine - OCR and Text Detection Expert Service
Handles text detection, OCR, and text-based image analysis

Capabilities:
- Text detection (bounding boxes)
- OCR (text extraction)
- Text quantity/density analysis
- Language detection
- Text readability assessment
- Font style classification (future)

Uses lightweight models (PaddleOCR or EasyOCR) for fast processing

Author: Intelligence Engine Team
Version: 1.0
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine


class TextEngine(BaseEngine):
    """
    Expert service for text detection and OCR
    
    Generates text profiles for images including:
    - Detected text regions
    - Extracted text content
    - Text density/quantity
    - Language detection
    - Readability scores
    """
    
    def __init__(self):
        super().__init__("text_engine", "1.0.0")
        
        # Caption cache (loaded from .txt files)
        self.captions = {}  # filename -> caption text
        self.dataset_path = None
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Text Engine"""
        try:
            self.logger.info("Initializing Text Engine (Caption Search)...")
            self._mark_initialized()
            
            return {
                "backend": "caption_search",
                "ocr_available": False,
                "caption_search": True
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self._mark_unhealthy(str(e))
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process text search requests"""
        action = request.get('action')
        
        if action == 'health_check':
            return await self.health_check()
        
        elif action == 'search_captions':
            return await self._search_captions(
                request['dataset_path'],
                request['query'],
                request.get('exact_match', False)
            )
        
        elif action == 'load_captions':
            return await self._load_captions(request['dataset_path'])
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _load_captions(self, dataset_path: str) -> Dict[str, Any]:
        """Load all captions from .txt files"""
        try:
            import os
            self.dataset_path = dataset_path
            self.captions = {}
            loaded_count = 0
            
            for filename in os.listdir(dataset_path):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    caption_file = os.path.join(dataset_path, filename.rsplit('.', 1)[0] + '.txt')
                    if os.path.exists(caption_file):
                        # Try UTF-8 first, fallback to latin-1 (handles most edge cases)
                        try:
                            with open(caption_file, 'r', encoding='utf-8') as f:
                                self.captions[filename] = f.read().strip()
                                loaded_count += 1
                        except UnicodeDecodeError:
                            try:
                                with open(caption_file, 'r', encoding='latin-1') as f:
                                    self.captions[filename] = f.read().strip()
                                    loaded_count += 1
                                    self.logger.debug(f"Loaded caption with latin-1 fallback: {filename}")
                            except Exception as e:
                                self.logger.warning(f"Failed to load caption {caption_file}: {e}")
            
            return {
                "loaded": loaded_count,
                "total_files": len(os.listdir(dataset_path))
            }
        except Exception as e:
            self.logger.error(f"Failed to load captions: {e}")
            raise
    
    async def _search_captions(self, dataset_path: str, query: str, exact_match: bool) -> Dict[str, Any]:
        """Search captions for keyword"""
        try:
            # Load captions if not loaded
            if not self.captions or self.dataset_path != dataset_path:
                await self._load_captions(dataset_path)
            
            query_lower = query.lower()
            matches = []
            
            for filename, caption in self.captions.items():
                caption_lower = caption.lower()
                
                if exact_match:
                    if query_lower in caption_lower:
                        matches.append({
                            "filename": filename,
                            "caption": caption,
                            "match_type": "exact"
                        })
                else:
                    # Fuzzy match - check if any query words are in caption
                    query_words = query_lower.split()
                    if any(word in caption_lower for word in query_words):
                        matches.append({
                            "filename": filename,
                            "caption": caption,
                            "match_type": "partial"
                        })
            
            return {
                "matches": matches,
                "count": len(matches),
                "query": query
            }
        except Exception as e:
            self.logger.error(f"Caption search failed: {e}")
            raise
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_model(self) -> bool:
        """
        Lazy-load the OCR model
        
        Returns:
            bool: True if model loaded successfully
        """
        if self.model_loaded:
            return True
        
        try:
            self.log_info(f"Loading {self.ocr_backend} model...")
            
            if self.ocr_backend == 'paddleocr':
                # PaddleOCR integration - install with: pip install paddlepaddle paddleocr
                try:
                    from paddleocr import PaddleOCR
                    self.ocr_model = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        use_gpu=(self.device == 'cuda'),
                        show_log=False
                    )
                    self.log_info("PaddleOCR loaded successfully")
                except ImportError:
                    self.log_warning("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
                    self.ocr_model = None
                
            elif self.ocr_backend == 'easyocr':
                # EasyOCR integration - install with: pip install easyocr
                try:
                    import easyocr
                    self.ocr_model = easyocr.Reader(
                        self.languages,
                        gpu=(self.device == 'cuda')
                    )
                    self.log_info("EasyOCR loaded successfully")
                except ImportError:
                    self.log_warning("EasyOCR not installed. Install with: pip install easyocr")
                    self.ocr_model = None
                
            elif self.ocr_backend == 'tesseract':
                # Tesseract is CPU-only and requires system installation
                import pytesseract
                self.ocr_model = pytesseract  # Direct module reference
                
            else:
                raise ValueError(f"Unknown OCR backend: {self.ocr_backend}")
            
            self.model_loaded = True
            self.log_info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to load model: {str(e)}")
            return False
    
    def detect_text(self, image_path: str) -> Dict[str, Any]:
        """
        Detect and extract text from an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing:
                - success: bool
                - has_text: bool
                - text_regions: List of detected text bounding boxes
                - extracted_text: List of extracted text strings
                - confidences: Confidence scores for each detection
                - text_count: Number of text regions
                - metadata: Additional info
        """
        try:
            self.log_debug(f"Detecting text in: {image_path}")
            
            # Load model if needed
            if not self.load_model():
                return self._error_response("Model not loaded")
            
            # Load image
            if not os.path.exists(image_path):
                return self._error_response(f"Image not found: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                return self._error_response(f"Failed to load image: {image_path}")
            
            # Perform OCR - use Qwen VLM for text detection (our existing system)
            if self.ocr_backend == 'qwen' or not self.ocr_model:
                result = self._ocr_qwen_vlm(image_path)  # Use VLM for OCR
            elif self.ocr_backend == 'tesseract':
                result = self._ocr_tesseract(image)
            else:
                # Default to Qwen VLM (our primary OCR system)
                result = self._ocr_qwen_vlm(image_path)
            
            # Add metadata
            result['metadata'] = {
                'image_path': image_path,
                'image_shape': image.shape,
                'backend': self.ocr_backend
            }
            
            self.log_debug(f"Detected {result['text_count']} text region(s)")
            return result
            
        except Exception as e:
            self.log_error(f"Text detection failed: {str(e)}")
            return self._error_response(str(e))
    
    def _ocr_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform OCR using Tesseract
        
        Args:
            image: Image array (BGR)
            
        Returns:
            OCR results dictionary
        """
        try:
            import pytesseract
            from pytesseract import Output
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                image_rgb,
                output_type=Output.DICT
            )
            
            # Filter results by confidence
            text_regions = []
            extracted_text = []
            confidences = []
            
            for i in range(len(ocr_data['text'])):
                conf = float(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if conf >= self.min_confidence * 100 and text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    text_regions.append([x, y, x+w, y+h])
                    extracted_text.append(text)
                    confidences.append(conf / 100.0)  # Normalize to [0, 1]
            
            return {
                'success': True,
                'has_text': len(extracted_text) > 0,
                'text_regions': text_regions,
                'extracted_text': extracted_text,
                'confidences': confidences,
                'text_count': len(extracted_text)
            }
            
        except Exception as e:
            self.log_error(f"Tesseract OCR failed: {str(e)}")
            return self._error_response(str(e))
    
    def _ocr_placeholder(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Placeholder OCR implementation
        Returns empty results until real OCR backend is integrated
        
        Args:
            image: Image array (BGR)
            
        Returns:
            Empty OCR results
        """
        return {
            'success': True,
            'has_text': False,
            'text_regions': [],
            'extracted_text': [],
            'confidences': [],
            'text_count': 0
        }
    
    def analyze_text_profile(self, image_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive text profile for Dataset Profile
        
        Args:
            image_path: Path to image
            
        Returns:
            Text profile dictionary
        """
        detection = self.detect_text(image_path)
        
        if not detection['success']:
            return {
                'has_text': False,
                'text_count': 0,
                'text_density': 0.0,
                'primary_language': None,
                'extracted_text': [],
                'error': detection.get('error')
            }
        
        # Calculate text density (percentage of image covered by text)
        text_density = 0.0
        if detection['has_text']:
            image = cv2.imread(image_path)
            image_area = image.shape[0] * image.shape[1]
            
            text_area = 0
            for bbox in detection['text_regions']:
                x1, y1, x2, y2 = bbox
                text_area += (x2 - x1) * (y2 - y1)
            
            text_density = text_area / image_area if image_area > 0 else 0.0
        
        # Detect primary language (simplified)
        primary_language = self._detect_language(detection['extracted_text'])
        
        return {
            'has_text': detection['has_text'],
            'text_count': detection['text_count'],
            'text_density': float(text_density),
            'primary_language': primary_language,
            'extracted_text': detection['extracted_text'],
            'text_regions': detection['text_regions'],
            'confidences': detection['confidences']
        }
    
    def _detect_language(self, texts: List[str]) -> Optional[str]:
        """
        Detect primary language from extracted text
        
        Args:
            texts: List of extracted text strings
            
        Returns:
            Detected language code or None
        """
        if not texts:
            return None
        
        # Simple heuristic: combine all text and check character sets
        combined_text = ' '.join(texts)
        
        # TODO: Integrate proper language detection (langdetect, fasttext)
        # For now, assume English
        return 'en'
    
    def _ocr_qwen_vlm(self, image_path: str) -> Dict[str, Any]:
        """
        VLM-based OCR stub.
        
        NOTE: Direct VLM calls from TextEngine are not supported - VLM uses IPC not HTTP.
        For VLM-based text extraction, route through the orchestrator which manages
        the VLM service lifecycle and IPC communication.
        
        This method returns a result indicating VLM OCR should be requested at the
        orchestrator level, not from within TextEngine.
        
        Args:
            image_path: Path to image file
            
        Returns:
            OCR result dictionary indicating VLM routing needed
        """
        self.log_debug(f"VLM OCR requested for {image_path} - route through orchestrator for VLM-based text extraction")
        
        # Return result that indicates text detection should use VLM via orchestrator
        # The orchestrator can call VLMService.answer_question with an OCR prompt
        return {
            'success': True,
            'text_regions': [],
            'full_text': '',
            'confidence': 0.0,
            'backend': 'vlm_deferred',
            'requires_vlm': True,
            'image_path': image_path,
            'message': 'VLM-based OCR available via orchestrator - use VLMService.answer_question with OCR prompt'
        }
    
    def _ocr_fallback(self) -> Dict[str, Any]:
        """
        Fallback OCR when VLM is unavailable
        Returns minimal result indicating no text detected
        """
        return {
            'success': True,
            'text_regions': [],
            'full_text': '',
            'confidence': 0.0,
            'backend': 'fallback',
            'message': 'VLM service unavailable, no text detected'
        }
    
    def batch_detect(
        self,
        image_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Detect text in multiple images (batch processing)
        
        Args:
            image_paths: List of image file paths
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping image paths to text detection results
        """
        self.log_info(f"Batch text detection for {len(image_paths)} images")
        
        results = {}
        total = len(image_paths)
        
        for idx, image_path in enumerate(image_paths):
            result = self.analyze_text_profile(image_path)
            results[image_path] = result
            
            # Progress reporting
            if progress_callback:
                progress_callback(idx + 1, total)
            
            self.report_progress(
                current=idx + 1,
                total=total,
                message=f"Processed {idx + 1}/{total} images"
            )
        
        self.log_info(f"Batch processing complete: {len(results)} results")
        
        return {
            'success': True,
            'total_processed': len(results),
            'results': results
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the text engine
        
        Returns:
            Health status dictionary
        """
        return {
            'service': 'TextEngine',
            'status': 'healthy' if self.model_loaded else 'model_not_loaded',
            'model_loaded': self.model_loaded,
            'backend': self.ocr_backend,
            'device': self.device,
            'languages': self.languages,
            'min_confidence': self.min_confidence
        }
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Helper to create error response"""
        return {
            'success': False,
            'error': message,
            'has_text': False,
            'text_regions': [],
            'extracted_text': [],
            'confidences': [],
            'text_count': 0
        }


async def main():
    """Run Text Engine as a service"""
    engine = TextEngine()
    
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
                engine.logger.error(f"Request processing error: {e}")
                print(json.dumps({
                    "success": False,
                    "error": {"message": str(e)}
                }), flush=True)
    except Exception as e:
        engine.logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

