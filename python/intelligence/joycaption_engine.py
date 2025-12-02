"""
JoyCaption Engine - Caption Service for Remote GPU Server
Generates image captions using the JoyCaption model (llama-joycaption)

Inherits from BaseEngine for standardized interface
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add parent directory to path for base_engine import
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine

# Lazy imports for heavy dependencies
torch = None
Image = None
AutoProcessor = None
LlavaForConditionalGeneration = None


class JoyCaptionEngine(BaseEngine):
    """
    JoyCaption Engine - Expert service for image captioning
    
    Uses fancyfeast/llama-joycaption-beta-one-hf-llava model
    Optimized for high-quality image descriptions
    """
    
    def __init__(self):
        super().__init__("joycaption_engine", "1.0.0")
        
        self.model = None
        self.processor = None
        self.device = None
        self.model_name = "fancyfeast/llama-joycaption-beta-one-hf-llava"
        
        # Default generation settings
        self.default_settings = {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 512,
            'do_sample': True
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the JoyCaption Engine
        Load the model and processor
        """
        try:
            self.logger.info("Initializing JoyCaption Engine...")
            
            # Import heavy dependencies
            global torch, Image, AutoProcessor, LlavaForConditionalGeneration
            import torch as _torch
            from PIL import Image as _Image
            from transformers import AutoProcessor as _AutoProcessor
            from transformers import LlavaForConditionalGeneration as _LlavaForConditionalGeneration
            
            torch = _torch
            Image = _Image
            AutoProcessor = _AutoProcessor
            LlavaForConditionalGeneration = _LlavaForConditionalGeneration
            
            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            # Load processor
            self.logger.info(f"Loading processor from {self.model_name}...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model
            self.logger.info("Loading JoyCaption model (this may take a while)...")
            
            if self.device == "cuda":
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            self.is_initialized = True
            self.health_status = "healthy"
            self.initialization_time = time.time()
            
            self.logger.info("JoyCaption Engine initialized successfully!")
            
            return {
                "success": True,
                "service": self.service_name,
                "version": self.version,
                "device": self.device,
                "model": self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize JoyCaption Engine: {e}", exc_info=True)
            self.health_status = "error"
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a caption request
        
        Supported actions:
        - caption_image: Generate caption for a single image
        """
        action = request.get('action')
        
        if action == 'caption_image':
            return await self._caption_image(
                request['image_path'],
                request.get('prompt'),
                request.get('temperature', self.default_settings['temperature']),
                request.get('top_p', self.default_settings['top_p']),
                request.get('max_tokens', self.default_settings['max_tokens'])
            )
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _caption_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Generate a caption for an image
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with caption and metadata
        """
        try:
            start_time = time.time()
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Build conversation
            if prompt:
                user_message = prompt
            else:
                user_message = "Write a descriptive caption for this image."
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_message}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    use_cache=True
                )
            
            # Decode
            generated_text = self.processor.decode(
                output[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            elapsed = time.time() - start_time
            
            self.logger.info(f"Caption generated in {elapsed:.2f}s: {generated_text[:50]}...")
            
            return {
                "success": True,
                "caption": generated_text,
                "image_path": image_path,
                "generation_time": elapsed,
                "model": self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Caption generation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def get_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "service": self.service_name,
            "status": self.health_status,
            "initialized": self.is_initialized,
            "device": self.device,
            "model": self.model_name if self.model else None
        }


# For standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        engine = JoyCaptionEngine()
        await engine.initialize()
        
        # Test with a sample image
        result = await engine.process({
            'action': 'caption_image',
            'image_path': '/path/to/test/image.jpg'
        })
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())

