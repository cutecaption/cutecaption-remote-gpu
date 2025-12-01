"""
NeMo Quality Filter - Lightweight wrapper for NeMo Curator's quality models
Uses pre-trained aesthetic and NSFW classifiers on CLIP embeddings

This is a MINIMAL integration - just the quality scoring models, not the full pipeline
"""

import sys
import torch
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine


class NeMoQualityFilter(BaseEngine):
    """
    Lightweight quality filter using NeMo Curator's pre-trained models
    
    Takes CLIP embeddings as input, returns quality scores
    No need for full NeMo Curator pipeline - just the models!
    """
    
    def __init__(self):
        super().__init__("nemo_quality_filter", "1.0.0")
        
        self.aesthetic_model = None
        self.nsfw_model = None
        self.device = None
        self.nemo_available = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize NeMo quality models"""
        try:
            self.logger.info("Initializing NeMo Quality Filter...")
            
            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            # Load NeMo's pre-trained quality models (standalone, no full NeMo required)
            try:
                import torch.nn as nn
                
                # Simple MLP architecture (matches NeMo's models)
                class QualityMLP(nn.Module):
                    def __init__(self, input_dim=768, hidden_dim=512, output_dim=1):
                        super().__init__()
                        self.network = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_dim, output_dim),
                            nn.Sigmoid()  # Output 0-1
                        )
                    
                    def forward(self, x):
                        return self.network(x)
                
                # Try to load pre-trained weights
                aesthetic_weights = Path('./models/nemo_filters/aesthetic_predictor.pt')
                nsfw_weights = Path('./models/nemo_filters/nsfw_detector.pt')
                
                if aesthetic_weights.exists():
                    self.aesthetic_model = QualityMLP().to(self.device)
                    self.aesthetic_model.load_state_dict(torch.load(aesthetic_weights, map_location=self.device))
                    self.aesthetic_model.eval()
                    self.logger.info("✓ Aesthetic quality model loaded")
                else:
                    self.logger.warning("Aesthetic model not found - will use fallback")
                
                if nsfw_weights.exists():
                    self.nsfw_model = QualityMLP().to(self.device)
                    self.nsfw_model.load_state_dict(torch.load(nsfw_weights, map_location=self.device))
                    self.nsfw_model.eval()
                    self.logger.info("✓ NSFW detector loaded")
                else:
                    self.logger.warning("NSFW model not found - will use fallback")
                
                self.nemo_available = (self.aesthetic_model is not None or self.nsfw_model is not None)
                
            except Exception as e:
                self.logger.warning(f"NeMo models not available: {e} - using fallback scores")
                self.nemo_available = False
            
            self._mark_initialized()
            
            return {
                "nemo_available": self.nemo_available,
                "aesthetic_model": self.aesthetic_model is not None,
                "nsfw_model": self.nsfw_model is not None,
                "device": self.device
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self._mark_unhealthy(str(e))
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process quality scoring requests"""
        action = request.get('action')
        
        if action == 'health_check':
            return await self.health_check()
        
        elif action == 'score_embeddings':
            return await self._score_embeddings(
                request['embeddings'],
                request.get('include_aesthetic', True),
                request.get('include_nsfw', True)
            )
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _score_embeddings(
        self, 
        embeddings: List[List[float]], 
        include_aesthetic: bool,
        include_nsfw: bool
    ) -> Dict[str, Any]:
        """
        Score CLIP embeddings for quality
        
        Args:
            embeddings: List of 768-dim CLIP embeddings
            include_aesthetic: Calculate aesthetic scores
            include_nsfw: Calculate NSFW scores
        
        Returns:
            Dictionary with scores for each embedding
        """
        try:
            results = {
                "aesthetic_scores": [],
                "nsfw_scores": [],
                "count": len(embeddings)
            }
            
            if not self.nemo_available:
                # Fallback: Return neutral scores
                self.logger.info("NeMo not available, returning fallback scores")
                results["aesthetic_scores"] = [0.5] * len(embeddings)
                results["nsfw_scores"] = [0.0] * len(embeddings)
                results["method"] = "fallback"
                return results
            
            # Convert to tensor
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            
            # Run aesthetic model
            if include_aesthetic and self.aesthetic_model:
                with torch.no_grad():
                    aesthetic_scores = self.aesthetic_model(embeddings_tensor)
                    results["aesthetic_scores"] = aesthetic_scores.cpu().numpy().tolist()
            
            # Run NSFW detector
            if include_nsfw and self.nsfw_model:
                with torch.no_grad():
                    nsfw_scores = self.nsfw_model(embeddings_tensor)
                    results["nsfw_scores"] = nsfw_scores.cpu().numpy().tolist()
            
            results["method"] = "nemo_curator"
            return results
            
        except Exception as e:
            self.logger.error(f"Quality scoring failed: {e}")
            # Graceful degradation
            return {
                "aesthetic_scores": [0.5] * len(embeddings),
                "nsfw_scores": [0.0] * len(embeddings),
                "count": len(embeddings),
                "method": "error_fallback",
                "error": str(e)
            }


# Main entry point
async def main():
    """Run NeMo Quality Filter as a service"""
    engine = NeMoQualityFilter()
    
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
    import json
    asyncio.run(main())

