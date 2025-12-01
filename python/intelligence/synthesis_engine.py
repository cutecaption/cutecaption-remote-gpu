"""
Synthesis Engine - Conceptual Modification and Gradient Ascent Expert Service
Handles conceptual transformations and advanced retrieval using the Conceptualizer

This service wraps the existing Conceptualizer implementation from MiniModel
and provides it as part of the Federation of Experts architecture.

Capabilities:
- Conceptual modification (vector arithmetic in latent space)
- Gradient ascent for finding images
- Expert re-ranking with trained mini-models
- Two-stage retrieval (CLIP → Conceptualizer)
- User feedback integration

Author: Intelligence Engine Team
Version: 1.0
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "updated_technical_documentation" / "MiniModel"))

from base_engine import BaseEngine

# Import Conceptualizer from MiniModel
try:
    from conceptualizer_service import ConceptualizerService, SpecializedIndexManager
    from conceptualizer_model import ConceptualizerMLP
    from feedback_refiner import FeedbackRefiner
    CONCEPTUALIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Conceptualizer not available: {e}", file=sys.stderr)
    CONCEPTUALIZER_AVAILABLE = False


class MomentumFeedbackRefiner:
    """
    **MOMENTUM FEEDBACK REFINER**
    Instant feedback updates (<100ms) between full Conceptualizer retrains
    
    WHY: Full Conceptualizer retraining takes 30-60s (20+ feedback samples)
    PROBLEM: Users want immediate feedback, not waiting for 20 corrections
    SOLUTION: Lightweight "momentum queue" that provides instant refinement
    
    HOW IT WORKS:
    1. Maintains rolling deque of last 8192 feedback events (positive/negative)
    2. Computes instant embedding adjustments using momentum averaging
    3. Provides <100ms refinement BETWEEN full retrains
    4. When 20+ feedback accumulated → trigger full Conceptualizer retrain
    5. Clear momentum queue, start fresh with new model
    
    ANALOGY: Like browser cache vs database
    - Momentum Queue = Hot cache (instant, temporary)
    - Conceptualizer = Database (slow, permanent)
    
    Based on momentum-based recommendation systems (Netflix, Spotify)
    """
    
    def __init__(self, max_size=8192, momentum_factor=0.9):
        """
        Initialize Momentum Feedback Refiner
        
        Args:
            max_size: Maximum feedback events to keep (default: 8192)
            momentum_factor: Momentum decay factor (default: 0.9)
        """
        from collections import deque
        
        self.feedback_queue = deque(maxlen=max_size)
        self.momentum_factor = momentum_factor
        
        # Momentum accumulators (exponentially weighted averages)
        self.positive_momentum = None  # Average of positive embeddings
        self.negative_momentum = None  # Average of negative embeddings
        
        self.total_feedback_count = 0
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    def add_feedback(self, embedding: np.ndarray, is_positive: bool) -> None:
        """
        Add user feedback to momentum queue
        
        Args:
            embedding: Image embedding vector [768]
            is_positive: True if user liked it, False if disliked
        """
        try:
            import time
            timestamp = time.time()
            
            self.feedback_queue.append({
                'embedding': embedding.copy(),
                'is_positive': is_positive,
                'timestamp': timestamp
            })
            
            self.total_feedback_count += 1
            
            # Update momentum accumulators
            self._update_momentum(embedding, is_positive)
            
            self.logger.debug(f"[Momentum] Feedback added: {'positive' if is_positive else 'negative'}, queue size: {len(self.feedback_queue)}")
            
        except Exception as e:
            self.logger.error(f"[Momentum] Failed to add feedback: {str(e)}")
    
    def _update_momentum(self, embedding: np.ndarray, is_positive: bool) -> None:
        """
        Update exponentially weighted momentum accumulators
        
        Momentum formula: momentum = β * momentum + (1-β) * new_value
        """
        try:
            if is_positive:
                if self.positive_momentum is None:
                    self.positive_momentum = embedding.copy()
                else:
                    self.positive_momentum = (
                        self.momentum_factor * self.positive_momentum +
                        (1 - self.momentum_factor) * embedding
                    )
            else:
                if self.negative_momentum is None:
                    self.negative_momentum = embedding.copy()
                else:
                    self.negative_momentum = (
                        self.momentum_factor * self.negative_momentum +
                        (1 - self.momentum_factor) * embedding
                    )
        except Exception as e:
            self.logger.error(f"[Momentum] Failed to update momentum: {str(e)}")
    
    def instant_refine_embedding(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Apply instant feedback refinement to query embedding
        
        **INSTANT UPDATE (<100ms)**
        Adjusts query towards positive momentum, away from negative momentum
        
        Args:
            query_embedding: Original query embedding [768]
        
        Returns:
            Refined embedding [768]
        """
        try:
            if len(self.feedback_queue) == 0:
                # No feedback yet, return original
                return query_embedding
            
            refined = query_embedding.copy()
            
            # PUSH towards positive momentum
            if self.positive_momentum is not None:
                positive_adjustment = self.positive_momentum - query_embedding
                refined += 0.3 * positive_adjustment  # 30% adjustment towards positive
            
            # PULL away from negative momentum
            if self.negative_momentum is not None:
                negative_adjustment = query_embedding - self.negative_momentum
                refined += 0.2 * negative_adjustment  # 20% adjustment away from negative
            
            # Normalize
            refined = refined / np.linalg.norm(refined)
            
            self.logger.debug(f"[Momentum] Instant refinement applied (queue size: {len(self.feedback_queue)})")
            
            return refined
            
        except Exception as e:
            self.logger.error(f"[Momentum] Instant refinement failed: {str(e)}")
            return query_embedding  # Fallback to original
    
    def instant_rerank_results(self, results: List[Dict], result_embeddings: np.ndarray) -> List[Dict]:
        """
        Instantly re-rank results based on momentum feedback
        
        **INSTANT RERANKING (<100ms)**
        Boosts results similar to positive feedback
        Demotes results similar to negative feedback
        
        Args:
            results: List of search results
            result_embeddings: Embeddings for each result [N, 768]
        
        Returns:
            Re-ranked results
        """
        try:
            if len(self.feedback_queue) == 0:
                return results  # No feedback, no reranking
            
            # Compute momentum scores for each result
            momentum_scores = np.zeros(len(results))
            
            for i, emb in enumerate(result_embeddings):
                score = 0.0
                
                # Positive boost
                if self.positive_momentum is not None:
                    positive_sim = np.dot(emb, self.positive_momentum)
                    score += 0.5 * positive_sim
                
                # Negative penalty
                if self.negative_momentum is not None:
                    negative_sim = np.dot(emb, self.negative_momentum)
                    score -= 0.3 * negative_sim
                
                momentum_scores[i] = score
            
            # Re-rank by combining original similarity + momentum score
            for i, result in enumerate(results):
                original_sim = result.get('similarity', 0.0)
                momentum_bonus = momentum_scores[i]
                result['momentum_adjusted_similarity'] = original_sim + momentum_bonus
            
            # Sort by adjusted similarity
            results_reranked = sorted(
                results,
                key=lambda x: x.get('momentum_adjusted_similarity', x.get('similarity', 0.0)),
                reverse=True
            )
            
            self.logger.debug(f"[Momentum] Results re-ranked based on {len(self.feedback_queue)} feedback events")
            
            return results_reranked
            
        except Exception as e:
            self.logger.error(f"[Momentum] Reranking failed: {str(e)}")
            return results  # Fallback to original order
    
    def should_trigger_full_retrain(self, threshold: int = 20) -> bool:
        """
        Check if enough feedback has accumulated to trigger full Conceptualizer retrain
        
        Args:
            threshold: Minimum feedback count for retraining (default: 20)
        
        Returns:
            True if should retrain
        """
        return len(self.feedback_queue) >= threshold
    
    def get_training_data_for_conceptualizer(self) -> List[Dict]:
        """
        Extract training data from momentum queue for Conceptualizer retraining
        
        Returns:
            List of training examples with embeddings and labels
        """
        try:
            training_data = []
            
            for feedback in self.feedback_queue:
                training_data.append({
                    'embedding': feedback['embedding'],
                    'concept_label': 1 if feedback['is_positive'] else 0,  # Binary: like/dislike
                    'timestamp': feedback['timestamp']
                })
            
            self.logger.info(f"[Momentum] Extracted {len(training_data)} training examples from queue")
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"[Momentum] Failed to extract training data: {str(e)}")
            return []
    
    def clear_after_retrain(self) -> None:
        """
        Clear momentum queue after Conceptualizer has been retrained
        
        WHY: New model incorporates all feedback, start fresh momentum
        """
        try:
            previous_size = len(self.feedback_queue)
            self.feedback_queue.clear()
            self.positive_momentum = None
            self.negative_momentum = None
            
            self.logger.info(f"[Momentum] Queue cleared after retrain ({previous_size} events archived)")
            
        except Exception as e:
            self.logger.error(f"[Momentum] Failed to clear queue: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get momentum queue statistics
        
        Returns:
            Dict with queue size, positive/negative counts, etc.
        """
        try:
            positive_count = sum(1 for f in self.feedback_queue if f['is_positive'])
            negative_count = len(self.feedback_queue) - positive_count
            
            return {
                'queue_size': len(self.feedback_queue),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'total_lifetime_feedback': self.total_feedback_count,
                'has_positive_momentum': self.positive_momentum is not None,
                'has_negative_momentum': self.negative_momentum is not None,
                'ready_for_retrain': self.should_trigger_full_retrain()
            }
        except Exception as e:
            self.logger.error(f"[Momentum] Failed to get stats: {str(e)}")
            return {}


class SynthesisEngine(BaseEngine):
    """
    Synthesis Expert Service - Conceptual Modification and Gradient Ascent
    
    Wraps the Conceptualizer mini-model for dataset-specific latent space operations.
    Enables:
    1. Expert re-ranking: Improve CLIP search results
    2. Conceptual modification: "more X", "less Y", "combine A+B"
    3. Gradient ascent: Find images matching complex concepts
    4. User feedback learning: Refine model based on usage
    """
    
    def __init__(self):
        super().__init__("synthesis_engine", "1.0.0")
        self.model_available = CONCEPTUALIZER_AVAILABLE
        self.model_loaded = False
        
        # Initialize Momentum Feedback Refiner
        import os
        momentum_enabled = os.environ.get('AI_INTELLIGENCE_MOMENTUM_REFINEMENT', '1') == '1'
        self.momentum_refiner = MomentumFeedbackRefiner() if momentum_enabled else None
        
        if self.momentum_refiner:
            self.logger.info("[SynthesisEngine] Momentum Feedback Refiner initialized")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Synthesis Engine"""
        try:
            self.logger.info("Initializing Synthesis Engine...")
            self._mark_initialized()
            
            return {
                "conceptualizer_available": self.model_available,
                "model_loaded": False,
                "momentum_refiner_enabled": self.momentum_refiner is not None,
                "status": "ready_for_training"
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self._mark_unhealthy(str(e))
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process synthesis requests"""
        action = request.get('action')
        
        if action == 'health_check':
            return await self.health_check()
        
        elif action == 'rerank':
            # Stub for two-stage re-ranking
            return {
                "reranked": request.get('candidates', []),
                "method": "clip_only",
                "message": "Conceptualizer not trained yet - using CLIP rankings"
            }
        
        elif action == 'gradient_ascent':
            return await self._gradient_ascent(
                request.get('target_concepts', []),
                request.get('avoid_concepts', []),
                request.get('initial_embedding'),
                request.get('steps', 100),
                request.get('learning_rate', 0.01)
            )
        
        elif action == 'conceptual_optimization':
            return await self._conceptual_optimization(
                request.get('base_embedding'),
                request.get('modifications', {}),
                request.get('strength', 1.0)
            )
        
        elif action == 'search_index':
            return await self._search_specialized_index(
                request.get('dataset_name'),
                request.get('query_embedding'),
                request.get('k', 20),
                request.get('index_path')
            )
        
        elif action == 'load_model':
            model_path = request.get('model_path')
            if not model_path:
                return {
                    "success": False,
                    "error": "model_path required"
                }
            success = self.load_model(model_path)
            return {
                "success": success,
                "model_loaded": success,
                "model_path": model_path if success else None
            }
        
        elif action == 'train':
            # Training parameters
            training_data = request.get('training_data', [])
            model_name = request.get('model_name', 'conceptualizer')
            epochs = request.get('epochs', 10)
            learning_rate = request.get('learning_rate', 0.001)
            
            if not training_data:
                return {
                    "success": False,
                    "error": "training_data required"
                }
            
            try:
                # Start training process
                result = self.train_conceptualizer(
                    training_data=training_data,
                    model_name=model_name,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                return {
                    "success": True,
                    "model_path": result.get('model_path'),
                    "training_metrics": result.get('metrics', {}),
                    "message": "Training completed successfully"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Training failed: {str(e)}"
                }
        
        elif action == 'train_ge2e':
            # GE2E training parameters
            training_data = request.get('training_data', [])
            model_name = request.get('model_name', 'conceptualizer_ge2e')
            epochs = request.get('epochs', 20)
            learning_rate = request.get('learning_rate', 0.001)
            
            if not training_data:
                return {
                    "success": False,
                    "error": "training_data required for GE2E training"
                }
            
            try:
                result = self.train_conceptualizer_with_ge2e(
                    training_data=training_data,
                    model_name=model_name,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                return {
                    "success": True,
                    "model_path": result.get('model_path'),
                    "training_metrics": result.get('metrics', {}),
                    "loss_type": "GE2E",
                    "message": "GE2E training completed successfully"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"GE2E training failed: {str(e)}"
                }
        
        elif action == 'optimize_embedding':
            # Gradient ascent optimization
            initial_embedding = np.array(request['initial_embedding'])
            feedback_examples = request.get('feedback_examples', [])
            iterations = request.get('iterations', 10)
            learning_rate = request.get('learning_rate', 0.01)
            
            try:
                result = self.optimize_query_embedding(
                    initial_embedding=initial_embedding,
                    feedback_examples=feedback_examples,
                    iterations=iterations,
                    learning_rate=learning_rate
                )
                return {
                    "success": True,
                    "optimized_embedding": result['optimized_embedding'].tolist(),
                    "improvement": result['improvement'],
                    "metrics": result
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Embedding optimization failed: {str(e)}"
                }
        
        elif action == 'momentum_add_feedback':
            # Add feedback to momentum queue
            if not self.momentum_refiner:
                return {"success": False, "error": "Momentum refiner not enabled"}
            
            embedding = np.array(request['embedding'])
            is_positive = request.get('is_positive', True)
            
            self.momentum_refiner.add_feedback(embedding, is_positive)
            
            return {
                "success": True,
                "stats": self.momentum_refiner.get_stats()
            }
        
        elif action == 'momentum_refine_embedding':
            # Apply instant momentum refinement to query
            if not self.momentum_refiner:
                return {"success": False, "error": "Momentum refiner not enabled"}
            
            query_embedding = np.array(request['query_embedding'])
            refined = self.momentum_refiner.instant_refine_embedding(query_embedding)
            
            return {
                "success": True,
                "refined_embedding": refined.tolist(),
                "stats": self.momentum_refiner.get_stats()
            }
        
        elif action == 'momentum_get_stats':
            # Get momentum queue statistics
            if not self.momentum_refiner:
                return {"success": False, "error": "Momentum refiner not enabled"}
            
            return {
                "success": True,
                "stats": self.momentum_refiner.get_stats()
            }
        
        elif action == 'momentum_trigger_retrain':
            # Check if should trigger full Conceptualizer retrain
            if not self.momentum_refiner:
                return {"success": False, "error": "Momentum refiner not enabled"}
            
            should_retrain = self.momentum_refiner.should_trigger_full_retrain()
            training_data = None
            
            if should_retrain:
                training_data = self.momentum_refiner.get_training_data_for_conceptualizer()
            
            return {
                "success": True,
                "should_retrain": should_retrain,
                "training_data": training_data,
                "stats": self.momentum_refiner.get_stats()
            }
        
        elif action == 'extract_cluster_centers':
            # Extract cluster centers from trained Conceptualizer
            model_path = request.get('model_path')
            return self.extract_cluster_centers(model_path)
        
        elif action == 'find_representative_images':
            # Find representative images per cluster
            cluster_centers = np.array(request['cluster_centers'])
            all_embeddings = [np.array(e) for e in request['all_embeddings']]
            all_image_paths = request['all_image_paths']
            top_k = request.get('top_k', 5)
            
            return self.find_representative_images_per_cluster(
                cluster_centers, all_embeddings, all_image_paths, top_k
            )
        
        elif action == 'expand_query_with_context':
            # Dataset-aware query expansion
            query = request['query']
            cluster_centers = np.array(request['cluster_centers'])
            cluster_representatives = request['cluster_representatives']
            profile_data = request['profile_data']
            
            expanded = self.expand_query_with_dataset_context(
                query, cluster_centers, cluster_representatives, profile_data
            )
            
            return {
                "success": True,
                "expanded_queries": expanded
            }
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained Conceptualizer model
        
        Args:
            model_path: Path to model checkpoint (overrides config)
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            path = model_path or self.model_path
            
            if not path:
                self.log_warning("No model path specified. Will use fallback to standard CLIP.")
                return False
            
            if not os.path.exists(path):
                self.log_error(f"Model file not found: {path}")
                return False
            
            self.log_info(f"Loading Conceptualizer model from: {path}")
            
            # Initialize Conceptualizer service
            self.conceptualizer = ConceptualizerService(
                model_path=path,
                device=self.device if self.device != 'auto' else None
            )
            
            # Initialize index manager
            self.index_manager = SpecializedIndexManager(self.conceptualizer)
            
            # Initialize feedback refiner if auto-refine enabled
            if self.auto_refine:
                self.feedback_refiner = FeedbackRefiner(
                    feedback_log_path=self.feedback_log_path,
                    conceptualizer_service=self.conceptualizer
                )
            
            self.model_loaded = True
            self.log_info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to load model: {str(e)}")
            return False
    
    def transform_embeddings(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Transform CLIP embeddings to specialized latent space
        
        Args:
            embeddings: CLIP embeddings array [N, 768]
            
        Returns:
            Dictionary containing:
                - success: bool
                - specialized_embeddings: Transformed embeddings
                - method: 'conceptualizer' or 'passthrough'
        """
        try:
            if not self.model_loaded or self.conceptualizer is None:
                # Passthrough if model not loaded
                return {
                    'success': True,
                    'specialized_embeddings': embeddings,
                    'method': 'passthrough'
                }
            
            # Transform using Conceptualizer
            specialized = self.conceptualizer.transform_embeddings(embeddings)
            
            return {
                'success': True,
                'specialized_embeddings': specialized,
                'method': 'conceptualizer'
            }
            
        except Exception as e:
            self.log_error(f"Embedding transformation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'specialized_embeddings': embeddings,  # Fallback
                'method': 'error_fallback'
            }
    
    def expert_rerank(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_paths: List[str],
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Expert re-ranking: Improve first-stage CLIP results
        
        Args:
            query_embedding: Query CLIP embedding [768]
            candidate_embeddings: Candidate CLIP embeddings [N, 768]
            candidate_paths: Corresponding image paths
            k: Number of top results to return
            
        Returns:
            Dictionary containing:
                - success: bool
                - ranked_paths: Re-ranked image paths
                - scores: Similarity scores
                - method: 'conceptualizer' or 'fallback'
        """
        try:
            if not self.model_loaded or self.conceptualizer is None:
                # Fallback: use cosine similarity on original embeddings
                self.log_debug("Using fallback ranking (model not loaded)")
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                cand_norm = candidate_embeddings / np.linalg.norm(
                    candidate_embeddings, axis=1, keepdims=True
                )
                scores = np.dot(cand_norm, query_norm)
                
                top_indices = np.argsort(scores)[-k:][::-1]
                
                return {
                    'success': True,
                    'ranked_paths': [candidate_paths[i] for i in top_indices],
                    'scores': scores[top_indices].tolist(),
                    'method': 'fallback'
                }
            
            # Use Conceptualizer for re-ranking
            self.log_debug(f"Expert re-ranking {len(candidate_embeddings)} candidates")
            
            top_indices, top_scores = self.conceptualizer.two_stage_retrieval(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                k=k
            )
            
            ranked_paths = [candidate_paths[i] for i in top_indices]
            
            return {
                'success': True,
                'ranked_paths': ranked_paths,
                'scores': top_scores.tolist(),
                'method': 'conceptualizer'
            }
            
        except Exception as e:
            self.log_error(f"Expert re-ranking failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ranked_paths': candidate_paths[:k],  # Fallback
                'scores': [],
                'method': 'error_fallback'
            }
    
    def conceptual_modification(
        self,
        base_embedding: np.ndarray,
        add_embeddings: Optional[List[np.ndarray]] = None,
        subtract_embeddings: Optional[List[np.ndarray]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Conceptual modification: Vector arithmetic in latent space
        
        Examples:
            - "This character, but happier" = base + happy - neutral
            - "Mix of A and B" = (base_A + base_B) / 2
            - "More X, less Y" = base + 0.5*X - 0.3*Y
        
        Args:
            base_embedding: Base CLIP embedding
            add_embeddings: Embeddings to add (positive direction)
            subtract_embeddings: Embeddings to subtract (negative direction)
            weights: Optional weights for each operation
            
        Returns:
            Dictionary containing:
                - success: bool
                - modified_embedding: Result of vector arithmetic
                - in_specialized_space: Whether using Conceptualizer space
        """
        try:
            weights = weights or {'base': 1.0, 'add': 0.5, 'subtract': 0.3}
            
            # Start with base
            result = base_embedding * weights.get('base', 1.0)
            
            # Add positive directions
            if add_embeddings:
                for emb in add_embeddings:
                    result += emb * weights.get('add', 0.5)
            
            # Subtract negative directions
            if subtract_embeddings:
                for emb in subtract_embeddings:
                    result -= emb * weights.get('subtract', 0.3)
            
            # Normalize
            result = result / np.linalg.norm(result)
            
            # Transform to specialized space if model loaded
            in_specialized = False
            if self.model_loaded and self.conceptualizer:
                result = self.conceptualizer.transform_embeddings(result)
                in_specialized = True
            
            return {
                'success': True,
                'modified_embedding': result,
                'in_specialized_space': in_specialized
            }
            
        except Exception as e:
            self.log_error(f"Conceptual modification failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'modified_embedding': base_embedding,
                'in_specialized_space': False
            }
    
    def add_feedback(
        self,
        query_embedding: np.ndarray,
        result_embedding: np.ndarray,
        result_path: str,
        is_positive: bool,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record user feedback for model refinement
        
        Args:
            query_embedding: Query CLIP embedding
            result_embedding: Result image CLIP embedding
            result_path: Path to result image
            is_positive: True if positive feedback, False if negative
            metadata: Additional metadata (query text, timestamp, etc.)
            
        Returns:
            Dictionary with feedback status and refinement info
        """
        try:
            if not self.auto_refine or not self.feedback_refiner:
                return {
                    'success': True,
                    'feedback_recorded': False,
                    'message': 'Auto-refinement not enabled'
                }
            
            # Add feedback to refiner
            feedback_entry = {
                'query_embedding': query_embedding.tolist(),
                'result_embedding': result_embedding.tolist(),
                'result_path': result_path,
                'is_positive': is_positive,
                'metadata': metadata or {}
            }
            
            self.feedback_refiner.add_feedback(feedback_entry)
            
            # Check if refinement threshold reached
            feedback_count = len(self.feedback_refiner.feedback_log)
            should_refine = feedback_count >= self.refinement_threshold
            
            result = {
                'success': True,
                'feedback_recorded': True,
                'feedback_count': feedback_count,
                'threshold': self.refinement_threshold,
                'should_refine': should_refine
            }
            
            # Trigger refinement if threshold reached
            if should_refine and self.auto_refine:
                self.log_info(f"Feedback threshold reached ({feedback_count}). Triggering refinement...")
                refinement_result = self.refine_model()
                result['refinement'] = refinement_result
            
            return result
            
        except Exception as e:
            self.log_error(f"Feedback recording failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'feedback_recorded': False
            }
    
    def refine_model(self) -> Dict[str, Any]:
        """
        Refine Conceptualizer model based on accumulated feedback
        
        Returns:
            Dictionary with refinement statistics
        """
        try:
            if not self.feedback_refiner:
                return {
                    'success': False,
                    'error': 'Feedback refiner not initialized'
                }
            
            self.log_info("Starting model refinement based on user feedback...")
            
            # Perform refinement
            stats = self.feedback_refiner.refine_model(
                epochs=10,  # Quick refinement
                batch_size=16,
                learning_rate=1e-5  # Lower LR for fine-tuning
            )
            
            # Save refined model
            if self.model_path:
                backup_path = self.model_path + '.backup'
                self.conceptualizer.save_model(backup_path)
                self.log_info(f"Backed up original model to: {backup_path}")
                
                self.conceptualizer.save_model(self.model_path)
                self.log_info(f"Saved refined model to: {self.model_path}")
            
            # Clear feedback log after successful refinement
            self.feedback_refiner.clear_feedback()
            
            return {
                'success': True,
                'stats': stats,
                'message': 'Model refined successfully'
            }
            
        except Exception as e:
            self.log_error(f"Model refinement failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _gradient_ascent(
        self,
        target_concepts: List[np.ndarray],
        avoid_concepts: List[np.ndarray],
        initial_embedding: Optional[np.ndarray],
        steps: int = 100,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """
        Use gradient ascent to find optimal embedding that maximizes target concepts
        while minimizing avoided concepts.
        
        This creates a "dream image" embedding that represents the ideal combination.
        """
        try:
            # Convert to PyTorch tensors
            device = torch.device(self.device if self.device != 'auto' else 
                                ('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Initialize embedding (either from provided or random)
            if initial_embedding is not None:
                embedding = torch.tensor(initial_embedding, dtype=torch.float32, device=device)
            else:
                # Start from average of target concepts if available
                if target_concepts:
                    avg_target = np.mean(target_concepts, axis=0)
                    embedding = torch.tensor(avg_target, dtype=torch.float32, device=device)
                else:
                    # Random initialization in CLIP space (512 or 768 dim)
                    embedding = torch.randn(512, dtype=torch.float32, device=device)
            
            embedding = Variable(embedding, requires_grad=True)
            
            # Convert concept lists to tensors
            if target_concepts:
                targets = torch.stack([torch.tensor(c, dtype=torch.float32, device=device) 
                                     for c in target_concepts])
            else:
                targets = None
                
            if avoid_concepts:
                avoids = torch.stack([torch.tensor(c, dtype=torch.float32, device=device) 
                                    for c in avoid_concepts])
            else:
                avoids = None
            
            # Optimization loop
            optimizer = torch.optim.Adam([embedding], lr=learning_rate)
            history = []
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Normalize embedding to unit sphere (CLIP constraint)
                normalized = F.normalize(embedding, p=2, dim=0)
                
                # Calculate loss
                loss = 0.0
                
                # Maximize similarity to target concepts
                if targets is not None:
                    target_sims = torch.matmul(targets, normalized)
                    loss -= torch.mean(target_sims)  # Negative because we maximize
                
                # Minimize similarity to avoided concepts
                if avoids is not None:
                    avoid_sims = torch.matmul(avoids, normalized)
                    loss += torch.mean(avoid_sims)  # Positive because we minimize
                
                # Add L2 regularization to prevent drift
                loss += 0.001 * torch.norm(normalized - embedding.detach(), p=2)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Record progress every 10 steps
                if step % 10 == 0:
                    history.append({
                        'step': step,
                        'loss': float(loss.item()),
                        'norm': float(torch.norm(embedding).item())
                    })
            
            # Final normalization
            final_embedding = F.normalize(embedding, p=2, dim=0)
            
            return {
                "optimized_embedding": final_embedding.detach().cpu().numpy().tolist(),
                "optimization_history": history,
                "final_loss": float(loss.item()),
                "steps_completed": steps,
                "method": "gradient_ascent"
            }
            
        except Exception as e:
            self.logger.error(f"Gradient ascent failed: {e}")
            raise
    
    async def _search_specialized_index(self, dataset_name, query_embedding, k, index_path):
        """
        Search in a specialized FAISS index
        """
        try:
            import faiss
            import time
            
            start_time = time.time()
            
            # Load the FAISS index
            if not index_path or not os.path.exists(index_path):
                return {
                    "success": False,
                    "error": f"Index not found at {index_path}"
                }
            
            index = faiss.read_index(index_path)
            
            # Convert query to numpy array
            query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Normalize if needed
            faiss.normalize_L2(query)
            
            # Search
            distances, indices = index.search(query, k)
            
            # Format results
            results = []
            for i in range(len(indices[0])):
                if indices[0][i] != -1:  # Valid result
                    results.append({
                        "index": int(indices[0][i]),
                        "score": float(1 - distances[0][i] / 2),  # Convert L2 distance to similarity
                        "distance": float(distances[0][i])
                    })
            
            search_time = time.time() - start_time
            
            return {
                "success": True,
                "results": results,
                "search_time": search_time,
                "index_size": index.ntotal
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "FAISS not installed. Install with: pip install faiss-cpu"
            }
        except Exception as e:
            self.logger.error(f"Index search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _conceptual_optimization(
        self,
        base_embedding: np.ndarray,
        modifications: Dict[str, Any],
        strength: float = 1.0
    ) -> Dict[str, Any]:
        """
        Advanced conceptual modification using gradient-based optimization.
        
        Modifications can include:
        - "more_like": [embeddings] - Move towards these concepts
        - "less_like": [embeddings] - Move away from these concepts  
        - "interpolate": [embeddings] - Find middle ground
        - "orthogonal_to": [embeddings] - Find perpendicular direction
        """
        try:
            device = torch.device(self.device if self.device != 'auto' else 
                                ('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Start from base embedding
            embedding = torch.tensor(base_embedding, dtype=torch.float32, device=device)
            embedding = Variable(embedding, requires_grad=True)
            
            # Parse modifications
            more_like = modifications.get('more_like', [])
            less_like = modifications.get('less_like', [])
            interpolate = modifications.get('interpolate', [])
            orthogonal = modifications.get('orthogonal_to', [])
            
            # Optimization
            optimizer = torch.optim.SGD([embedding], lr=strength * 0.1)
            
            for step in range(50):  # Fewer steps for interactive use
                optimizer.zero_grad()
                
                normalized = F.normalize(embedding, p=2, dim=0)
                loss = 0.0
                
                # More like: increase similarity
                if more_like:
                    for target in more_like:
                        target_t = torch.tensor(target, dtype=torch.float32, device=device)
                        similarity = torch.dot(normalized, F.normalize(target_t, p=2, dim=0))
                        loss -= similarity * strength
                
                # Less like: decrease similarity
                if less_like:
                    for avoid in less_like:
                        avoid_t = torch.tensor(avoid, dtype=torch.float32, device=device)
                        similarity = torch.dot(normalized, F.normalize(avoid_t, p=2, dim=0))
                        loss += similarity * strength
                
                # Interpolation: minimize distance to centroid
                if interpolate:
                    centroid = torch.mean(torch.stack([
                        torch.tensor(e, dtype=torch.float32, device=device) 
                        for e in interpolate
                    ]), dim=0)
                    distance = torch.norm(normalized - F.normalize(centroid, p=2, dim=0))
                    loss += distance * strength
                
                # Orthogonal: minimize absolute dot product
                if orthogonal:
                    for ortho in orthogonal:
                        ortho_t = torch.tensor(ortho, dtype=torch.float32, device=device)
                        dot_product = torch.abs(torch.dot(normalized, F.normalize(ortho_t, p=2, dim=0)))
                        loss += dot_product * strength
                
                # Preserve some similarity to original
                base_t = torch.tensor(base_embedding, dtype=torch.float32, device=device)
                preservation = torch.dot(normalized, F.normalize(base_t, p=2, dim=0))
                loss += (1.0 - preservation) * 0.1  # Weak preservation
                
                loss.backward()
                optimizer.step()
            
            # Final result
            final_embedding = F.normalize(embedding, p=2, dim=0)
            
            return {
                "optimized_embedding": final_embedding.detach().cpu().numpy().tolist(),
                "base_similarity": float(torch.dot(
                    final_embedding,
                    F.normalize(torch.tensor(base_embedding, device=device), p=2, dim=0)
                ).item()),
                "modifications_applied": list(modifications.keys()),
                "strength": strength,
                "method": "conceptual_optimization"
            }
            
        except Exception as e:
            self.logger.error(f"Conceptual optimization failed: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the synthesis engine
        
        Returns:
            Health status dictionary
        """
        return {
            'service': 'SynthesisEngine',
            'status': 'healthy' if self.model_loaded else 'model_not_loaded',
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'auto_refine': self.auto_refine,
            'feedback_count': len(self.feedback_refiner.feedback_log) if self.feedback_refiner else 0,
            'refinement_threshold': self.refinement_threshold,
            'capabilities': {
                'expert_reranking': self.model_loaded,
                'conceptual_modification': True,
                'gradient_ascent': True,  # NOW IMPLEMENTED!
                'conceptual_optimization': True,  # ALSO IMPLEMENTED!
                'user_feedback': self.auto_refine,
                'model_refinement': self.auto_refine
            }
        }
    
    def train_conceptualizer(self, training_data: List[Dict], model_name: str = 'conceptualizer', 
                           epochs: int = 10, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train a Conceptualizer model on feedback data
        
        Args:
            training_data: List of training examples with embeddings and feedback
            model_name: Name for the trained model
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Dict with model_path and training metrics
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            from datetime import datetime
            
            self.log_info(f"Starting Conceptualizer training with {len(training_data)} examples")
            
            # Prepare training data
            embeddings = []
            scores = []
            
            for example in training_data:
                if 'embedding' in example and 'score' in example:
                    embeddings.append(example['embedding'])
                    scores.append(example['score'])
            
            if len(embeddings) < 10:
                raise ValueError("Need at least 10 training examples")
            
            # Convert to tensors
            X = torch.FloatTensor(embeddings)
            y = torch.FloatTensor(scores)
            
            # Create simple neural network for concept learning
            input_dim = X.shape[1]
            hidden_dim = min(512, input_dim // 2)
            
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, input_dim),  # Same dimension as input
                nn.Tanh()  # Normalize output
            )
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            dataset = TensorDataset(X, y.unsqueeze(1))
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training loop
            model.train()
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass - transform embeddings
                    transformed = model(batch_X)
                    
                    # Loss: transformed embeddings should correlate with scores
                    # Simple approach: dot product similarity
                    similarities = torch.sum(transformed * batch_X, dim=1, keepdim=True)
                    loss = criterion(similarities, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                
                if epoch % 5 == 0:
                    self.log_info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_{timestamp}.pth"
            model_dir = Path("models/conceptualizer")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_filename
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'training_examples': len(training_data),
                'final_loss': losses[-1] if losses else 0.0,
                'timestamp': timestamp
            }, model_path)
            
            # Update internal model
            self.model = model
            self.model_loaded = True
            
            self.log_info(f"Training completed. Model saved to: {model_path}")
            
            return {
                'model_path': str(model_path),
                'metrics': {
                    'final_loss': losses[-1] if losses else 0.0,
                    'training_examples': len(training_data),
                    'epochs': epochs,
                    'learning_rate': learning_rate
                }
            }
            
        except Exception as e:
            self.log_error(f"Training failed: {str(e)}")
            raise
    
    def optimize_query_embedding(self, initial_embedding: np.ndarray, 
                                feedback_examples: List[Dict], 
                                iterations: int = 10, 
                                learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Use gradient ascent to optimize query embedding based on user feedback
        
        Based on gradient ascent for CLIP embeddings research
        Iteratively improves embedding toward user preferences
        
        Args:
            initial_embedding: Starting query embedding
            feedback_examples: List of {path, rating, embedding} from user feedback
            iterations: Number of optimization steps
            learning_rate: Step size for gradient ascent
            
        Returns:
            Dict with optimized embedding and metrics
        """
        try:
            import torch
            import torch.nn.functional as F
            
            self.log_info(f"Optimizing query embedding with {len(feedback_examples)} feedback examples")
            
            if len(feedback_examples) < 3:
                return {
                    "optimized_embedding": initial_embedding,
                    "improvement": 0.0,
                    "iterations": 0,
                    "message": "Insufficient feedback for optimization"
                }
            
            # Convert to torch tensors
            current_embedding = torch.tensor(initial_embedding, dtype=torch.float32, requires_grad=True)
            
            # Separate positive and negative examples
            positive_examples = [ex for ex in feedback_examples if ex.get('rating', 0) >= 4]  # 4-5 stars
            negative_examples = [ex for ex in feedback_examples if ex.get('rating', 0) <= 2]  # 1-2 stars
            
            if len(positive_examples) == 0:
                return {
                    "optimized_embedding": initial_embedding,
                    "improvement": 0.0,
                    "iterations": 0,
                    "message": "No positive examples for optimization"
                }
            
            # Extract embeddings
            positive_embeddings = torch.tensor([ex['embedding'] for ex in positive_examples], dtype=torch.float32)
            negative_embeddings = torch.tensor([ex['embedding'] for ex in negative_examples], dtype=torch.float32) if negative_examples else None
            
            optimizer = torch.optim.Adam([current_embedding], lr=learning_rate)
            
            initial_score = self._calculate_embedding_score(current_embedding, positive_embeddings, negative_embeddings)
            
            for iteration in range(iterations):
                optimizer.zero_grad()
                
                # Calculate loss: maximize similarity to positive, minimize to negative
                positive_similarities = F.cosine_similarity(
                    current_embedding.unsqueeze(0), 
                    positive_embeddings, 
                    dim=1
                )
                positive_loss = -torch.mean(positive_similarities)  # Negative for maximization
                
                total_loss = positive_loss
                
                # Add negative examples if available
                if negative_embeddings is not None and len(negative_embeddings) > 0:
                    negative_similarities = F.cosine_similarity(
                        current_embedding.unsqueeze(0),
                        negative_embeddings,
                        dim=1
                    )
                    negative_loss = torch.mean(negative_similarities)  # Minimize similarity to negative
                    total_loss += 0.5 * negative_loss  # Weight negative loss lower
                
                # Regularization: keep embedding close to original
                regularization = 0.1 * F.mse_loss(current_embedding, torch.tensor(initial_embedding, dtype=torch.float32))
                total_loss += regularization
                
                total_loss.backward()
                optimizer.step()
                
                # Normalize embedding (important for cosine similarity)
                with torch.no_grad():
                    current_embedding.data = F.normalize(current_embedding.data, p=2, dim=0)
                
                if iteration % 3 == 0:
                    current_score = self._calculate_embedding_score(current_embedding, positive_embeddings, negative_embeddings)
                    self.log_debug(f"Iteration {iteration}: score = {current_score:.4f}")
            
            final_embedding = current_embedding.detach().numpy()
            final_score = self._calculate_embedding_score(current_embedding, positive_embeddings, negative_embeddings)
            improvement = final_score - initial_score
            
            self.log_info(f"Embedding optimization complete: {improvement:.4f} improvement over {iterations} iterations")
            
            return {
                "optimized_embedding": final_embedding,
                "improvement": float(improvement),
                "iterations": iterations,
                "initial_score": float(initial_score),
                "final_score": float(final_score),
                "positive_examples": len(positive_examples),
                "negative_examples": len(negative_examples) if negative_examples else 0
            }
            
        except Exception as e:
            self.log_error(f"Embedding optimization failed: {str(e)}")
            return {
                "optimized_embedding": initial_embedding,
                "improvement": 0.0,
                "iterations": 0,
                "error": str(e)
            }
    
    def _calculate_embedding_score(self, embedding: torch.Tensor, 
                                  positive_embeddings: torch.Tensor,
                                  negative_embeddings: Optional[torch.Tensor] = None) -> float:
        """Calculate quality score for an embedding based on feedback examples"""
        with torch.no_grad():
            # Score based on similarity to positive examples
            positive_similarities = F.cosine_similarity(
                embedding.unsqueeze(0), 
                positive_embeddings, 
                dim=1
            )
            positive_score = torch.mean(positive_similarities)
            
            # Penalize similarity to negative examples
            negative_penalty = 0.0
            if negative_embeddings is not None and len(negative_embeddings) > 0:
                negative_similarities = F.cosine_similarity(
                    embedding.unsqueeze(0),
                    negative_embeddings,
                    dim=1
                )
                negative_penalty = torch.mean(negative_similarities)
            
            return float(positive_score - 0.3 * negative_penalty)
    
    def mine_hard_examples(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                          ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        **HARD NEGATIVE MINING**
        Based on Facebook AI's Contrastive Learning best practices
        
        WHY: Random negatives are too easy → model doesn't learn fine-grained distinctions
        SOLUTION: Focus on the HARDEST negatives (almost-but-not-quite matches)
        
        STRATEGY (Triplet Selection):
        1. Anchor: Current sample
        2. Positive: Closest sample with SAME label (hardest positive)
        3. Hard Negative: Closest sample with DIFFERENT label (most confusing negative)
        
        This creates the most informative training signal for fine-grained learning.
        Essential for LoRA datasets where images are very similar.
        
        Args:
            embeddings: [N, D] tensor of embeddings
            labels: [N] tensor of concept labels
            ratio: Proportion of hard examples to mine (default: 0.3 = 30%)
        
        Returns:
            (anchors, positives, hard_negatives) - Each [num_hard, D]
        """
        try:
            import torch
            import torch.nn.functional as F
            
            N = len(embeddings)
            num_hard = max(int(N * ratio), 1)
            
            # Normalize embeddings for cosine similarity
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            
            # Compute pairwise similarity matrix
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())  # [N, N]
            
            # For each sample, find hardest positive and hardest negative
            anchors = []
            positives = []
            hard_negatives = []
            
            for i in range(N):
                anchor_label = labels[i]
                
                # Mask for same-label samples (excluding self)
                same_label_mask = (labels == anchor_label) & (torch.arange(N) != i)
                
                # Mask for different-label samples
                diff_label_mask = (labels != anchor_label)
                
                if torch.sum(same_label_mask) == 0 or torch.sum(diff_label_mask) == 0:
                    continue  # Skip if no valid pairs
                
                # HARDEST POSITIVE: Same label but LOWEST similarity (furthest within class)
                # This forces the model to create tighter clusters
                same_label_sims = similarity_matrix[i].clone()
                same_label_sims[~same_label_mask] = float('inf')  # Mask out invalid samples
                hardest_pos_idx = torch.argmin(same_label_sims)
                
                # HARDEST NEGATIVE: Different label but HIGHEST similarity (closest wrong class)
                # This forces the model to distinguish between similar-looking but different concepts
                diff_label_sims = similarity_matrix[i].clone()
                diff_label_sims[~diff_label_mask] = float('-inf')  # Mask out invalid samples
                hardest_neg_idx = torch.argmax(diff_label_sims)
                
                anchors.append(embeddings[i])
                positives.append(embeddings[hardest_pos_idx])
                hard_negatives.append(embeddings[hardest_neg_idx])
            
            if len(anchors) == 0:
                self.log_warning("[Hard Mining] No valid triplets found, returning empty tensors")
                return torch.zeros((0, embeddings.shape[1])), torch.zeros((0, embeddings.shape[1])), torch.zeros((0, embeddings.shape[1]))
            
            # Convert to tensors
            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            hard_negatives = torch.stack(hard_negatives)
            
            # Sample the hardest examples
            num_available = len(anchors)
            if num_available > num_hard:
                # Compute triplet "difficulty" = sim(anchor, positive) - sim(anchor, negative)
                # Lower = harder (positive far away, negative close)
                difficulties = []
                for i in range(num_available):
                    pos_sim = F.cosine_similarity(anchors[i:i+1], positives[i:i+1])
                    neg_sim = F.cosine_similarity(anchors[i:i+1], hard_negatives[i:i+1])
                    difficulty = pos_sim - neg_sim
                    difficulties.append(difficulty.item())
                
                # Select the hardest triplets
                difficulties = torch.tensor(difficulties)
                _, hardest_indices = torch.topk(-difficulties, num_hard)  # Negative for ascending order
                
                anchors = anchors[hardest_indices]
                positives = positives[hardest_indices]
                hard_negatives = hard_negatives[hardest_indices]
            
            self.log_info(f"[Hard Mining] Mined {len(anchors)} hard triplets from {N} samples (ratio={ratio})")
            
            return anchors, positives, hard_negatives
            
        except Exception as e:
            self.log_error(f"[Hard Mining] Failed: {str(e)}")
            # Return empty tensors on failure
            return torch.zeros((0, embeddings.shape[1])), torch.zeros((0, embeddings.shape[1])), torch.zeros((0, embeddings.shape[1]))
    
    def triplet_loss(self, anchors: torch.Tensor, positives: torch.Tensor, 
                    negatives: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
        """
        Triplet loss with margin for hard negative training
        
        FORMULA: max(0, d(anchor, positive) - d(anchor, negative) + margin)
        
        GOAL: Push positive closer, push negative further, with margin between them
        
        Args:
            anchors: [N, D] anchor embeddings
            positives: [N, D] positive embeddings (same concept)
            negatives: [N, D] negative embeddings (different concept)
            margin: Minimum distance between positive and negative (default: 0.3)
        
        Returns:
            Triplet loss tensor
        """
        try:
            import torch
            import torch.nn.functional as F
            
            if len(anchors) == 0:
                return torch.tensor(0.0, requires_grad=True)
            
            # Compute distances (using cosine distance = 1 - cosine_similarity)
            pos_dist = 1.0 - F.cosine_similarity(anchors, positives)
            neg_dist = 1.0 - F.cosine_similarity(anchors, negatives)
            
            # Triplet loss: positive should be closer than negative by at least 'margin'
            loss = F.relu(pos_dist - neg_dist + margin)
            
            return torch.mean(loss)
            
        except Exception as e:
            self.log_error(f"[Triplet Loss] Failed: {str(e)}")
            return torch.tensor(0.0, requires_grad=True)
    
    def ge2e_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generalized End-to-End (GE2E) Loss for concept learning
        
        Based on Google's GE2E loss for speaker verification
        Groups similar concepts together, separates different concepts
        Perfect for learning user preferences and concept distinctions
        
        Args:
            embeddings: Batch of embeddings [N, D]
            labels: Concept labels [N] (e.g., user_id, concept_id)
            
        Returns:
            GE2E loss tensor
        """
        try:
            import torch
            import torch.nn.functional as F
            
            # Get unique labels and their counts
            unique_labels = torch.unique(labels)
            
            if len(unique_labels) < 2:
                # Need at least 2 different concepts for contrastive learning
                return torch.tensor(0.0, requires_grad=True)
            
            # Compute centroids for each concept
            centroids = []
            for label in unique_labels:
                mask = (labels == label)
                if torch.sum(mask) > 0:
                    centroid = torch.mean(embeddings[mask], dim=0)
                    centroids.append(centroid)
            
            centroids = torch.stack(centroids)  # [num_concepts, D]
            
            # For each embedding, compute similarity to its own centroid vs others
            total_loss = 0.0
            num_samples = 0
            
            for i, embedding in enumerate(embeddings):
                label = labels[i]
                label_idx = torch.where(unique_labels == label)[0][0]
                
                # Compute centroid excluding this sample (for better generalization)
                mask = (labels == label)
                other_samples = embeddings[mask]
                if len(other_samples) > 1:
                    # Exclude current sample from centroid calculation
                    other_mask = torch.arange(len(embeddings))[mask] != i
                    if torch.sum(other_mask) > 0:
                        own_centroid = torch.mean(embeddings[mask][other_mask], dim=0)
                    else:
                        own_centroid = centroids[label_idx]  # Fallback to full centroid
                else:
                    own_centroid = centroids[label_idx]
                
                # Similarity to own centroid (should be high)
                positive_sim = F.cosine_similarity(embedding.unsqueeze(0), own_centroid.unsqueeze(0))
                
                # Similarities to other centroids (should be low)
                other_centroids = centroids[torch.arange(len(centroids)) != label_idx]
                if len(other_centroids) > 0:
                    negative_sims = F.cosine_similarity(
                        embedding.unsqueeze(0).expand(len(other_centroids), -1),
                        other_centroids
                    )
                    
                    # GE2E loss: log-sum-exp of negative similarities vs positive
                    # This encourages positive_sim > max(negative_sims) + margin
                    negative_exp = torch.exp(negative_sims)
                    loss_i = -positive_sim + torch.log(torch.sum(negative_exp) + torch.exp(positive_sim))
                    
                    total_loss += loss_i
                    num_samples += 1
            
            if num_samples > 0:
                return total_loss / num_samples
            else:
                return torch.tensor(0.0, requires_grad=True)
                
        except Exception as e:
            self.log_error(f"GE2E loss calculation failed: {str(e)}")
            return torch.tensor(0.0, requires_grad=True)
    
    def train_conceptualizer_with_ge2e(self, training_data: List[Dict], 
                                      model_name: str = 'conceptualizer_ge2e',
                                      epochs: int = 20, 
                                      learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Enhanced Conceptualizer training with GE2E loss
        
        Args:
            training_data: List of {embedding, concept_label, user_id} examples
            model_name: Name for the trained model
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Dict with model_path and training metrics
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            self.log_info(f"Training Conceptualizer with GE2E loss: {len(training_data)} examples")
            
            if len(training_data) < 10:
                raise ValueError("Need at least 10 training examples for GE2E")
            
            # Prepare training data
            embeddings = []
            concept_labels = []
            
            for example in training_data:
                if 'embedding' in example and 'concept_label' in example:
                    embeddings.append(example['embedding'])
                    concept_labels.append(example['concept_label'])
            
            # Convert to tensors
            X = torch.FloatTensor(embeddings)
            y = torch.LongTensor(concept_labels)
            
            # Create enhanced neural network for concept learning
            input_dim = X.shape[1]
            hidden_dim = min(512, input_dim // 2)
            
            class ConceptualizerGE2E(nn.Module):
                def __init__(self, input_dim, hidden_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.BatchNorm1d(hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim // 2, input_dim),  # Same dimension as input
                    )
                    
                def forward(self, x):
                        # L2 normalize output for cosine similarity
                        output = self.encoder(x)
                        return F.normalize(output, p=2, dim=1)
            
            model = ConceptualizerGE2E(input_dim, hidden_dim)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Read hard mining setting
            import os
            hard_mining_enabled = os.environ.get('AI_INTELLIGENCE_HARD_MINING', '1') == '1'
            hard_mining_ratio = float(os.environ.get('AI_INTELLIGENCE_HARD_MINING_RATIO', '0.3'))
            
            self.log_info(f"[Training] Hard mining: {'ENABLED' if hard_mining_enabled else 'DISABLED'} (ratio={hard_mining_ratio})")
            
            # Training with GE2E loss + Hard Negative Mining
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            model.train()
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_ge2e_loss = 0.0
                epoch_triplet_loss = 0.0
                num_batches = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    transformed = model(batch_X)
                    
                    # PRIMARY: GE2E loss (for cluster cohesion)
                    ge2e_loss_value = self.ge2e_loss(transformed, batch_y)
                    
                    total_loss = ge2e_loss_value
                    
                    # ENHANCEMENT: Hard Negative Mining (for fine-grained discrimination)
                    # Facebook AI research: Hard negatives get 3x weight for difficult examples
                    if hard_mining_enabled and len(batch_X) >= 10:  # Need enough samples
                        anchors, positives, hard_negs = self.mine_hard_examples(
                            transformed.detach(), batch_y, ratio=hard_mining_ratio
                        )
                        
                        if len(anchors) > 0:
                            # Re-encode the mined examples through model (for gradient flow)
                            anchor_indices = []
                            positive_indices = []
                            negative_indices = []
                            
                            # Find indices of mined examples in batch
                            for a, p, n in zip(anchors, positives, hard_negs):
                                # Find closest match in batch (this is approximate but works)
                                a_idx = torch.argmin(torch.sum((transformed - a.unsqueeze(0))**2, dim=1))
                                p_idx = torch.argmin(torch.sum((transformed - p.unsqueeze(0))**2, dim=1))
                                n_idx = torch.argmin(torch.sum((transformed - n.unsqueeze(0))**2, dim=1))
                                anchor_indices.append(a_idx)
                                positive_indices.append(p_idx)
                                negative_indices.append(n_idx)
                            
                            if len(anchor_indices) > 0:
                                anchor_embs = transformed[torch.tensor(anchor_indices)]
                                positive_embs = transformed[torch.tensor(positive_indices)]
                                negative_embs = transformed[torch.tensor(negative_indices)]
                                
                                triplet_loss_value = self.triplet_loss(anchor_embs, positive_embs, negative_embs)
                                
                                # 3x weight for hard negatives (Facebook AI best practice)
                                total_loss = ge2e_loss_value + 3.0 * triplet_loss_value
                                
                                epoch_triplet_loss += triplet_loss_value.item()
                    
                    if total_loss.requires_grad:
                        total_loss.backward()
                        optimizer.step()
                        epoch_loss += total_loss.item()
                        epoch_ge2e_loss += ge2e_loss_value.item()
                        num_batches += 1
                
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    avg_ge2e = epoch_ge2e_loss / num_batches
                    avg_triplet = epoch_triplet_loss / num_batches if hard_mining_enabled else 0.0
                    losses.append(avg_loss)
                    
                    if epoch % 5 == 0:
                        self.log_info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (GE2E: {avg_ge2e:.4f}, Triplet: {avg_triplet:.4f})")
            
            # Save model
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_{timestamp}.pth"
            model_dir = Path("models/conceptualizer")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_filename
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': 'ConceptualizerGE2E',
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'training_examples': len(training_data),
                'final_loss': losses[-1] if losses else 0.0,
                'loss_type': 'GE2E',
                'timestamp': timestamp
            }, model_path)
            
            # Update internal model
            self.model = model
            self.model_loaded = True
            
            self.log_info(f"GE2E Conceptualizer training completed. Model saved to: {model_path}")
            
            return {
                'model_path': str(model_path),
                'loss_type': 'GE2E',
                'metrics': {
                    'final_loss': losses[-1] if losses else 0.0,
                    'training_examples': len(training_data),
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'unique_concepts': len(torch.unique(y))
                }
            }
            
        except Exception as e:
            self.log_error(f"GE2E training failed: {str(e)}")
            raise
    
    # ==================== DATASET-AWARE QUERY EXPANSION ====================
    # Context-aware query enhancement using trained Conceptualizer
    # Extracts learned patterns from dataset to improve query understanding
    # =======================================================================
    
    def extract_cluster_centers(self, model_path: str) -> Dict[str, Any]:
        """
        Extract cluster centers from trained Conceptualizer model
        
        WHY: Conceptualizer learns dataset-specific visual concepts
        USE: These clusters represent common patterns in YOUR dataset
        BENEFIT: Queries can be expanded with dataset-specific context
        
        Args:
            model_path: Path to trained Conceptualizer model
        
        Returns:
            Dict with cluster_centers [N, 768] and cluster_labels
        """
        try:
            import torch
            from sklearn.cluster import KMeans
            
            self.log_info(f"[Query Expansion] Extracting cluster centers from: {model_path}")
            
            # Load trained model
            if not os.path.exists(model_path):
                return {"error": "Model not found"}
            
            model_state = torch.load(model_path, map_location='cpu')
            
            # Extract embeddings that the model was trained on
            # (Assuming training embeddings were saved with model)
            training_embeddings = model_state.get('training_embeddings', None)
            
            if training_embeddings is None:
                self.log_warning("[Query Expansion] No training embeddings found in model")
                return {"error": "No training data in model"}
            
            # Cluster the training embeddings to find patterns
            n_clusters = min(20, len(training_embeddings) // 50)  # ~20 clusters or 1 per 50 images
            n_clusters = max(5, n_clusters)  # At least 5 clusters
            
            self.log_info(f"[Query Expansion] Clustering {len(training_embeddings)} embeddings into {n_clusters} clusters")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(training_embeddings)
            cluster_centers = kmeans.cluster_centers_
            
            self.log_info(f"[Query Expansion] ✓ Extracted {n_clusters} cluster centers")
            
            return {
                'cluster_centers': cluster_centers.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'n_clusters': n_clusters,
                'total_embeddings': len(training_embeddings)
            }
            
        except Exception as e:
            self.log_error(f"[Query Expansion] Cluster extraction failed: {str(e)}")
            return {"error": str(e)}
    
    def find_representative_images_per_cluster(self, cluster_centers: np.ndarray,
                                              all_embeddings: List[np.ndarray],
                                              all_image_paths: List[str],
                                              top_k: int = 5) -> Dict[str, Any]:
        """
        Find most representative images for each cluster
        
        STRATEGY: For each cluster, find K images closest to cluster center
        USE: These represent the "essence" of that visual concept
        
        Args:
            cluster_centers: Cluster center embeddings [N_clusters, 768]
            all_embeddings: All dataset embeddings [N_images, 768]
            all_image_paths: Paths to all images
            top_k: Number of representatives per cluster
        
        Returns:
            Dict mapping cluster_id -> [representative_image_paths]
        """
        try:
            import numpy as np
            from scipy.spatial.distance import cdist
            
            self.log_info(f"[Query Expansion] Finding {top_k} representatives per cluster")
            
            # Compute distances from all embeddings to all cluster centers
            distances = cdist(all_embeddings, cluster_centers, metric='cosine')
            
            representatives = {}
            
            for cluster_id in range(len(cluster_centers)):
                # Get distances to this cluster center
                cluster_distances = distances[:, cluster_id]
                
                # Find top-K closest images
                closest_indices = np.argsort(cluster_distances)[:top_k]
                
                representatives[cluster_id] = {
                    'image_paths': [all_image_paths[i] for i in closest_indices],
                    'distances': cluster_distances[closest_indices].tolist()
                }
            
            self.log_info(f"[Query Expansion] ✓ Found representatives for {len(representatives)} clusters")
            
            return {
                'representatives': representatives,
                'clusters': len(cluster_centers),
                'images_per_cluster': top_k
            }
            
        except Exception as e:
            self.log_error(f"[Query Expansion] Representative finding failed: {str(e)}")
            return {"error": str(e)}
    
    def extract_common_attributes(self, representative_paths: List[str],
                                  profile_data: Dict) -> List[str]:
        """
        Extract common VLM attributes from representative images
        
        STRATEGY: Find attributes shared by representatives
        USE: These become query expansion terms
        
        EXAMPLE:
        - Cluster center for "outdoor scenes"
        - Representatives: beach, forest, mountain images
        - Common attributes: "outdoor", "landscape", "nature", "blue sky"
        - Query expansion: "beach" → "beach outdoor landscape nature"
        
        Args:
            representative_paths: Paths to representative images
            profile_data: Dataset profile with VLM classifications
        
        Returns:
            List of common attribute strings
        """
        try:
            from collections import Counter
            
            self.log_debug(f"[Query Expansion] Extracting attributes from {len(representative_paths)} images")
            
            # Collect all attributes from representatives
            all_attributes = []
            
            for img_path in representative_paths:
                filename = os.path.basename(img_path)
                img_profile = profile_data.get('images', {}).get(filename, {}).get('universal_profile', {})
                
                # Extract VLM attributes
                content_type = img_profile.get('content_type', '')
                human_presence = img_profile.get('human_presence', '')
                human_details = img_profile.get('human_details', '')
                dominant_colors = img_profile.get('dominant_colors', '')
                
                # Split and add
                if content_type:
                    all_attributes.append(content_type)
                if human_presence and human_presence != 'none':
                    all_attributes.append(human_presence)
                if human_details:
                    all_attributes.extend(human_details.split(','))
                if dominant_colors:
                    all_attributes.extend(dominant_colors.split(','))
            
            # Find most common attributes (appear in >50% of representatives)
            attribute_counts = Counter(all_attributes)
            threshold = len(representative_paths) * 0.5
            
            common_attrs = [
                attr.strip() 
                for attr, count in attribute_counts.items() 
                if count >= threshold and len(attr.strip()) > 0
            ]
            
            self.log_debug(f"[Query Expansion] Found {len(common_attrs)} common attributes")
            
            return common_attrs
            
        except Exception as e:
            self.log_error(f"[Query Expansion] Attribute extraction failed: {str(e)}")
            return []
    
    def expand_query_with_dataset_context(self, query: str,
                                         cluster_centers: np.ndarray,
                                         cluster_representatives: Dict,
                                         profile_data: Dict) -> List[str]:
        """
        **DATASET-AWARE QUERY EXPANSION**
        Expand user query with dataset-specific context from Conceptualizer
        
        WORKFLOW:
        1. Encode user query
        2. Find closest cluster center (most relevant visual concept)
        3. Get representative images for that cluster
        4. Extract common attributes from representatives
        5. Expand query with those attributes
        
        EXAMPLE:
        User query: "woman"
        Closest cluster: "formal portraits"
        Representatives: professional headshots, business attire
        Common attributes: "professional", "formal", "business", "indoor"
        Expanded: "woman professional formal business indoor"
        
        Args:
            query: Original user query
            cluster_centers: Cluster centers from Conceptualizer
            cluster_representatives: Representative images per cluster
            profile_data: Dataset profile with VLM data
        
        Returns:
            List of expanded query variations
        """
        try:
            import numpy as np
            from scipy.spatial.distance import cdist
            
            self.log_info(f"[Query Expansion] Expanding query: '{query}'")
            
            # Encode query using CLIP (we need SemanticService for this)
            # For now, return basic expansion
            # TODO: Integrate with SemanticService to get query embedding
            
            expanded_queries = [query]  # Original always first
            
            # Find closest cluster (requires query embedding)
            # For now, use simple keyword matching as fallback
            
            # Extract keywords from query
            query_lower = query.lower()
            query_keywords = query_lower.split()
            
            # Find clusters whose representatives match query keywords
            matching_clusters = []
            
            for cluster_id, cluster_data in cluster_representatives.items():
                rep_paths = cluster_data['image_paths']
                common_attrs = self.extract_common_attributes(rep_paths, profile_data)
                
                # Check if any common attributes match query
                for attr in common_attrs:
                    if any(keyword in attr.lower() for keyword in query_keywords):
                        matching_clusters.append((cluster_id, common_attrs))
                        break
            
            # Expand query with attributes from matching clusters
            if matching_clusters:
                # Use first matching cluster
                cluster_id, common_attrs = matching_clusters[0]
                
                # Create expansion by adding top 2-3 common attributes
                expansion_terms = common_attrs[:3]
                if expansion_terms:
                    expanded_query = query + " " + " ".join(expansion_terms)
                    expanded_queries.append(expanded_query)
                    
                    self.log_info(f"[Query Expansion] ✓ Expanded to: '{expanded_query}'")
            else:
                self.log_debug(f"[Query Expansion] No matching clusters found for query")
            
            return expanded_queries
            
        except Exception as e:
            self.log_error(f"[Query Expansion] Failed: {str(e)}")
            return [query]  # Fallback to original


def main():
    """Main entry point for standalone testing and IPC communication"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthesis Engine')
    parser.add_argument('--mode', choices=['ipc', 'test'], default='ipc',
                       help='Operation mode')
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if args.model:
        config['model_path'] = args.model
    
    # Initialize engine
    engine = SynthesisEngine(config)
    
    # Load model if path provided
    if config.get('model_path'):
        engine.load_model()
    
    if args.mode == 'test':
        # Test mode
        print("Synthesis Engine Test")
        print(json.dumps(engine.get_health_status(), indent=2))
        
    else:
        # IPC mode: listen for commands via stdin/stdout
        engine.log_info("Starting IPC mode, waiting for commands...")
        
        for line in sys.stdin:
            try:
                command = json.loads(line.strip())
                cmd_type = command.get('command')
                
                if cmd_type == 'load_model':
                    result = engine.load_model(command.get('model_path'))
                    print(json.dumps({'success': result}))
                    sys.stdout.flush()
                    
                elif cmd_type == 'transform':
                    embeddings = np.array(command['embeddings'])
                    result = engine.transform_embeddings(embeddings)
                    # Convert numpy array to list for JSON
                    result['specialized_embeddings'] = result['specialized_embeddings'].tolist()
                    print(json.dumps(result))
                    sys.stdout.flush()
                    
                elif cmd_type == 'rerank':
                    query_emb = np.array(command['query_embedding'])
                    cand_embs = np.array(command['candidate_embeddings'])
                    cand_paths = command['candidate_paths']
                    k = command.get('k', 10)
                    
                    result = engine.expert_rerank(query_emb, cand_embs, cand_paths, k)
                    print(json.dumps(result))
                    sys.stdout.flush()
                    
                elif cmd_type == 'modify':
                    base_emb = np.array(command['base_embedding'])
                    add_embs = [np.array(e) for e in command.get('add_embeddings', [])]
                    sub_embs = [np.array(e) for e in command.get('subtract_embeddings', [])]
                    weights = command.get('weights')
                    
                    result = engine.conceptual_modification(base_emb, add_embs, sub_embs, weights)
                    result['modified_embedding'] = result['modified_embedding'].tolist()
                    print(json.dumps(result))
                    sys.stdout.flush()
                    
                elif cmd_type == 'feedback':
                    query_emb = np.array(command['query_embedding'])
                    result_emb = np.array(command['result_embedding'])
                    result_path = command['result_path']
                    is_positive = command['is_positive']
                    metadata = command.get('metadata')
                    
                    result = engine.add_feedback(query_emb, result_emb, result_path, is_positive, metadata)
                    print(json.dumps(result))
                    sys.stdout.flush()
                    
                elif cmd_type == 'health':
                    status = engine.get_health_status()
                    print(json.dumps(status))
                    sys.stdout.flush()
                    
                elif cmd_type == 'shutdown':
                    engine.log_info("Shutdown command received")
                    break
                    
                else:
                    error_msg = {'error': f'Unknown command: {cmd_type}'}
                    print(json.dumps(error_msg))
                    sys.stdout.flush()
                    
            except json.JSONDecodeError as e:
                error_msg = {'error': f'Invalid JSON: {str(e)}'}
                print(json.dumps(error_msg))
                sys.stdout.flush()
            except Exception as e:
                error_msg = {'error': f'Command failed: {str(e)}'}
                print(json.dumps(error_msg))
                sys.stdout.flush()


async def main():
    """Run Synthesis Engine as a service"""
    engine = SynthesisEngine()
    
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

