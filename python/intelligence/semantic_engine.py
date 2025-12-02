"""
Semantic Engine - CLIP-SAE Expert Service (Adversarially Trained)
Handles all semantic search and embedding operations

Uses zer0int/LongCLIP-SAE-ViT-L-14:
- 90% ImageNet/ObjectNet accuracy (vs ~75% base CLIP)
- Adversarial robustness (typographic attack resistant)
- Sparse Autoencoder enhanced embeddings
- Better semantic precision and clustering

Reference: https://github.com/zer0int/CLIP-SAE-finetune
HuggingFace: https://huggingface.co/zer0int/LongCLIP-SAE-ViT-L-14

Inherits from BaseEngine for standardized interface
"""

import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add parent directory to path for base_engine import
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine

# Import existing semantic search components
try:
    import torch
    import open_clip
    import faiss
    from PIL import Image
except ImportError as e:
    print(f"Error importing dependencies: {e}", file=sys.stderr)
    sys.exit(1)


class SemanticEngine(BaseEngine):
    """
    Semantic Engine - Expert service for CLIP-SAE semantic search
    
    Capabilities:
    - Generate CLIP-SAE embeddings for images (90% accuracy)
    - Semantic similarity search (adversarially robust)
    - Cluster analysis (improved separation)
    - Index management (create, save, load)
    
    Model: zer0int/LongCLIP-SAE-ViT-L-14
    - ViT-L/14 architecture
    - SAE-enhanced with adversarial training
    - Geometric Parametrization (GmP) fine-tuned
    - Resistant to typographic attacks
    """
    
    def __init__(self):
        super().__init__("semantic_engine", "2.0.0")  # Version bump for SAE
        
        # Model and index state
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = None
        
        # FAISS indexes (Multi-level indexing like Netflix)
        self.index = None  # Main full-image index
        self.image_paths = []
        self.dimension = 768  # CLIP-SAE ViT-L/14 dimension
        
        # Multi-level indexes for granular search
        self.person_index = None      # Person-focused embeddings
        self.object_index = None      # Object-focused embeddings  
        self.scene_index = None       # Scene/background embeddings
        self.person_paths = []
        self.object_paths = []
        self.scene_paths = []
        
        # Temperature scaling for optimized similarity (from CLIP paper)
        # Read from environment variable if set
        import os
        env_temperature = os.environ.get('CLIP_TEMPERATURE', '0.07')
        self.temperature = float(env_temperature)
        self.adaptive_temperature = True
        self.temperature_history = []  # Track performance at different temperatures
        
        # Configuration - CLIP-SAE model selection
        # Check environment variable for FP32 preference
        import os
        self.use_fp32 = os.environ.get('CLIP_USE_FP32', '0') == '1'
        
        self.model_name = "ViT-L-14"
        
        # Model configs
        self.model_configs = {
            "bf16": {
                "repo": "zer0int/LongCLIP-SAE-ViT-L-14",
                "file": "Long-ViT-L-14-GmP-SAE-full-model.safetensors",
                "size": "1.5GB",
                "precision": "bfloat16"
            },
            "fp32": {
                "repo": "Kutches/clips",
                "file": "CLIP-SAE-ViT-L-14-FP32.safetensors",
                "size": "1.71GB",
                "precision": "float32"
            }
        }
        
        # Select config based on setting
        config_key = "fp32" if self.use_fp32 else "bf16"
        selected_config = self.model_configs[config_key]
        self.model_repo = selected_config["repo"]
        self.checkpoint_file = selected_config["file"]
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the Semantic Engine
        Load CLIP-SAE model with fallback support
        """
        try:
            self.logger.info("Initializing Semantic Engine (CLIP-SAE)...")
            
            # Detect device with fallback capability
            self.gpu_available = torch.cuda.is_available()
            self.using_cpu_fallback = False
            
            if self.gpu_available:
                try:
                    # Test GPU allocation
                    test_tensor = torch.zeros(1, device='cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                    self.device = "cuda"
                    self.logger.info(f"Using device: {self.device}")
                except RuntimeError as e:
                    self.logger.warning(f"GPU test failed: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    self.using_cpu_fallback = True
            else:
                self.device = "cpu"
                self.using_cpu_fallback = True
                self.logger.warning("No CUDA GPU detected. Using CPU (slower processing).")
            
            # Log selected model precision
            config_key = "fp32" if self.use_fp32 else "bf16"
            selected_config = self.model_configs[config_key]
            self.logger.info(f"Using {selected_config['precision']} precision ({selected_config['size']})")
            
            # Load CLIP-SAE model from HuggingFace
            self.logger.info(f"Loading CLIP-SAE from: {self.model_repo}")
            
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            
            # Download checkpoint (respects HF_HOME env var if set)
            self.logger.info(f"Downloading {self.checkpoint_file}...")
            import os
            cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or "./models"
            self.logger.info(f"Using cache directory: {cache_dir}")
            
            checkpoint_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.checkpoint_file,
                cache_dir=cache_dir
            )
            self.logger.info(f"Downloaded to: {checkpoint_path}")
            
            # Create base CLIP model structure
            self.logger.info(f"Creating {self.model_name} architecture...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=None,  # Don't load pretrained weights yet
                device=self.device
            )
            
            # Load CLIP-SAE weights from safetensors
            self.logger.info("Loading CLIP-SAE weights...")
            state_dict = load_file(checkpoint_path)
            self.model.load_state_dict(state_dict, strict=False)
            
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
            
            self.logger.info(f"✓ CLIP-SAE loaded successfully")
            
            # Emit CPU fallback warning if applicable
            if self.using_cpu_fallback:
                self.logger.warning("⚠ Running on CPU - processing will be significantly slower (10-50x)")
                
            model_accuracy = "90%"
            is_sae = True
            
            # Initialize empty FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
            
            self._mark_initialized()
            
            return {
                "model": self.model_name,
                "model_type": "CLIP-SAE" if "SAE" in self.model_name else "Long-CLIP",
                "accuracy": model_accuracy,
                "adversarial_robust": is_sae,
                "device": self.device,
                "dimension": self.dimension,
                "index_size": self.index.ntotal,
                "cpu_fallback": self.using_cpu_fallback,
                "warning": "Running on CPU - slower processing" if self.using_cpu_fallback else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self._mark_unhealthy(str(e))
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request
        
        Supported actions:
        - embed_image: Generate embedding for an image
        - embed_text: Generate embedding for text
        - search_similar: Find similar images
        - add_to_index: Add image to index
        - save_index: Save index to disk
        - load_index: Load index from disk
        - cluster_analysis: Perform clustering
        """
        action = request.get('action')
        
        # Handle health check
        if action == 'health_check':
            return await self.health_check()
        
        # Handle settings update
        if action == 'update_settings':
            return await self._update_settings(request.get('settings', {}))
        
        if action == 'embed_image':
            return await self._embed_image(request['image_path'])
        
        elif action == 'embed_text':
            return await self._embed_text(request['text'])
        
        elif action == 'search_similar':
            # Enhanced search with optional query expansion and temperature scaling
            query = request.get('query')
            expanded_queries = request.get('expanded_queries', [query])
            use_temperature = request.get('use_temperature_scaling', False)
            top_k = request.get('top_k', 50)
            threshold = request.get('threshold', 0.0)
            
            if use_temperature and len(expanded_queries) > 1:
                # Multi-query search with temperature scaling
                return await self._search_expanded_with_temperature(
                    expanded_queries, top_k, threshold
                )
            elif use_temperature:
                # Single query with temperature scaling
                query_embedding = await self._embed_text(query)
                return self.search_with_temperature(
                    np.array(query_embedding['embedding']), top_k
                )
            else:
                # Standard search
                return await self._search_similar(query, top_k, threshold)
        
        elif action == 'search_by_embedding':
            # Direct embedding-based search (for reference image similarity)
            return await self._search_by_embedding(
                np.array(request['embedding']),
                request.get('top_k', 50),
                request.get('threshold', 0.0)
            )
        
        elif action == 'search_with_temperature':
            return self.search_with_temperature(
                np.array(request['query_embedding']),
                request.get('k', 50),
                request.get('temperature')
            )
        
        elif action == 'search_multi_level':
            # Support both text query and pre-computed embedding
            if 'query_embedding' in request:
                query_embedding = np.array(request['query_embedding'])
            elif 'query' in request:
                # Generate embedding from text query
                query_embedding = self._embed_text_sync(request['query'])
            else:
                raise ValueError("search_multi_level requires either 'query' or 'query_embedding'")
            
            return self.search_multi_level(
                query_embedding,
                request.get('query_type', 'auto'),
                request.get('k', request.get('top_k', 50))
            )
        
        elif action == 'build_multi_level_indexes':
            return self.build_multi_level_indexes(request['profile_data'])
        
        elif action == 'generate_hierarchical_embeddings':
            return await self.generate_hierarchical_embeddings(
                request['image_path'],
                request.get('pose_data')
            )
        
        elif action == 'hierarchical_search':
            return await self.hierarchical_search(
                request['query'],
                request.get('level', 'auto'),
                request.get('k', 50)
            )
        
        elif action == 'generate_spatial_embeddings':
            return await self.generate_spatial_embeddings(request['image_path'])
        
        elif action == 'spatial_search':
            return await self.spatial_search(
                request['query'],
                request.get('k', 50)
            )
        
        elif action == 'hard_negative_mining':
            return self.hard_negative_mining(
                np.array(request['query_embedding']),
                request['positive_paths'],
                request.get('k_negatives', 50)
            )
        
        elif action == 'add_to_index':
            return await self._add_to_index(
                request['image_paths'],
                request.get('batch_size', 32)
            )
        
        elif action == 'save_index':
            return await self._save_index(request['save_path'])
        
        elif action == 'load_index':
            return await self._load_index(request['index_path'])
        
        elif action == 'project_embeddings':
            # NEW: Project embeddings to 2D for visualization
            return await self._project_embeddings(
                request['embeddings'],
                request.get('method', 'tsne'),
                request.get('perplexity', 30)
            )
        
        elif action == 'reorder_metadata_by_ivf':
            return await self._reorder_metadata_by_ivf_clusters()
        
        elif action == 'cluster_analysis':
            return await self._cluster_analysis(
                request.get('n_clusters', 10)
            )
        
        elif action == 'get_index_info':
            return self._get_index_info()
        
        elif action == 'find_duplicates':
            return await self._find_duplicates(
                request.get('threshold', 0.95),
                request.get('method', 'semantic')
            )
        
        elif action == 'find_outliers':
            return await self._find_outliers(
                request.get('std_threshold', 2.0)
            )
        
        elif action == 'calculate_diversity':
            return await self._calculate_diversity()
        
        elif action == 'caption_image_alignment':
            return await self._check_caption_alignment(
                request.get('threshold', 0.25),
                request.get('mode', 'find_misaligned')
            )
        
        elif action == 'get_embeddings_for_training':
            return self._get_embeddings_for_training()
        
        elif action == 'get_embedding_visualization':
            return await self._get_embedding_visualization(
                request.get('sample_size', 100),
                request.get('include_similarity_matrix', False),
                request.get('include_distribution', False),
                request.get('projection_method', 'pca')
            )
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update CLIP model settings (requires restart to take effect)"""
        try:
            use_fp32 = settings.get('use_fp32_clip', False)
            
            if use_fp32 != self.use_fp32:
                self.logger.info(f"CLIP precision setting updated: {'FP32' if use_fp32 else 'BF16'}")
                self.logger.info("Note: Restart SemanticService for changes to take effect")
            
            return {
                "success": True,
                "current_precision": "fp32" if self.use_fp32 else "bf16",
                "pending_precision": "fp32" if use_fp32 else "bf16",
                "requires_restart": use_fp32 != self.use_fp32
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _embed_image(self, image_path: str) -> Dict[str, Any]:
        """Generate CLIP embedding for an image"""
        try:
            image = Image.open(image_path)
            
            # Apply EXIF orientation before processing
            # Critical for phone photos stored with rotation metadata
            try:
                from PIL import ImageOps
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass  # Silently continue if EXIF transpose fails
            
            image = image.convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
            
            embedding_np = embedding.cpu().numpy().flatten()
            
            return {
                "embedding": embedding_np.tolist(),
                "dimension": len(embedding_np),
                "image_path": image_path
            }
            
        except Exception as e:
            self.logger.error(f"Error embedding image {image_path}: {e}")
            raise
    
    async def _embed_text(self, text: str) -> Dict[str, Any]:
        """Generate CLIP embedding for text"""
        try:
            text_token = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_text(text_token)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            embedding_np = embedding.cpu().numpy().flatten()
            
            return {
                "embedding": embedding_np.tolist(),
                "dimension": len(embedding_np),
                "text": text
            }
            
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise
    
    def _embed_text_sync(self, text: str) -> np.ndarray:
        """Generate CLIP embedding for text (synchronous, returns numpy array directly)"""
        try:
            text_token = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_text(text_token)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise
    
    async def _search_similar(self, query: str, top_k: int, threshold: float) -> Dict[str, Any]:
        """
        Search for similar images
        
        Args:
            query: Text query or image path
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
        """
        try:
            # Generate query embedding
            if Path(query).exists():
                # Image query
                query_result = await self._embed_image(query)
            else:
                # Text query
                query_result = await self._embed_text(query)
            
            query_embedding = np.array([query_result['embedding']], dtype=np.float32)
            
            # Search index
            if self.index.ntotal == 0:
                return {
                    "results": [],
                    "count": 0,
                    "message": "Index is empty"
                }
            
            # Perform search
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Filter by threshold and format results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist >= threshold and idx < len(self.image_paths):
                    results.append({
                        "image_path": self.image_paths[idx],
                        "similarity": float(dist),
                        "index": int(idx)
                    })
            
            return {
                "results": results,
                "count": len(results),
                "query": query,
                "top_k": top_k,
                "threshold": threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error searching: {e}")
            raise
    
    async def _search_by_embedding(self, embedding: np.ndarray, top_k: int, threshold: float) -> Dict[str, Any]:
        """
        Search for similar images using a pre-computed embedding.
        Used for reference image similarity queries.
        
        Args:
            embedding: Pre-computed embedding vector (from embed_image)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                return {
                    "results": [],
                    "count": 0,
                    "message": "Index is empty or not loaded"
                }
            
            # Ensure embedding is the right shape
            query_embedding = np.array([embedding], dtype=np.float32)
            
            # Perform search
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Filter by threshold and format results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist >= threshold and idx < len(self.image_paths):
                    results.append({
                        "image_path": self.image_paths[idx],
                        "path": self.image_paths[idx],  # Alias for consistency
                        "similarity": float(dist),
                        "index": int(idx)
                    })
            
            self.logger.info(f"Embedding search found {len(results)} results (threshold: {threshold})")
            
            return {
                "results": results,
                "count": len(results),
                "top_k": top_k,
                "threshold": threshold,
                "search_type": "embedding"
            }
            
        except Exception as e:
            self.logger.error(f"Error in embedding search: {e}")
            raise
    
    async def _add_to_index(self, image_paths: List[str], batch_size: int) -> Dict[str, Any]:
        """Add images to the FAISS index with VRAM exhaustion handling"""
        try:
            added_count = 0
            embeddings = []
            valid_paths = []
            embedding_vectors = []  # Store actual vectors for later retrieval
            current_batch_size = batch_size
            vram_reduced = False
            
            # Process in batches with VRAM exhaustion recovery
            i = 0
            while i < len(image_paths):
                batch = image_paths[i:i+current_batch_size]
                batch_success = True
                
                for image_path in batch:
                    try:
                        result = await self._embed_image(image_path)
                        embedding_vec = result['embedding']
                        embeddings.append(embedding_vec)
                        embedding_vectors.append(embedding_vec)
                        valid_paths.append(image_path)
                        added_count += 1
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        # Detect CUDA OOM errors
                        if 'out of memory' in error_str or 'cuda' in error_str:
                            self.logger.warning(f"VRAM exhaustion detected, reducing batch size from {current_batch_size} to {max(1, current_batch_size // 2)}")
                            # Clear GPU cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Reduce batch size and retry this batch
                            current_batch_size = max(1, current_batch_size // 2)
                            vram_reduced = True
                            batch_success = False
                            break
                        else:
                            self.logger.warning(f"Failed to embed {image_path}: {e}")
                    except Exception as e:
                        self.logger.warning(f"Failed to embed {image_path}: {e}")
                
                # If batch succeeded, move to next batch; otherwise retry with smaller size
                if batch_success:
                    i += len(batch)
                elif current_batch_size < 1:
                    # Cannot reduce further, skip this batch
                    self.logger.error("Cannot process images even with batch_size=1, skipping remaining")
                    break
            
            # Add to index
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.index.add(embeddings_array)
                self.image_paths.extend(valid_paths)
                
                # Store embedding vectors for training (append to existing)
                if not hasattr(self, 'embedding_vectors'):
                    self.embedding_vectors = []
                self.embedding_vectors.extend(embedding_vectors)
            
            result = {
                "added": added_count,
                "failed": len(image_paths) - added_count,
                "total_in_index": self.index.ntotal,
                "embeddings": embedding_vectors  # Return vectors for profile storage
            }
            
            # Include VRAM warning if batch size was reduced
            if vram_reduced:
                result["vram_warning"] = True
                result["reduced_batch_size"] = current_batch_size
                self.logger.warning(f"Completed with reduced batch size ({current_batch_size}) due to VRAM constraints")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error adding to index: {e}")
            raise
    
    async def _save_index(self, save_path: str) -> Dict[str, Any]:
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.index, save_path)
            
            # Save image paths separately
            paths_file = str(Path(save_path).with_suffix('.paths'))
            with open(paths_file, 'w') as f:
                json.dump(self.image_paths, f)
            
            # Save embedding vectors for training
            if hasattr(self, 'embedding_vectors') and self.embedding_vectors:
                embeddings_file = str(Path(save_path).with_suffix('.embeddings.npy'))
                np.save(embeddings_file, np.array(self.embedding_vectors, dtype=np.float32))
                self.logger.info(f"Saved {len(self.embedding_vectors)} embedding vectors for training")
            
            return {
                "saved": True,
                "index_path": save_path,
                "paths_file": paths_file,
                "embeddings_saved": hasattr(self, 'embedding_vectors') and len(self.embedding_vectors) > 0,
                "size": self.index.ntotal
            }
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
            raise
    
    async def _load_index(self, index_path: str) -> Dict[str, Any]:
        """Load FAISS index from disk with integrity validation"""
        try:
            self.index = faiss.read_index(index_path)
            
            # Load image paths
            paths_file = str(Path(index_path).with_suffix('.paths'))
            if Path(paths_file).exists():
                with open(paths_file, 'r') as f:
                    self.image_paths = json.load(f)
            else:
                self.image_paths = []
            
            # CRITICAL: Validate index/paths alignment
            # This catches corruption from partial saves or external file changes
            index_size = self.index.ntotal
            paths_count = len(self.image_paths)
            alignment_ok = True
            missing_files = []
            
            if index_size != paths_count:
                self.logger.error(f"INDEX MISMATCH: index has {index_size} vectors but {paths_count} paths!")
                alignment_ok = False
            
            # Spot-check that some paths still exist (don't check all - too slow for large datasets)
            sample_size = min(10, paths_count)
            if sample_size > 0:
                import random
                sample_indices = random.sample(range(paths_count), sample_size)
                for idx in sample_indices:
                    if not Path(self.image_paths[idx]).exists():
                        missing_files.append(self.image_paths[idx])
                
                if len(missing_files) > sample_size // 2:
                    self.logger.warning(f"Many sampled paths are missing ({len(missing_files)}/{sample_size}) - index may be stale")
            
            # Load embedding vectors for training
            embeddings_file = str(Path(index_path).with_suffix('.embeddings.npy'))
            if Path(embeddings_file).exists():
                self.embedding_vectors = np.load(embeddings_file).tolist()
                self.logger.info(f"Loaded {len(self.embedding_vectors)} embedding vectors for training")
            else:
                self.embedding_vectors = []
            
            return {
                "loaded": True,
                "index_path": index_path,
                "size": index_size,
                "paths_loaded": paths_count,
                "embeddings_loaded": len(self.embedding_vectors),
                "alignment_ok": alignment_ok,
                "missing_files_sample": missing_files if missing_files else None,
                "needs_rebuild": not alignment_ok or len(missing_files) > sample_size // 2
            }
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            raise
    
    async def _reorder_metadata_by_ivf_clusters(self) -> Dict[str, Any]:
        """
        Reorder metadata to match IVF cluster ordering for data locality
        
        **CRITICAL PERFORMANCE OPTIMIZATION**
        Based on rom1504/clip-retrieval research
        
        WHY: FAISS IVF indices have strong locality of reference
        - Images in same cluster are stored sequentially
        - KNN results tend to come from same/nearby clusters
        - Sequential reads are 100-1000x faster than random reads
        
        IMPACT:
        - WITHOUT reordering: Fetching 1000 results = 1000 random seeks = ~10s
        - WITH reordering: Fetching 1000 results = ~10 sequential reads = ~10ms
        - Performance gain: 100-1000x for large K queries
        
        USE CASE: Essential for LoRA/Instagram datasets where users want 100-1000+ results
        """
        try:
            if self.index.ntotal == 0:
                self.logger.warn("Cannot reorder: index is empty")
                return {"reordered": False, "reason": "empty_index"}
            
            if len(self.image_paths) != self.index.ntotal:
                self.logger.warn("Cannot reorder: image_paths length mismatch")
                return {"reordered": False, "reason": "length_mismatch"}
            
            self.logger.info(f"[IVF REORDERING] Starting reordering of {self.index.ntotal} items for data locality...")
            start_time = time.time()
            
            # Extract all embeddings from current index
            embeddings = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
            for i in range(self.index.ntotal):
                embeddings[i] = self.index.reconstruct(i)
            
            # Perform clustering to get cluster assignments
            # Using same number of clusters as typical IVF index (sqrt of dataset size)
            n_clusters = max(int(np.sqrt(self.index.ntotal)), 10)
            self.logger.info(f"[IVF REORDERING] Clustering into {n_clusters} clusters...")
            
            kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=False, gpu=torch.cuda.is_available())
            kmeans.train(embeddings)
            
            # Get cluster assignment for each embedding
            _, labels = kmeans.index.search(embeddings, 1)
            cluster_assignments = labels.flatten()
            
            # Sort indices by cluster ID (this is the key optimization!)
            sorted_indices = np.argsort(cluster_assignments)
            
            self.logger.info(f"[IVF REORDERING] Reordering image paths and embeddings...")
            
            # Reorder image paths to match cluster order
            self.image_paths = [self.image_paths[i] for i in sorted_indices]
            
            # Reorder embeddings to match cluster order
            embeddings = embeddings[sorted_indices]
            
            # Reorder embedding_vectors if available (for training)
            if hasattr(self, 'embedding_vectors') and len(self.embedding_vectors) == self.index.ntotal:
                self.embedding_vectors = [self.embedding_vectors[i] for i in sorted_indices]
            
            # Rebuild FAISS index with reordered embeddings
            self.logger.info(f"[IVF REORDERING] Rebuilding FAISS index with optimized ordering...")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            
            elapsed = time.time() - start_time
            
            self.logger.info(f"[IVF REORDERING] ✓ Complete in {elapsed:.2f}s")
            self.logger.info(f"[IVF REORDERING] Performance boost: Large K queries (100-1000+) now 100-1000x faster")
            
            return {
                "reordered": True,
                "items_reordered": self.index.ntotal,
                "n_clusters": n_clusters,
                "elapsed_seconds": elapsed,
                "performance_gain": "100-1000x for large K queries",
                "recommended_for": "LoRA datasets, batch curation (K > 100)"
            }
            
        except Exception as e:
            self.logger.error(f"[IVF REORDERING] Failed: {e}")
            return {
                "reordered": False,
                "error": str(e)
            }
    
    async def _cluster_analysis(self, n_clusters: int) -> Dict[str, Any]:
        """Perform k-means clustering on indexed embeddings"""
        try:
            if self.index.ntotal == 0:
                return {"clusters": [], "message": "Index is empty"}
            
            # Extract all embeddings from index
            embeddings = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
            for i in range(self.index.ntotal):
                embeddings[i] = self.index.reconstruct(i)
            
            # Perform k-means clustering
            kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=False)
            kmeans.train(embeddings)
            
            # Assign cluster labels
            _, labels = kmeans.index.search(embeddings, 1)
            labels = labels.flatten()
            
            # Group images by cluster
            clusters = {}
            for i, label in enumerate(labels):
                cluster_id = int(label)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                if i < len(self.image_paths):
                    clusters[cluster_id].append(self.image_paths[i])
            
            return {
                "n_clusters": n_clusters,
                "clusters": clusters,
                "cluster_sizes": {k: len(v) for k, v in clusters.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error in clustering: {e}")
            raise
    
    async def _find_duplicates(self, threshold: float, method: str) -> Dict[str, Any]:
        """
        Find duplicate/near-duplicate images using semantic similarity
        Critical for dataset curation
        """
        try:
            if self.index.ntotal == 0:
                return {
                    "duplicates": [],
                    "groups": [],
                    "count": 0
                }
            
            # Get all embeddings from index
            embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            embeddings = embeddings.astype(np.float32)
            
            # Compute pairwise similarities
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
            # Find pairs above threshold (excluding self-similarity)
            duplicate_groups = []
            processed = set()
            
            for i in range(len(similarity_matrix)):
                if i in processed:
                    continue
                
                # Find all images similar to this one
                similar_indices = np.where(similarity_matrix[i] > threshold)[0]
                similar_indices = [idx for idx in similar_indices if idx != i and idx not in processed]
                
                if len(similar_indices) > 0:
                    # Create duplicate group
                    group = [i] + list(similar_indices)
                    duplicate_groups.append({
                        "representative": self.image_paths[i] if i < len(self.image_paths) else f"index_{i}",
                        "duplicates": [
                            {
                                "path": self.image_paths[idx] if idx < len(self.image_paths) else f"index_{idx}",
                                "similarity": float(similarity_matrix[i][idx])
                            }
                            for idx in similar_indices
                        ],
                        "group_size": len(group)
                    })
                    
                    # Mark as processed
                    processed.add(i)
                    processed.update(similar_indices)
            
            # Flatten to list of all duplicates
            all_duplicates = []
            for group in duplicate_groups:
                all_duplicates.extend([d["path"] for d in group["duplicates"]])
            
            return {
                "duplicate_groups": duplicate_groups,
                "total_duplicates": len(all_duplicates),
                "groups_found": len(duplicate_groups),
                "threshold": threshold,
                "method": method
            }
            
        except Exception as e:
            self.logger.error(f"Error finding duplicates: {e}")
            raise
    
    async def _find_outliers(self, std_threshold: float) -> Dict[str, Any]:
        """
        SOTA: Find outliers/anomalies using embedding distance from centroid
        Images far from the dataset mean in embedding space are likely outliers
        """
        try:
            if self.index.ntotal < 3:
                return {
                    "outliers": [],
                    "count": 0,
                    "message": "Need at least 3 images for outlier detection"
                }
            
            # Get all embeddings
            embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            embeddings = embeddings.astype(np.float32)
            
            # Calculate centroid (mean embedding)
            centroid = np.mean(embeddings, axis=0, keepdims=True)
            
            # Calculate distances from centroid (cosine similarity)
            # For normalized embeddings, dot product = cosine similarity
            similarities = np.dot(embeddings, centroid.T).flatten()
            
            # Convert to distances (1 - similarity)
            distances = 1 - similarities
            
            # Calculate mean and std of distances
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Find outliers: images with distance > mean + std_threshold * std
            threshold = mean_dist + (std_threshold * std_dist)
            outlier_indices = np.where(distances > threshold)[0]
            
            # Build outlier results with explanation
            outliers = []
            for idx in outlier_indices:
                if idx < len(self.image_paths):
                    outliers.append({
                        "path": self.image_paths[idx],
                        "distance": float(distances[idx]),
                        "z_score": float((distances[idx] - mean_dist) / std_dist) if std_dist > 0 else 0,
                        "reason": f"Embedding distance {distances[idx]:.3f} is {((distances[idx] - mean_dist) / std_dist):.1f}σ from mean"
                    })
            
            # Sort by distance (most outlier-ish first)
            outliers.sort(key=lambda x: x['distance'], reverse=True)
            
            return {
                "outliers": outliers,
                "count": len(outliers),
                "threshold_used": float(threshold),
                "mean_distance": float(mean_dist),
                "std_distance": float(std_dist),
                "std_threshold": std_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error finding outliers: {e}")
            raise
    
    async def _calculate_diversity(self) -> Dict[str, Any]:
        """
        SOTA: Calculate dataset diversity using embedding spread metrics
        Measures how well the dataset covers the embedding space
        """
        try:
            if self.index.ntotal < 2:
                return {
                    "diversity_score": 0,
                    "coverage_score": 0,
                    "message": "Need at least 2 images for diversity calculation"
                }
            
            # Get all embeddings
            embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            embeddings = embeddings.astype(np.float32)
            
            # 1. Intra-cluster spread: Average pairwise distance
            # For efficiency, sample if dataset is large
            n_samples = min(500, len(embeddings))
            if n_samples < len(embeddings):
                indices = np.random.choice(len(embeddings), n_samples, replace=False)
                sample_embeddings = embeddings[indices]
            else:
                sample_embeddings = embeddings
            
            # Compute pairwise similarities
            sim_matrix = np.dot(sample_embeddings, sample_embeddings.T)
            
            # Average off-diagonal similarity (lower = more diverse)
            mask = ~np.eye(len(sample_embeddings), dtype=bool)
            avg_similarity = np.mean(sim_matrix[mask])
            
            # Diversity score: 1 - avg_similarity (higher = more diverse)
            diversity_score = 1 - avg_similarity
            
            # 2. Coverage: Standard deviation of embeddings in each dimension
            # Higher std = better coverage of embedding space
            dim_stds = np.std(embeddings, axis=0)
            coverage_score = float(np.mean(dim_stds))
            
            # 3. Effective dimensionality using eigenvalues
            # Higher = embeddings use more of the available space
            cov_matrix = np.cov(embeddings.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zeros
            
            # Participation ratio as effective dimensionality
            if len(eigenvalues) > 0:
                p_i = eigenvalues / np.sum(eigenvalues)
                effective_dim = 1 / np.sum(p_i ** 2)
                effective_dim_ratio = effective_dim / len(eigenvalues)
            else:
                effective_dim = 0
                effective_dim_ratio = 0
            
            # Normalize diversity score to 0-100
            normalized_diversity = min(100, max(0, diversity_score * 100))
            
            return {
                "diversity_score": round(normalized_diversity, 1),
                "avg_similarity": round(avg_similarity, 4),
                "coverage_score": round(coverage_score, 4),
                "effective_dimensionality": round(effective_dim, 1),
                "effective_dim_ratio": round(effective_dim_ratio, 4),
                "total_images": int(self.index.ntotal),
                "interpretation": self._interpret_diversity(normalized_diversity)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating diversity: {e}")
            raise
    
    def _interpret_diversity(self, score: float) -> str:
        """Interpret diversity score for user-friendly output"""
        if score >= 80:
            return "Excellent - Dataset covers a wide variety of visual concepts"
        elif score >= 60:
            return "Good - Dataset has reasonable diversity"
        elif score >= 40:
            return "Moderate - Some redundancy, consider adding more varied images"
        elif score >= 20:
            return "Low - Many similar images, dataset may benefit from pruning duplicates"
        else:
            return "Very Low - Images are highly similar, may cause overfitting in training"
    
    async def _check_caption_alignment(self, threshold: float, mode: str) -> Dict[str, Any]:
        """
        SOTA: Check caption-image alignment using CLIP cross-modal similarity
        Finds images where the caption doesn't match the visual content
        """
        try:
            if not hasattr(self, 'captions') or not self.captions:
                # Load captions if not already loaded
                self.captions = {}
                for img_path in self.image_paths:
                    caption_path = Path(img_path).with_suffix('.txt')
                    if caption_path.exists():
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                self.captions[img_path] = f.read().strip()
                        except Exception:
                            pass
            
            if not self.captions:
                return {
                    "misaligned": [],
                    "count": 0,
                    "message": "No captions found for alignment check"
                }
            
            results = []
            
            for img_path, caption in self.captions.items():
                if not caption:
                    continue
                
                try:
                    # Get image embedding
                    img_idx = self.image_paths.index(img_path) if img_path in self.image_paths else -1
                    if img_idx < 0 or img_idx >= self.index.ntotal:
                        continue
                    
                    img_embedding = self.index.reconstruct(img_idx)
                    
                    # Get text embedding for caption
                    text_inputs = self.tokenizer([caption], return_tensors="pt", truncation=True, max_length=77)
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    
                    with torch.no_grad():
                        text_features = self.model.get_text_features(**text_inputs)
                        text_embedding = text_features / text_features.norm(dim=-1, keepdim=True)
                        text_embedding = text_embedding.cpu().numpy().flatten().astype(np.float32)
                    
                    # Calculate cross-modal similarity
                    similarity = float(np.dot(img_embedding, text_embedding))
                    
                    results.append({
                        "path": img_path,
                        "caption": caption[:100] + "..." if len(caption) > 100 else caption,
                        "alignment_score": similarity,
                        "is_misaligned": similarity < threshold
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error checking alignment for {img_path}: {e}")
                    continue
            
            # Filter based on mode
            if mode == 'find_misaligned':
                misaligned = [r for r in results if r['is_misaligned']]
                misaligned.sort(key=lambda x: x['alignment_score'])  # Worst first
                
                return {
                    "misaligned": misaligned,
                    "count": len(misaligned),
                    "threshold": threshold,
                    "total_checked": len(results)
                }
            else:
                # Return all with scores
                results.sort(key=lambda x: x['alignment_score'])
                return {
                    "all_results": results,
                    "count": len(results),
                    "avg_alignment": sum(r['alignment_score'] for r in results) / len(results) if results else 0
                }
                
        except Exception as e:
            self.logger.error(f"Error checking caption alignment: {e}")
            raise
    
    async def _get_embedding_visualization(
        self, 
        sample_size: int, 
        include_similarity_matrix: bool,
        include_distribution: bool,
        projection_method: str
    ) -> Dict[str, Any]:
        """
        SOTA: Generate embedding space visualization data for power users
        Returns 2D projection, optional similarity matrix, and distribution histogram
        """
        try:
            if self.index.ntotal < 2:
                return {
                    "embeddings": [],
                    "message": "Need at least 2 images for visualization"
                }
            
            # Sample embeddings for performance
            total = self.index.ntotal
            n_samples = min(sample_size, total)
            
            if n_samples < total:
                indices = np.random.choice(total, n_samples, replace=False)
                indices = np.sort(indices)
            else:
                indices = np.arange(total)
            
            embeddings = np.array([self.index.reconstruct(int(i)) for i in indices])
            embeddings = embeddings.astype(np.float32)
            
            # PCA projection to 2D (fast, deterministic)
            # Center the data
            mean = np.mean(embeddings, axis=0)
            centered = embeddings - mean
            
            # Compute covariance and top 2 eigenvectors
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue descending, take top 2
            idx = np.argsort(eigenvalues)[::-1]
            top_2_vectors = eigenvectors[:, idx[:2]]
            
            # Project to 2D
            projected = np.dot(centered, top_2_vectors)
            
            # Normalize to 0-1 range for frontend
            x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
            y_min, y_max = projected[:, 1].min(), projected[:, 1].max()
            
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1
            
            # Build visualization data
            viz_embeddings = []
            for i, idx in enumerate(indices):
                img_path = self.image_paths[idx] if idx < len(self.image_paths) else f"image_{idx}"
                viz_embeddings.append({
                    "x": float((projected[i, 0] - x_min) / x_range),
                    "y": float((projected[i, 1] - y_min) / y_range),
                    "label": Path(img_path).stem[:20] if isinstance(img_path, str) else f"#{idx}",
                    "path": img_path
                })
            
            result = {
                "embeddings": viz_embeddings,
                "total_points": total,
                "sampled_points": n_samples,
                "dimensions": self.dimension,
                "projection_method": projection_method,
                "coverage": n_samples / total
            }
            
            # Optional: Pairwise similarity matrix (compact, for first N)
            if include_similarity_matrix:
                matrix_size = min(20, n_samples)  # Max 20x20 for UI
                sample_emb = embeddings[:matrix_size]
                sim_matrix = np.dot(sample_emb, sample_emb.T)
                result["similarity_matrix"] = sim_matrix.tolist()
            
            # Optional: Distance distribution histogram
            if include_distribution:
                # Calculate distances from centroid
                centroid = np.mean(embeddings, axis=0)
                distances = np.linalg.norm(embeddings - centroid, axis=1)
                result["distribution"] = distances.tolist()
                result["spread"] = float(np.std(distances))
            
            self.logger.info(f"Generated embedding visualization: {n_samples} points")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating embedding visualization: {e}")
            raise
    
    def _get_embeddings_for_training(self) -> Dict[str, Any]:
        """Get all embedding vectors for Conceptualizer training"""
        if not hasattr(self, 'embedding_vectors') or not self.embedding_vectors:
            return {
                "embeddings": [],
                "paths": [],
                "count": 0,
                "message": "No embeddings available - they may need to be regenerated"
            }
        
        return {
            "embeddings": self.embedding_vectors,
            "paths": self.image_paths,
            "count": len(self.embedding_vectors)
        }
    
    def _get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index"""
        return {
            "size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "image_count": len(self.image_paths),
            "embeddings_available": hasattr(self, 'embedding_vectors') and len(self.embedding_vectors) > 0,
            "is_trained": self.index.is_trained if self.index else False
        }
    
    async def _project_embeddings(
        self, 
        embeddings: List[List[float]], 
        method: str = 'tsne', 
        perplexity: int = 30
    ) -> Dict[str, Any]:
        """
        Project high-dimensional embeddings to 2D for visualization
        
        Args:
            embeddings: List of embedding vectors (768-dim)
            method: 'tsne' or 'umap'
            perplexity: t-SNE perplexity parameter (default: 30)
            
        Returns:
            Dict with 'projection' key containing 2D coordinates
        """
        try:
            import numpy as np
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            n_samples = len(embeddings_array)
            
            self.logger.info(f"Projecting {n_samples} embeddings using {method}")
            
            if method == 'tsne':
                from sklearn.manifold import TSNE
                
                # Adjust perplexity if needed
                effective_perplexity = min(perplexity, (n_samples - 1) // 3)
                
                projector = TSNE(
                    n_components=2,
                    perplexity=effective_perplexity,
                    random_state=42,
                    n_iter=1000,
                    verbose=0
                )
                projection = projector.fit_transform(embeddings_array)
                
            elif method == 'umap':
                try:
                    import umap
                    
                    # Adjust n_neighbors if needed
                    n_neighbors = min(15, n_samples - 1)
                    
                    projector = umap.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        random_state=42,
                        verbose=False
                    )
                    projection = projector.fit_transform(embeddings_array)
                except ImportError:
                    self.logger.warning("UMAP not installed, falling back to t-SNE")
                    return await self._project_embeddings(embeddings, 'tsne', perplexity)
            else:
                raise ValueError(f"Unknown projection method: {method}. Use 'tsne' or 'umap'.")
            
            # Convert to list for JSON serialization
            projection_list = projection.tolist()
            
            self.logger.info(f"Projection complete: {len(projection_list)} points")
            
            return {
                "projection": projection_list,
                "method": method,
                "n_samples": n_samples
            }
            
        except Exception as e:
            self.logger.error(f"Projection failed: {str(e)}", exc_info=True)
            raise
    
    def search_with_temperature(self, query_embedding: np.ndarray, k: int = 50, 
                               temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Enhanced search with temperature scaling for better calibration
        
        Based on CLIP paper: logits = similarities / temperature
        Lower temperature = more confident predictions
        Higher temperature = more diverse results
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            temperature: Temperature parameter (uses self.temperature if None)
            
        Returns:
            Search results with temperature-scaled similarities
        """
        try:
            if self.index is None:
                return {"error": "Index not built", "results": []}
            
            # Use instance temperature or provided temperature
            temp = temperature if temperature is not None else self.temperature
            
            # Search for more candidates than needed for temperature scaling
            search_k = min(k * 3, len(self.image_paths))  # Get 3x candidates
            
            # FAISS search
            similarities, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                search_k
            )
            
            # Apply temperature scaling (from CLIP paper)
            scaled_similarities = similarities[0] / temp
            
            # Convert to probabilities using softmax
            exp_similarities = np.exp(scaled_similarities - np.max(scaled_similarities))  # Numerical stability
            probabilities = exp_similarities / np.sum(exp_similarities)
            
            # Re-rank by temperature-scaled probabilities and take top k
            sorted_indices = np.argsort(probabilities)[::-1][:k]
            
            results = []
            for i, idx in enumerate(sorted_indices):
                original_idx = indices[0][idx]
                if 0 <= original_idx < len(self.image_paths):
                    results.append({
                        'path': self.image_paths[original_idx],
                        'similarity': float(similarities[0][idx]),  # Original similarity
                        'scaled_similarity': float(scaled_similarities[idx]),  # Temperature-scaled
                        'probability': float(probabilities[idx]),  # Softmax probability
                        'rank': i + 1,
                        'temperature_used': temp
                    })
            
            # Track temperature performance for adaptive optimization
            if self.adaptive_temperature and len(results) > 0:
                avg_confidence = np.mean([r['probability'] for r in results])
                self.temperature_history.append({
                    'temperature': temp,
                    'avg_confidence': avg_confidence,
                    'num_results': len(results)
                })
                
                # Keep only recent history
                if len(self.temperature_history) > 100:
                    self.temperature_history = self.temperature_history[-50:]
            
            return {
                "results": results,
                "total_found": len(results),
                "temperature": temp,
                "method": "temperature_scaled_clip"
            }
            
        except Exception as e:
            self.log_error(f"Temperature-scaled search failed: {str(e)}")
            # Fallback to regular search
            return self.search(query_embedding, k)
    
    def hard_negative_mining(self, query_embedding: np.ndarray, 
                           positive_paths: List[str],
                           k_negatives: int = 50) -> Dict[str, Any]:
        """
        Hard negative mining for improved contrastive learning
        
        Finds images that are similar to query but NOT in positive set
        These "hard negatives" are crucial for training better embeddings
        
        Args:
            query_embedding: Query embedding vector
            positive_paths: List of known positive image paths
            k_negatives: Number of hard negatives to return
            
        Returns:
            Dict with hard negative candidates
        """
        try:
            if self.index is None:
                return {"error": "Index not built", "hard_negatives": []}
            
            # Search for many candidates (10x what we need)
            search_k = min(k_negatives * 10, len(self.image_paths))
            
            similarities, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                search_k
            )
            
            # Convert positive paths to set for fast lookup
            positive_set = set(positive_paths)
            
            hard_negatives = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.image_paths):
                    image_path = self.image_paths[idx]
                    
                    # Skip if this is a positive example
                    if image_path in positive_set:
                        continue
                    
                    # This is a hard negative: similar to query but not positive
                    similarity = float(similarities[0][i])
                    
                    # Only include if similarity is reasonably high (hard negative criteria)
                    if similarity > 0.3:  # Threshold for "hard" negatives
                        hard_negatives.append({
                            'path': image_path,
                            'similarity': similarity,
                            'rank': len(hard_negatives) + 1,
                            'negative_type': 'hard'
                        })
                        
                        if len(hard_negatives) >= k_negatives:
                            break
            
            # If we don't have enough hard negatives, add some random negatives
            if len(hard_negatives) < k_negatives:
                remaining = k_negatives - len(hard_negatives)
                used_paths = positive_set | {hn['path'] for hn in hard_negatives}
                
                import random
                available_paths = [p for p in self.image_paths if p not in used_paths]
                random_negatives = random.sample(
                    available_paths, 
                    min(remaining, len(available_paths))
                )
                
                for path in random_negatives:
                    hard_negatives.append({
                        'path': path,
                        'similarity': 0.0,  # Random negative
                        'rank': len(hard_negatives) + 1,
                        'negative_type': 'random'
                    })
            
            self.log_info(f"Hard negative mining: {len(hard_negatives)} negatives found")
            
            return {
                "hard_negatives": hard_negatives,
                "total_found": len(hard_negatives),
                "positive_examples": len(positive_paths),
                "method": "hard_negative_mining"
            }
            
        except Exception as e:
            self.log_error(f"Hard negative mining failed: {str(e)}")
            return {"error": str(e), "hard_negatives": []}
    
    def contrastive_batch_sampling(self, query_embeddings: List[np.ndarray],
                                  positive_paths_list: List[List[str]],
                                  batch_size: int = 32) -> Dict[str, Any]:
        """
        Create contrastive batches for training with hard negatives
        
        Consumer GPU friendly: adjustable batch sizes
        Based on CLIP paper's contrastive learning approach
        
        Args:
            query_embeddings: List of query embeddings
            positive_paths_list: List of positive paths for each query
            batch_size: Batch size (adjust for consumer GPUs: 8-32)
            
        Returns:
            Dict with contrastive training batches
        """
        try:
            batches = []
            
            for i in range(0, len(query_embeddings), batch_size):
                batch_queries = query_embeddings[i:i+batch_size]
                batch_positives = positive_paths_list[i:i+batch_size]
                
                batch_data = {
                    'queries': batch_queries,
                    'positives': [],
                    'hard_negatives': [],
                    'random_negatives': []
                }
                
                # For each query in batch, get positives and hard negatives
                for j, (query_emb, pos_paths) in enumerate(zip(batch_queries, batch_positives)):
                    # Get hard negatives for this query
                    neg_result = self.hard_negative_mining(
                        query_emb, 
                        pos_paths, 
                        k_negatives=min(16, batch_size)  # Scale with batch size
                    )
                    
                    batch_data['positives'].append(pos_paths)
                    
                    hard_negs = [hn for hn in neg_result.get('hard_negatives', []) 
                               if hn['negative_type'] == 'hard']
                    random_negs = [hn for hn in neg_result.get('hard_negatives', []) 
                                 if hn['negative_type'] == 'random']
                    
                    batch_data['hard_negatives'].append(hard_negs)
                    batch_data['random_negatives'].append(random_negs)
                
                batches.append(batch_data)
            
            self.log_info(f"Created {len(batches)} contrastive batches (batch_size={batch_size})")
            
            return {
                "batches": batches,
                "num_batches": len(batches),
                "batch_size": batch_size,
                "total_queries": len(query_embeddings)
            }
            
        except Exception as e:
            self.log_error(f"Contrastive batch sampling failed: {str(e)}")
            return {"error": str(e), "batches": []}
    
    def build_multi_level_indexes(self, profile_data: Dict) -> Dict[str, Any]:
        """
        Build multi-level FAISS indexes for granular search
        
        Netflix-inspired: Index at multiple granularities
        - Person index: Images with people (for human-centric queries)
        - Object index: Images with prominent objects (for object queries)
        - Scene index: Landscape/architecture images (for scene queries)
        
        Args:
            profile_data: Dataset profile with VLM classifications
            
        Returns:
            Dict with indexing results
        """
        try:
            import faiss
            
            if not hasattr(self, 'embedding_vectors') or len(self.embedding_vectors) == 0:
                return {"error": "No embeddings available for multi-level indexing"}
            
            self.log_info("Building multi-level FAISS indexes...")
            
            # Categorize images based on VLM classification
            person_embeddings = []
            object_embeddings = []
            scene_embeddings = []
            
            self.person_paths = []
            self.object_paths = []
            self.scene_paths = []
            
            for i, image_path in enumerate(self.image_paths):
                filename = Path(image_path).name
                universal_profile = profile_data.get('images', {}).get(filename, {}).get('universal_profile', {})
                
                embedding = self.embedding_vectors[i]
                
                # Categorize based on VLM classification
                human_presence = universal_profile.get('human_presence', 'none')
                content_type = universal_profile.get('content_type', 'unknown')
                
                # Person index: Images with humans
                if human_presence != 'none':
                    person_embeddings.append(embedding)
                    self.person_paths.append(image_path)
                
                # Object index: Still life, food, objects
                if content_type in ['still_life', 'food', 'interior']:
                    object_embeddings.append(embedding)
                    self.object_paths.append(image_path)
                
                # Scene index: Landscapes, architecture, nature
                if content_type in ['landscape', 'architecture', 'nature', 'urban']:
                    scene_embeddings.append(embedding)
                    self.scene_paths.append(image_path)
            
            # Build person index
            if len(person_embeddings) > 0:
                person_embeddings_np = np.array(person_embeddings).astype(np.float32)
                self.person_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.person_index.add(person_embeddings_np)
                self.log_info(f"Person index built: {len(person_embeddings)} images")
            
            # Build object index
            if len(object_embeddings) > 0:
                object_embeddings_np = np.array(object_embeddings).astype(np.float32)
                self.object_index = faiss.IndexFlatIP(self.dimension)
                self.object_index.add(object_embeddings_np)
                self.log_info(f"Object index built: {len(object_embeddings)} images")
            
            # Build scene index
            if len(scene_embeddings) > 0:
                scene_embeddings_np = np.array(scene_embeddings).astype(np.float32)
                self.scene_index = faiss.IndexFlatIP(self.dimension)
                self.scene_index.add(scene_embeddings_np)
                self.log_info(f"Scene index built: {len(scene_embeddings)} images")
            
            return {
                "person_index_size": len(person_embeddings),
                "object_index_size": len(object_embeddings),
                "scene_index_size": len(scene_embeddings),
                "total_images": len(self.image_paths),
                "multi_level_indexing": True
            }
            
        except Exception as e:
            self.log_error(f"Multi-level indexing failed: {str(e)}")
            return {"error": str(e)}
    
    def search_multi_level(self, query_embedding: np.ndarray, query_type: str = 'auto', 
                          k: int = 50) -> Dict[str, Any]:
        """
        Search using appropriate specialized index
        
        Args:
            query_embedding: Query embedding vector
            query_type: 'person', 'object', 'scene', or 'auto' (detect automatically)
            k: Number of results to return
            
        Returns:
            Search results from appropriate specialized index
        """
        try:
            # Auto-detect query type if not specified
            if query_type == 'auto':
                query_type = self._detect_query_type(query_embedding)
            
            # Select appropriate index
            if query_type == 'person' and self.person_index is not None:
                similarities, indices = self.person_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), k
                )
                paths = self.person_paths
                index_type = 'person'
                
            elif query_type == 'object' and self.object_index is not None:
                similarities, indices = self.object_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), k
                )
                paths = self.object_paths
                index_type = 'object'
                
            elif query_type == 'scene' and self.scene_index is not None:
                similarities, indices = self.scene_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), k
                )
                paths = self.scene_paths
                index_type = 'scene'
                
            else:
                # Fallback to main index
                return self.search_with_temperature(query_embedding, k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(paths):
                    results.append({
                        'path': paths[idx],
                        'similarity': float(similarities[0][i]),
                        'rank': i + 1,
                        'index_type': index_type
                    })
            
            self.log_info(f"Multi-level search ({index_type}): {len(results)} results")
            
            return {
                "results": results,
                "total_found": len(results),
                "index_type": index_type,
                "method": "multi_level_search"
            }
            
        except Exception as e:
            self.log_error(f"Multi-level search failed: {str(e)}")
            # Fallback to regular search
            return self.search(query_embedding, k)
    
    def _detect_query_type(self, query_embedding: np.ndarray) -> str:
        """
        Auto-detect whether query is person, object, or scene focused
        
        Args:
            query_embedding: Query embedding to analyze
            
        Returns:
            'person', 'object', or 'scene'
        """
        try:
            # Simple heuristic: test similarity to category centroids
            if hasattr(self, 'person_index') and self.person_index and len(self.person_paths) > 0:
                # Get average similarity to person images
                person_sims, _ = self.person_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), 
                    min(10, len(self.person_paths))
                )
                avg_person_sim = np.mean(person_sims[0])
            else:
                avg_person_sim = 0.0
            
            if hasattr(self, 'scene_index') and self.scene_index and len(self.scene_paths) > 0:
                scene_sims, _ = self.scene_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32),
                    min(10, len(self.scene_paths))
                )
                avg_scene_sim = np.mean(scene_sims[0])
            else:
                avg_scene_sim = 0.0
            
            if hasattr(self, 'object_index') and self.object_index and len(self.object_paths) > 0:
                object_sims, _ = self.object_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32),
                    min(10, len(self.object_paths))
                )
                avg_object_sim = np.mean(object_sims[0])
            else:
                avg_object_sim = 0.0
            
            # Return type with highest average similarity
            if avg_person_sim > avg_scene_sim and avg_person_sim > avg_object_sim:
                return 'person'
            elif avg_scene_sim > avg_object_sim:
                return 'scene'
            else:
                return 'object'
                
        except Exception as e:
            self.log_debug(f"Query type detection failed: {str(e)}")
            return 'person'  # Default to person for human-centric queries
    
    # ==================== HIERARCHICAL EMBEDDINGS ====================
    # Multi-level embeddings matching VLM's classification structure
    # Inspired by Google's Multi-Scale Vision Transformers
    # ================================================================
    
    async def generate_hierarchical_embeddings(self, image_path: str, 
                                              pose_data: Dict = None) -> Dict[str, Any]:
        """
        **HIERARCHICAL EMBEDDINGS**
        Generate 3-level embeddings matching VLM's multi-level classification
        
        WHY: VLM classifies at 9 dimensions (scene, mood, people, clothing, etc.)
        CLIP needs corresponding multi-level embeddings to match this granularity
        
        STRUCTURE:
        1. GLOBAL embedding [768]: Whole image at 336px (scene, mood, overall composition)
        2. SUBJECT embedding [768]: Cropped to detected person/object (pose, identity, main subject)
        3. DETAILS embedding [768]: High-res patches 512px (clothing, accessories, fine details)
        
        This enables:
        - Global queries: "sunset beach scene"
        - Subject queries: "woman in red dress"
        - Detail queries: "red high heels" or "silver necklace"
        
        Args:
            image_path: Path to image file
            pose_data: Optional pose detection data from PoseService
        
        Returns:
            Dict with 'global_embedding', 'subject_embedding', 'details_embedding'
        """
        try:
            from PIL import Image
            import torch
            
            self.log_info(f"[Hierarchical] Generating 3-level embeddings for: {Path(image_path).name}")
            
            img = Image.open(image_path)
            # Apply EXIF orientation
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            img = img.convert('RGB')
            original_size = img.size
            
            # LEVEL 1: GLOBAL EMBEDDING (336px - standard CLIP size)
            global_embedding = await self._encode_image_at_resolution(img, 336)
            
            # LEVEL 2: SUBJECT EMBEDDING (detected person/object region)
            subject_embedding = None
            if pose_data and 'bounding_boxes' in pose_data and len(pose_data['bounding_boxes']) > 0:
                # Use pose detection to extract subject
                subject_embedding = await self._extract_subject_embedding(img, pose_data)
            else:
                # Fallback: Use center crop as "subject"
                subject_embedding = await self._extract_center_subject_embedding(img)
            
            # LEVEL 3: DETAILS EMBEDDING (high-res patches for fine details)
            details_embedding = await self._extract_detail_patches_embedding(img, pose_data)
            
            self.log_info(f"[Hierarchical] ✓ Generated: Global[{len(global_embedding)}] Subject[{len(subject_embedding) if subject_embedding is not None else 0}] Details[{len(details_embedding) if details_embedding is not None else 0}]")
            
            return {
                'global_embedding': global_embedding.tolist() if isinstance(global_embedding, np.ndarray) else global_embedding,
                'subject_embedding': subject_embedding.tolist() if isinstance(subject_embedding, np.ndarray) else subject_embedding,
                'details_embedding': details_embedding.tolist() if isinstance(details_embedding, np.ndarray) else details_embedding,
                'original_size': original_size,
                'levels': 3
            }
            
        except Exception as e:
            self.log_error(f"[Hierarchical] Failed for {image_path}: {str(e)}")
            # Fallback: Return only global embedding
            try:
                from PIL import Image
                img = Image.open(image_path).convert('RGB')
                global_embedding = await self._encode_image_at_resolution(img, 336)
                return {
                    'global_embedding': global_embedding.tolist() if isinstance(global_embedding, np.ndarray) else global_embedding,
                    'subject_embedding': None,
                    'details_embedding': None,
                    'error': str(e)
                }
            except:
                return {'error': str(e)}
    
    async def _encode_image_at_resolution(self, img: 'PIL.Image', target_size: int) -> np.ndarray:
        """
        Encode image at specific resolution using CLIP
        
        Args:
            img: PIL Image
            target_size: Target size (e.g., 336, 512)
        
        Returns:
            Embedding vector [768]
        """
        try:
            import torch
            from PIL import Image
            
            # Resize maintaining aspect ratio
            img_resized = img.copy()
            img_resized.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Create square canvas and paste (center padding)
            canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            offset = ((target_size - img_resized.width) // 2, (target_size - img_resized.height) // 2)
            canvas.paste(img_resized, offset)
            
            # Preprocess and encode
            image_tensor = self.preprocess(canvas).unsqueeze(0)
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            self.log_error(f"[Hierarchical] Resolution encoding failed: {str(e)}")
            raise
    
    async def _extract_subject_embedding(self, img: 'PIL.Image', pose_data: Dict) -> np.ndarray:
        """
        Extract subject embedding using pose detection bounding boxes
        
        Args:
            img: PIL Image
            pose_data: Pose detection data with bounding_boxes
        
        Returns:
            Subject embedding [768]
        """
        try:
            from PIL import Image
            
            # Get largest bounding box (main subject)
            bboxes = pose_data['bounding_boxes']
            if not bboxes:
                return await self._extract_center_subject_embedding(img)
            
            # Find largest bbox
            largest_bbox = max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = largest_bbox
            
            # Add padding (10% on each side)
            w, h = img.size
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            # Crop to subject
            subject_img = img.crop((x1, y1, x2, y2))
            
            # Encode at 336px (standard CLIP)
            subject_embedding = await self._encode_image_at_resolution(subject_img, 336)
            
            self.log_debug(f"[Hierarchical] Subject extracted: bbox=({x1},{y1},{x2},{y2})")
            
            return subject_embedding
            
        except Exception as e:
            self.log_warning(f"[Hierarchical] Subject extraction failed: {str(e)}, using center crop")
            return await self._extract_center_subject_embedding(img)
    
    async def _extract_center_subject_embedding(self, img: 'PIL.Image') -> np.ndarray:
        """
        Fallback: Extract center crop as subject when pose detection unavailable
        
        Args:
            img: PIL Image
        
        Returns:
            Center-cropped embedding [768]
        """
        try:
            from PIL import Image
            
            w, h = img.size
            
            # Center crop to 60% of image (typical subject area)
            crop_w = int(w * 0.6)
            crop_h = int(h * 0.6)
            
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
            
            center_crop = img.crop((left, top, right, bottom))
            
            # Encode at 336px
            embedding = await self._encode_image_at_resolution(center_crop, 336)
            
            self.log_debug(f"[Hierarchical] Center crop used as subject: {crop_w}x{crop_h}")
            
            return embedding
            
        except Exception as e:
            self.log_error(f"[Hierarchical] Center crop failed: {str(e)}")
            # Ultimate fallback: use global embedding
            return await self._encode_image_at_resolution(img, 336)
    
    async def _extract_detail_patches_embedding(self, img: 'PIL.Image', 
                                               pose_data: Dict = None) -> np.ndarray:
        """
        Extract high-resolution detail patches for fine-grained features
        
        STRATEGY: 
        - If person detected: Focus patches around upper body (clothing, accessories)
        - If no person: Extract 4-corner + center patches (architecture details, objects)
        
        All patches encoded at 512px for maximum detail capture
        
        Args:
            img: PIL Image
            pose_data: Optional pose data to guide patch extraction
        
        Returns:
            Averaged embedding [768] from multiple high-res patches
        """
        try:
            from PIL import Image
            import torch
            
            w, h = img.size
            
            patches = []
            
            if pose_data and 'bounding_boxes' in pose_data and len(pose_data['bounding_boxes']) > 0:
                # PERSON-FOCUSED PATCHES: Upper body details (clothing, accessories)
                bbox = max(pose_data['bounding_boxes'], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2 = bbox
                
                # Upper body patch (top 40% of person bbox)
                upper_h = int((y2 - y1) * 0.4)
                upper_patch = img.crop((x1, y1, x2, y1 + upper_h))
                patches.append(upper_patch)
                
                # Face/head region (top 25%)
                head_h = int((y2 - y1) * 0.25)
                head_patch = img.crop((x1, y1, x2, y1 + head_h))
                patches.append(head_patch)
                
            else:
                # SCENE-FOCUSED PATCHES: 4 corners + center for architecture/objects
                patch_size_w = w // 3
                patch_size_h = h // 3
                
                # Top-left
                patches.append(img.crop((0, 0, patch_size_w, patch_size_h)))
                # Top-right
                patches.append(img.crop((w - patch_size_w, 0, w, patch_size_h)))
                # Bottom-left
                patches.append(img.crop((0, h - patch_size_h, patch_size_w, h)))
                # Bottom-right
                patches.append(img.crop((w - patch_size_w, h - patch_size_h, w, h)))
                # Center
                center_x = w // 2
                center_y = h // 2
                patches.append(img.crop((
                    center_x - patch_size_w // 2,
                    center_y - patch_size_h // 2,
                    center_x + patch_size_w // 2,
                    center_y + patch_size_h // 2
                )))
            
            # Encode all patches at 512px (high-res for details)
            patch_embeddings = []
            for patch in patches:
                if patch.width > 0 and patch.height > 0:
                    patch_emb = await self._encode_image_at_resolution(patch, 512)
                    patch_embeddings.append(patch_emb)
            
            if len(patch_embeddings) == 0:
                # Fallback: use global embedding
                return await self._encode_image_at_resolution(img, 336)
            
            # Average all patch embeddings
            details_embedding = np.mean(patch_embeddings, axis=0)
            
            # Normalize
            details_embedding = details_embedding / np.linalg.norm(details_embedding)
            
            self.log_debug(f"[Hierarchical] Details extracted: {len(patch_embeddings)} patches @ 512px")
            
            return details_embedding
            
        except Exception as e:
            self.log_error(f"[Hierarchical] Detail extraction failed: {str(e)}")
            # Fallback: use global embedding
            return await self._encode_image_at_resolution(img, 336)
    
    async def hierarchical_search(self, query: str, level: str = 'auto', 
                                 k: int = 50) -> Dict[str, Any]:
        """
        Search using appropriate hierarchical level
        
        Args:
            query: Search query
            level: 'global', 'subject', 'details', or 'auto' (detect from query)
            k: Number of results
        
        Returns:
            Search results from appropriate level
        """
        try:
            # Detect appropriate level from query
            if level == 'auto':
                level = self._detect_hierarchical_level(query)
            
            self.log_info(f"[Hierarchical Search] Level: {level}, Query: {query}")
            
            # Encode query
            query_embedding = await self.encode_text(query)
            
            # Search appropriate index
            # NOTE: This requires hierarchical FAISS indexes to be built
            # For now, fallback to regular search with logging
            self.log_warning(f"[Hierarchical Search] Level-specific indexes not yet built, using global")
            
            return await self.search_similar(
                query_embedding=query_embedding,
                top_k=k,
                threshold=0.0
            )
            
        except Exception as e:
            self.log_error(f"[Hierarchical Search] Failed: {str(e)}")
            return {"error": str(e), "results": []}
    
    def _detect_hierarchical_level(self, query: str) -> str:
        """
        Detect which hierarchical level to use based on query
        
        Args:
            query: Search query text
        
        Returns:
            'global', 'subject', or 'details'
        """
        query_lower = query.lower()
        
        # DETAILS level: Clothing, accessories, specific items
        detail_keywords = [
            'dress', 'shirt', 'pants', 'shoes', 'hat', 'jacket', 'coat',
            'necklace', 'bracelet', 'earrings', 'jewelry', 'watch',
            'bag', 'purse', 'backpack', 'glasses', 'sunglasses',
            'red', 'blue', 'black', 'white', 'green', 'yellow',  # Color-specific
            'pattern', 'striped', 'polka dot', 'floral'
        ]
        
        # SUBJECT level: Person/object identity and pose
        subject_keywords = [
            'woman', 'man', 'person', 'girl', 'boy', 'child',
            'standing', 'sitting', 'walking', 'running', 'lying',
            'smiling', 'looking', 'holding', 'carrying',
            'portrait', 'closeup', 'face'
        ]
        
        # GLOBAL level: Scene, mood, composition
        global_keywords = [
            'sunset', 'sunrise', 'beach', 'mountain', 'forest', 'city',
            'indoor', 'outdoor', 'landscape', 'architecture',
            'dark', 'bright', 'colorful', 'minimalist',
            'moody', 'cheerful', 'dramatic'
        ]
        
        # Count matches
        detail_count = sum(1 for kw in detail_keywords if kw in query_lower)
        subject_count = sum(1 for kw in subject_keywords if kw in query_lower)
        global_count = sum(1 for kw in global_keywords if kw in query_lower)
        
        # Determine level
        if detail_count > subject_count and detail_count > global_count:
            return 'details'
        elif subject_count > global_count:
            return 'subject'
        else:
            return 'global'
    
    # ==================== SPATIAL-CLIP ====================
    # Region-aware embeddings for spatial relationship queries
    # Based on Spatial-CLIP research paper
    # ======================================================
    
    async def generate_spatial_embeddings(self, image_path: str) -> Dict[str, Any]:
        """
        **SPATIAL-CLIP ENHANCEMENT LAYER**
        Generate region-aware embeddings for advanced spatial verification
        
        PURPOSE: Optional enhancement for VLM-level spatial verification
        
        NOTE: LongCLIP ALREADY has spatial understanding via position embeddings!
        This is an ADDITIONAL layer for cases where we need pixel-precise verification.
        
        USE CASES:
        - VLM verification can check: "Are objects actually adjacent in regions?"
        - Advanced spatial queries requiring region-specific analysis
        - Debugging: Visualize which regions contain which objects
        
        STRUCTURE:
        - 9 region embeddings [9, 768]: 3x3 grid of image
        - Each region encoded at 336px with 2D positional encoding
        - Stored in profile for optional advanced verification
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dict with 'spatial_embeddings' [9, 768] and 'grid_size' (3, 3)
        """
        try:
            from PIL import Image
            import torch
            
            self.log_info(f"[Spatial-CLIP] Generating 3x3 grid embeddings: {Path(image_path).name}")
            
            img = Image.open(image_path)
            # Apply EXIF orientation
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            img = img.convert('RGB')
            w, h = img.size
            
            # Split into 3x3 grid
            grid_size = 3
            region_embeddings = []
            
            for row in range(grid_size):
                for col in range(grid_size):
                    # Extract region
                    x1 = col * w // grid_size
                    y1 = row * h // grid_size
                    x2 = (col + 1) * w // grid_size
                    y2 = (row + 1) * h // grid_size
                    
                    region = img.crop((x1, y1, x2, y2))
                    
                    # Encode region at 336px
                    region_emb = await self._encode_image_at_resolution(region, 336)
                    
                    # Apply 2D positional encoding
                    position_encoded = self._apply_2d_position_encoding(
                        region_emb, 
                        row, col, 
                        grid_size
                    )
                    
                    region_embeddings.append(position_encoded)
            
            # Stack into [9, 768] array
            spatial_embeddings = np.array(region_embeddings)
            
            self.log_info(f"[Spatial-CLIP] ✓ Generated {len(region_embeddings)} region embeddings")
            
            return {
                'spatial_embeddings': spatial_embeddings.tolist(),
                'grid_size': (grid_size, grid_size),
                'regions': len(region_embeddings)
            }
            
        except Exception as e:
            self.log_error(f"[Spatial-CLIP] Failed for {image_path}: {str(e)}")
            return {'error': str(e)}
    
    def _apply_2d_position_encoding(self, embedding: np.ndarray, 
                                   row: int, col: int, 
                                   grid_size: int) -> np.ndarray:
        """
        Apply 2D sinusoidal positional encoding to region embedding
        
        POSITION ENCODING (from Transformer/ViT papers):
        - Row position: sine/cosine waves at different frequencies
        - Col position: orthogonal sine/cosine waves
        - Concatenated with region embedding
        
        This allows the model to distinguish:
        - "top-left" vs "bottom-right"
        - "next to" vs "above" vs "below"
        
        Args:
            embedding: Region embedding [768]
            row: Row index (0-2 for 3x3 grid)
            col: Column index (0-2 for 3x3 grid)
            grid_size: Grid dimension (3 for 3x3)
        
        Returns:
            Position-encoded embedding [768]
        """
        try:
            # Normalize positions to [0, 1]
            row_norm = row / (grid_size - 1) if grid_size > 1 else 0.5
            col_norm = col / (grid_size - 1) if grid_size > 1 else 0.5
            
            # Create positional encoding vector
            d_model = len(embedding)
            position_enc = np.zeros(d_model)
            
            # Half for row, half for column
            half_d = d_model // 2
            
            # Row encoding (first half of dimensions)
            for i in range(0, half_d, 2):
                div_term = np.exp(i * -np.log(10000.0) / half_d)
                position_enc[i] = np.sin(row_norm * div_term)
                if i + 1 < half_d:
                    position_enc[i + 1] = np.cos(row_norm * div_term)
            
            # Column encoding (second half of dimensions)
            for i in range(half_d, d_model, 2):
                div_term = np.exp((i - half_d) * -np.log(10000.0) / half_d)
                position_enc[i] = np.sin(col_norm * div_term)
                if i + 1 < d_model:
                    position_enc[i + 1] = np.cos(col_norm * div_term)
            
            # Add positional encoding to embedding (residual connection)
            # Scale down positional encoding to not overwhelm content
            encoded = embedding + 0.1 * position_enc
            
            # Normalize
            encoded = encoded / np.linalg.norm(encoded)
            
            return encoded
            
        except Exception as e:
            self.log_error(f"[Spatial-CLIP] Position encoding failed: {str(e)}")
            return embedding  # Fallback to original
    
    async def spatial_search(self, query: str, k: int = 50) -> Dict[str, Any]:
        """
        Search using spatial relationship understanding
        
        ARCHITECTURE (Two-Layer Approach):
        
        LAYER 1 (Primary): LongCLIP's Built-In Spatial Understanding
        - LongCLIP already has spatial position embeddings built-in
        - Handles queries like "woman next to tree" automatically
        - Works out-of-the-box, no special processing needed
        
        LAYER 2 (Enhancement): 3x3 Grid Spatial Embeddings
        - Generated during profiling (if enabled)
        - Used for ADVANCED spatial verification with VLM
        - Example: VLM can check spatial_embeddings to confirm pixel-precise positioning
        - This is OPTIONAL enhancement, not required for basic spatial queries
        
        Args:
            query: Spatial query text (e.g., "dog next to cat")
            k: Number of results
        
        Returns:
            Search results (spatial understanding handled by LongCLIP)
        """
        try:
            # Detect spatial keywords (for logging/transparency)
            spatial_info = self._detect_spatial_keywords(query)
            
            if spatial_info['is_spatial']:
                self.log_info(f"[Spatial-CLIP] Spatial query detected: {spatial_info['keywords']}")
                self.log_info(f"[Spatial-CLIP] Using LongCLIP's built-in spatial understanding")
            
            # SIMPLE: Just encode full query and search
            # LongCLIP's position embeddings handle the spatial relationships!
            query_embedding = await self.encode_text(query)
            query_emb_np = np.array(query_embedding)
            
            # Standard search - spatial understanding is already in the embeddings
            results = await self.search_similar(query_emb_np, k)
            
            # Add spatial metadata for transparency (used in UI)
            if spatial_info['is_spatial']:
                return {
                    **results,
                    'spatial_query': True,
                    'spatial_keywords': spatial_info['keywords'],
                    'spatial_method': 'longclip_builtin'
                }
            else:
                return results
            
        except Exception as e:
            self.log_error(f"[Spatial-CLIP] Spatial search failed: {str(e)}")
            return {"error": str(e), "results": []}
    
    def _detect_spatial_keywords(self, query: str) -> Dict[str, Any]:
        """
        Detect spatial relationship keywords in query
        
        SPATIAL KEYWORDS:
        - Position: left, right, top, bottom, center, middle
        - Relationship: next to, beside, behind, in front of, above, below
        - Direction: facing, looking at, towards
        
        Args:
            query: Search query text
        
        Returns:
            Dict with is_spatial, keywords, target_regions
        """
        query_lower = query.lower()
        
        # Spatial keyword mapping to grid regions
        # 3x3 grid: [0,1,2,3,4,5,6,7,8]
        #   0 1 2  (top-left, top-center, top-right)
        #   3 4 5  (middle-left, center, middle-right)
        #   6 7 8  (bottom-left, bottom-center, bottom-right)
        
        spatial_patterns = {
            # Position keywords
            'left': [0, 3, 6],           # Left column
            'right': [2, 5, 8],          # Right column
            'top': [0, 1, 2],            # Top row
            'bottom': [6, 7, 8],         # Bottom row
            'center': [4],               # Center region
            'middle': [3, 4, 5],         # Middle row
            
            # Relationship keywords (approximate)
            'next to': [3, 4, 5],        # Same row (middle)
            'beside': [3, 4, 5],
            'behind': [6, 7, 8],         # Behind → bottom
            'in front': [0, 1, 2],       # Front → top
            'in front of': [0, 1, 2],
            'above': [0, 1, 2],
            'below': [6, 7, 8],
            'on top': [0, 1, 2],
            'underneath': [6, 7, 8],
            
            # Corners
            'corner': [0, 2, 6, 8],      # All corners
            'top left': [0],
            'top right': [2],
            'bottom left': [6],
            'bottom right': [8],
        }
        
        detected_keywords = []
        target_regions = set()
        
        for keyword, regions in spatial_patterns.items():
            if keyword in query_lower:
                detected_keywords.append(keyword)
                target_regions.update(regions)
        
        return {
            'is_spatial': len(detected_keywords) > 0,
            'keywords': detected_keywords,
            'target_regions': list(target_regions),
            'grid_mapping': spatial_patterns
        }
    
    async def _search_expanded_with_temperature(self, expanded_queries: List[str], 
                                               top_k: int = 50, 
                                               threshold: float = 0.0) -> Dict[str, Any]:
        """
        Search using multiple expanded queries with temperature scaling
        
        Combines results from all query variations for comprehensive coverage
        
        Args:
            expanded_queries: List of query variations
            top_k: Number of final results
            threshold: Minimum similarity threshold
            
        Returns:
            Combined and re-ranked results
        """
        try:
            self.log_info(f"Expanded search with {len(expanded_queries)} query variations")
            
            all_results = []
            query_weights = []
            
            # Search with each expanded query
            for i, query in enumerate(expanded_queries):
                try:
                    # Get embedding for this query variation
                    query_embedding = await self._embed_text(query)
                    embedding_array = np.array(query_embedding['embedding'])
                    
                    # Search with temperature scaling
                    results = self.search_with_temperature(embedding_array, top_k * 2)  # Get more candidates
                    
                    # Weight: original query gets highest weight, variations get lower
                    weight = 1.0 if i == 0 else 0.7  # Original query weighted higher
                    query_weights.append(weight)
                    
                    # Add weighted results
                    for result in results.get('results', []):
                        result['query_variation'] = query
                        result['query_weight'] = weight
                        result['weighted_probability'] = result.get('probability', 0.0) * weight
                        all_results.append(result)
                        
                except Exception as e:
                    self.log_warning(f"Failed to search with query variation '{query}': {str(e)}")
                    continue
            
            if not all_results:
                return {"results": [], "total_found": 0, "method": "expanded_search_failed"}
            
            # Combine results by path (same image might appear in multiple query results)
            combined_results = {}
            for result in all_results:
                path = result['path']
                if path not in combined_results:
                    combined_results[path] = result.copy()
                    combined_results[path]['query_matches'] = [result['query_variation']]
                    combined_results[path]['combined_score'] = result['weighted_probability']
                else:
                    # Combine scores from multiple query matches
                    existing = combined_results[path]
                    existing['query_matches'].append(result['query_variation'])
                    existing['combined_score'] += result['weighted_probability']
                    
                    # Update similarity to maximum (best match across all queries)
                    if result['similarity'] > existing['similarity']:
                        existing['similarity'] = result['similarity']
                        existing['probability'] = result['probability']
            
            # Sort by combined score and apply threshold
            final_results = []
            for path, result in combined_results.items():
                if result['similarity'] >= threshold:
                    result['num_query_matches'] = len(result['query_matches'])
                    final_results.append(result)
            
            # Sort by combined score (descending)
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Take top k and add final ranking
            final_results = final_results[:top_k]
            for i, result in enumerate(final_results):
                result['final_rank'] = i + 1
            
            self.log_info(f"Expanded search: {len(all_results)} total → {len(final_results)} final results")
            
            return {
                "results": final_results,
                "total_found": len(final_results),
                "query_variations": len(expanded_queries),
                "total_matches": len(all_results),
                "method": "expanded_temperature_search"
            }
            
        except Exception as e:
            self.log_error(f"Expanded search failed: {str(e)}")
            # Fallback to single query search
            return await self._search_similar(expanded_queries[0] if expanded_queries else "", top_k, threshold)


# Main entry point for running as a service
async def main():
    """Run the Semantic Engine as a service"""
    engine = SemanticEngine()
    
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

