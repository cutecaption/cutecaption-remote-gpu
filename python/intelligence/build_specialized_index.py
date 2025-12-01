#!/usr/bin/env python3
"""
Build Specialized FAISS Index
Helper script to build FAISS indexes from Conceptualizer-transformed embeddings

Usage:
    python build_specialized_index.py --embeddings path/to/embeddings.npy \\
                                      --metadata path/to/metadata.json \\
                                      --output path/to/output.index
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import faiss
except ImportError:
    print("Error: FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
    sys.exit(1)


def build_index(embeddings_path, metadata_path, output_path, index_type='flat'):
    """
    Build FAISS index from embeddings
    
    Args:
        embeddings_path: Path to embeddings (.npy or .json)
        metadata_path: Path to metadata JSON
        output_path: Path for output index
        index_type: Type of index ('flat', 'ivf', 'hnsw')
    """
    print(f"Building specialized FAISS index...")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Output: {output_path}")
    print(f"  Type: {index_type}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nDataset: {metadata['dataset_name']}")
    print(f"Images: {metadata['num_images']}")
    print(f"Method: {metadata['method']}")
    
    # Load embeddings
    embeddings_path = Path(embeddings_path)
    
    if embeddings_path.suffix == '.npy':
        embeddings = np.load(str(embeddings_path))
    elif embeddings_path.suffix == '.json' or embeddings_path.with_suffix('.json').exists():
        # Fallback to JSON (less efficient)
        json_path = embeddings_path if embeddings_path.suffix == '.json' else embeddings_path.with_suffix('.json')
        with open(json_path, 'r') as f:
            embeddings = np.array(json.load(f))
    else:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    print(f"\nLoaded embeddings: {embeddings.shape}")
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype(np.float32)
    
    dimension = embeddings.shape[1]
    
    # Build index based on type
    if index_type == 'flat':
        # Flat index (exact search, best quality)
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        print("\nBuilding Flat index (exact search)...")
        
    elif index_type == 'ivf':
        # IVF index (faster, approximate)
        nlist = min(100, len(embeddings) // 10)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        print(f"\nBuilding IVF index (nlist={nlist})...")
        print("Training index...")
        index.train(embeddings)
        
    elif index_type == 'hnsw':
        # HNSW index (fast, high quality)
        M = 32  # Number of connections per layer
        index = faiss.IndexHNSWFlat(dimension, M)
        print(f"\nBuilding HNSW index (M={M})...")
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add embeddings to index
    print("Adding embeddings to index...")
    index.add(embeddings)
    
    # Save index
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_path))
    
    print(f"\n✓ Index saved to: {output_path}")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {dimension}")
    
    # Test search
    print("\nTesting index with sample query...")
    query = embeddings[0:1]  # Use first embedding as test query
    k = min(5, len(embeddings))
    
    distances, indices = index.search(query, k)
    
    print(f"  Top {k} results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"    {i+1}. Index {idx}, Distance: {dist:.4f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Build specialized FAISS index')
    parser.add_argument('--embeddings', required=True,
                       help='Path to embeddings file (.npy or .json)')
    parser.add_argument('--metadata', required=True,
                       help='Path to metadata JSON file')
    parser.add_argument('--output', required=True,
                       help='Path for output index file')
    parser.add_argument('--type', default='flat', choices=['flat', 'ivf', 'hnsw'],
                       help='Index type (default: flat)')
    
    args = parser.parse_args()
    
    try:
        build_index(
            embeddings_path=args.embeddings,
            metadata_path=args.metadata,
            output_path=args.output,
            index_type=args.type
        )
        
        print("\n✅ Index build complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

