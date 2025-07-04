"""
FAISS Index Builder for Instrument & Patch Identifier

This module combines text and audio embeddings into a searchable FAISS index
for fast nearest-neighbor retrieval during inference.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pickle

from text_db import TextDatabase
from audio_db import AudioDatabase


class IndexBuilder:
    """Builds and manages FAISS index for instrument identification."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.text_db = TextDatabase(data_dir)
        self.audio_db = AudioDatabase(data_dir)
        
    def load_embeddings(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load both text and audio embeddings.
        
        Returns:
            Tuple of (combined_embeddings, labels, types)
            where types indicates 'text' or 'audio' for each embedding
        """
        print("Loading embeddings...")
        
        # Load text embeddings
        try:
            text_embs, text_labels = self.text_db.load_embeddings()
            print(f"Loaded {len(text_labels)} text embeddings")
        except FileNotFoundError:
            print("Text embeddings not found. Creating them...")
            text_embs, text_labels = self.text_db.create_text_database()
        
        # Load audio embeddings (optional)
        audio_embs, audio_paths = self.audio_db.load_audio_embeddings()
        if len(audio_embs) > 0:
            print(f"Loaded {len(audio_paths)} audio embeddings")
        else:
            print("No audio embeddings found (this is optional)")
        
        # Combine embeddings
        all_embeddings = []
        all_labels = []
        all_types = []
        
        # Add text embeddings
        if len(text_embs) > 0:
            all_embeddings.append(text_embs)
            all_labels.extend(text_labels)
            all_types.extend(['text'] * len(text_labels))
        
        # Add audio embeddings
        if len(audio_embs) > 0:
            all_embeddings.append(audio_embs)
            # Extract filename from path for labels
            audio_labels = [Path(path).stem for path in audio_paths]
            all_labels.extend(audio_labels)
            all_types.extend(['audio'] * len(audio_labels))
        
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            print(f"Combined embeddings shape: {combined_embeddings.shape}")
            return combined_embeddings, all_labels, all_types
        else:
            raise ValueError("No embeddings found. Please create text or audio embeddings first.")
    
    def build_faiss_index(self, embeddings: np.ndarray, 
                         index_type: str = "flat") -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Embedding vectors
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        n_embeddings = embeddings.shape[0]
        
        print(f"Building FAISS index with {n_embeddings} embeddings, dimension {dimension}")
        
        if index_type == "flat":
            # Simple flat index (exact search)
            index = faiss.IndexFlatL2(dimension)
            
        elif index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, n_embeddings // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            print("Training IVF index...")
            index.train(embeddings.astype(np.float32))
            
        elif index_type == "hnsw":
            # HNSW index for very fast approximate search
            M = 16  # Number of connections
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        print("Adding embeddings to index...")
        index.add(embeddings.astype(np.float32))
        
        print(f"Index built successfully with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, labels: List[str], 
                   types: List[str], metadata: Dict[str, Any] = None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index
            labels: Labels corresponding to each embedding
            types: Types ('text' or 'audio') for each embedding
            metadata: Additional metadata to save
        """
        # Save FAISS index
        index_path = self.data_dir / "faiss.index"
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_dict = {
            "labels": labels,
            "types": types,
            "dimension": index.d,
            "total_vectors": index.ntotal,
            "index_type": type(index).__name__,
            "metadata": metadata or {}
        }
        
        metadata_path = self.data_dir / "index_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        print(f"Saved index metadata to {metadata_path}")
    
    def load_index(self) -> Tuple[faiss.Index, List[str], List[str], Dict[str, Any]]:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            Tuple of (index, labels, types, metadata)
        """
        index_path = self.data_dir / "faiss.index"
        metadata_path = self.data_dir / "index_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Index not found. Run build_index() first.")
        
        # Load index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        labels = metadata_dict["labels"]
        types = metadata_dict["types"]
        metadata = metadata_dict.get("metadata", {})
        
        print(f"Loaded index with {index.ntotal} vectors, dimension {index.d}")
        return index, labels, types, metadata
    
    def search_index(self, index: faiss.Index, query_embedding: np.ndarray,
                    labels: List[str], types: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for nearest neighbors.
        
        Args:
            index: FAISS index
            query_embedding: Query embedding vector
            labels: Labels for each index entry
            types: Types for each index entry
            k: Number of nearest neighbors to return
            
        Returns:
            List of search results with distances and labels
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = index.search(query_embedding.astype(np.float32), k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # Valid index
                results.append({
                    "rank": i + 1,
                    "label": labels[idx],
                    "type": types[idx],
                    "distance": float(dist),
                    "similarity": 1.0 / (1.0 + dist)  # Convert distance to similarity
                })
        
        return results
    
    def build_complete_index(self, index_type: str = "flat") -> Tuple[faiss.Index, List[str], List[str]]:
        """
        Build complete index from all available embeddings.
        
        Args:
            index_type: Type of FAISS index to build
            
        Returns:
            Tuple of (index, labels, types)
        """
        print("Building complete FAISS index...")
        
        # Load all embeddings
        embeddings, labels, types = self.load_embeddings()
        
        # Build index
        index = self.build_faiss_index(embeddings, index_type)
        
        # Save index
        metadata = {
            "created_at": str(Path(__file__).stat().st_mtime),
            "index_type": index_type,
            "text_embeddings": sum(1 for t in types if t == 'text'),
            "audio_embeddings": sum(1 for t in types if t == 'audio')
        }
        self.save_index(index, labels, types, metadata)
        
        print("Index building complete!")
        return index, labels, types


def main():
    """Main function to build the FAISS index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index for Instrument & Patch Identifier")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="flat",
                       help="Type of FAISS index to build")
    parser.add_argument("--test-search", action="store_true",
                       help="Test the index with a sample search")
    
    args = parser.parse_args()
    
    print("Building FAISS index for Instrument & Patch Identifier...")
    
    builder = IndexBuilder()
    
    try:
        index, labels, types = builder.build_complete_index(args.index_type)
        
        if args.test_search:
            print("\nTesting index with sample search...")
            # Create a random query vector
            dimension = index.d
            query = np.random.randn(1, dimension).astype(np.float32)
            
            results = builder.search_index(index, query, labels, types, k=5)
            
            print("Sample search results:")
            for result in results:
                print(f"  {result['rank']}. {result['label']} ({result['type']}) - "
                      f"similarity: {result['similarity']:.3f}")
        
        print(f"\nIndex built successfully!")
        print(f"- Total vectors: {index.ntotal}")
        print(f"- Dimension: {index.d}")
        print(f"- Text labels: {sum(1 for t in types if t == 'text')}")
        print(f"- Audio samples: {sum(1 for t in types if t == 'audio')}")
        print(f"- Index saved to 'data/faiss.index'")
        
    except Exception as e:
        print(f"Error building index: {e}")
        print("Make sure you have created text embeddings first:")
        print("  python src/text_db.py")


if __name__ == "__main__":
    main()
