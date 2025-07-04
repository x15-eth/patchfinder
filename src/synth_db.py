"""
Synthesizer-Specific Database Manager for Instrument & Patch Identifier

This module manages databases of synthesizer-specific patches rendered from SoundFonts,
allowing for targeted instrument identification by specific keyboard models.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from tqdm import tqdm

# Add CLAP to path if it exists
clap_path = Path(__file__).parent.parent / "CLAP"
if clap_path.exists():
    sys.path.insert(0, str(clap_path))

try:
    from src.laion_clap import CLAP_Module
except ImportError:
    print("CLAP not found. Please run setup.sh/setup.bat first to install CLAP.")
    CLAP_Module = None

from audio_db import AudioDatabase


class SynthesizerDatabase:
    """Manages synthesizer-specific patch databases."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.soundfonts_dir = self.data_dir / "soundfonts"
        self.soundfonts_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.sample_rate = 48000  # CLAP expects 48kHz
        
    def load_model(self):
        """Load the CLAP model for audio embeddings."""
        if self.model is None and CLAP_Module is not None:
            print("Loading CLAP model...")
            self.model = CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()
            print("CLAP model loaded successfully.")
    
    def list_synthesizers(self) -> List[str]:
        """
        List all available synthesizer databases.
        
        Returns:
            List of synthesizer names
        """
        synthesizers = []
        for synth_dir in self.soundfonts_dir.iterdir():
            if synth_dir.is_dir() and (synth_dir / "metadata.json").exists():
                synthesizers.append(synth_dir.name)
        return synthesizers
    
    def get_synthesizer_info(self, synthesizer_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific synthesizer database.
        
        Args:
            synthesizer_name: Name of the synthesizer
            
        Returns:
            Synthesizer metadata or None if not found
        """
        synth_dir = self.soundfonts_dir / synthesizer_name
        metadata_file = synth_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Error loading synthesizer metadata: {e}")
            return None
    
    def create_synthesizer_embeddings(self, synthesizer_name: str) -> Optional[Dict[str, Any]]:
        """
        Create CLAP embeddings for all patches in a synthesizer database.
        
        Args:
            synthesizer_name: Name of the synthesizer
            
        Returns:
            Results dictionary or None if failed
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            print("CLAP model not available")
            return None
        
        # Get synthesizer info
        synth_info = self.get_synthesizer_info(synthesizer_name)
        if synth_info is None:
            print(f"Synthesizer '{synthesizer_name}' not found")
            return None
        
        patches = synth_info.get("patches", [])
        if not patches:
            print(f"No patches found for synthesizer '{synthesizer_name}'")
            return None
        
        print(f"Creating embeddings for {len(patches)} patches from {synthesizer_name}")
        
        # Create audio database instance for loading audio
        audio_db = AudioDatabase()
        
        embeddings_list = []
        valid_patches = []
        
        for patch in tqdm(patches, desc="Creating embeddings"):
            file_path = patch["file_path"]
            
            if not Path(file_path).exists():
                print(f"Warning: Audio file not found: {file_path}")
                continue
            
            try:
                # Load audio
                audio = audio_db.load_audio(file_path)
                if audio is None:
                    continue
                
                # Ensure minimum length (1 second)
                min_samples = int(1.0 * self.sample_rate)
                if len(audio) < min_samples:
                    audio = np.pad(audio, (0, min_samples - len(audio)))
                
                # Get embedding
                embedding = self.model.get_audio_embedding([audio])
                embeddings_list.append(embedding[0])
                valid_patches.append(patch)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not embeddings_list:
            print("No valid embeddings created")
            return None
        
        # Stack embeddings
        embeddings = np.vstack(embeddings_list)
        
        # Save embeddings and metadata
        synth_dir = self.soundfonts_dir / synthesizer_name
        
        # Save embeddings
        embeddings_file = synth_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Save patch metadata
        patches_file = synth_dir / "patches.json"
        with open(patches_file, 'w') as f:
            json.dump(valid_patches, f, indent=2)
        
        # Update metadata
        embedding_metadata = {
            "synthesizer_name": synthesizer_name,
            "total_patches": len(valid_patches),
            "embedding_dimension": embeddings.shape[1],
            "embeddings_file": str(embeddings_file),
            "patches_file": str(patches_file),
            "sample_rate": self.sample_rate
        }
        
        embedding_metadata_file = synth_dir / "embedding_metadata.json"
        with open(embedding_metadata_file, 'w') as f:
            json.dump(embedding_metadata, f, indent=2)
        
        print(f"Created embeddings for {len(valid_patches)} patches")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        return embedding_metadata
    
    def build_synthesizer_index(self, synthesizer_name: str, 
                               index_type: str = "flat") -> Optional[Dict[str, Any]]:
        """
        Build FAISS index for a specific synthesizer.
        
        Args:
            synthesizer_name: Name of the synthesizer
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            
        Returns:
            Index metadata or None if failed
        """
        synth_dir = self.soundfonts_dir / synthesizer_name
        embeddings_file = synth_dir / "embeddings.npy"
        patches_file = synth_dir / "patches.json"
        
        if not embeddings_file.exists() or not patches_file.exists():
            print(f"Embeddings not found for {synthesizer_name}. Run create_synthesizer_embeddings first.")
            return None
        
        # Load embeddings and patches
        embeddings = np.load(embeddings_file)
        with open(patches_file, 'r') as f:
            patches = json.load(f)
        
        print(f"Building {index_type} index for {synthesizer_name}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        n_embeddings = embeddings.shape[0]
        
        if index_type == "flat":
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            nlist = min(100, n_embeddings // 10)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings.astype(np.float32))
        elif index_type == "hnsw":
            M = 16
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        # Save index
        index_file = synth_dir / "faiss.index"
        faiss.write_index(index, str(index_file))
        
        # Save index metadata
        index_metadata = {
            "synthesizer_name": synthesizer_name,
            "index_type": index_type,
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "index_file": str(index_file),
            "patches_file": str(patches_file)
        }
        
        index_metadata_file = synth_dir / "index_metadata.json"
        with open(index_metadata_file, 'w') as f:
            json.dump(index_metadata, f, indent=2)
        
        print(f"Index built successfully with {index.ntotal} vectors")
        return index_metadata
    
    def load_synthesizer_index(self, synthesizer_name: str) -> Optional[Tuple[faiss.Index, List[Dict], Dict]]:
        """
        Load FAISS index for a specific synthesizer.
        
        Args:
            synthesizer_name: Name of the synthesizer
            
        Returns:
            Tuple of (index, patches, metadata) or None if failed
        """
        synth_dir = self.soundfonts_dir / synthesizer_name
        index_file = synth_dir / "faiss.index"
        patches_file = synth_dir / "patches.json"
        metadata_file = synth_dir / "index_metadata.json"
        
        if not all(f.exists() for f in [index_file, patches_file, metadata_file]):
            return None
        
        try:
            # Load index
            index = faiss.read_index(str(index_file))
            
            # Load patches
            with open(patches_file, 'r') as f:
                patches = json.load(f)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return index, patches, metadata
            
        except Exception as e:
            print(f"Error loading synthesizer index: {e}")
            return None
    
    def search_synthesizer(self, synthesizer_name: str, query_embedding: np.ndarray,
                          k: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Search for similar patches in a specific synthesizer database.
        
        Args:
            synthesizer_name: Name of the synthesizer
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of search results or None if failed
        """
        # Load synthesizer index
        result = self.load_synthesizer_index(synthesizer_name)
        if result is None:
            print(f"Failed to load index for synthesizer '{synthesizer_name}'")
            return None
        
        index, patches, metadata = result
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = index.search(query_embedding.astype(np.float32), k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(patches):
                patch = patches[idx]
                results.append({
                    "rank": i + 1,
                    "synthesizer": synthesizer_name,
                    "program": patch["program"],
                    "bank": patch["bank"],
                    "preset": patch["preset"],
                    "name": patch["name"],
                    "file_path": patch["file_path"],
                    "distance": float(dist),
                    "similarity": 1.0 / (1.0 + dist)
                })
        
        return results


def main():
    """Main function for synthesizer database management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage synthesizer-specific databases")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available synthesizers')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get synthesizer info')
    info_parser.add_argument('synthesizer', help='Synthesizer name')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Create embeddings')
    embed_parser.add_argument('synthesizer', help='Synthesizer name')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build FAISS index')
    index_parser.add_argument('synthesizer', help='Synthesizer name')
    index_parser.add_argument('--type', choices=['flat', 'ivf', 'hnsw'], 
                             default='flat', help='Index type')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    db = SynthesizerDatabase()
    
    if args.command == 'list':
        synthesizers = db.list_synthesizers()
        if synthesizers:
            print("Available synthesizers:")
            for synth in synthesizers:
                info = db.get_synthesizer_info(synth)
                if info:
                    print(f"  {synth}: {info.get('rendered_patches', 0)} patches")
        else:
            print("No synthesizers found")
    
    elif args.command == 'info':
        info = db.get_synthesizer_info(args.synthesizer)
        if info:
            print(f"Synthesizer: {info['synthesizer_name']}")
            print(f"SoundFont: {info['soundfont_path']}")
            print(f"Patches: {info['rendered_patches']}")
            print(f"Sample rate: {info['sample_rate']} Hz")
            print(f"Duration: {info['duration']} seconds")
        else:
            print(f"Synthesizer '{args.synthesizer}' not found")
    
    elif args.command == 'embed':
        result = db.create_synthesizer_embeddings(args.synthesizer)
        if result:
            print(f"Embeddings created for {result['total_patches']} patches")
        else:
            print("Failed to create embeddings")
    
    elif args.command == 'index':
        result = db.build_synthesizer_index(args.synthesizer, args.type)
        if result:
            print(f"Index built with {result['total_vectors']} vectors")
        else:
            print("Failed to build index")
    
    return 0


if __name__ == "__main__":
    exit(main())
