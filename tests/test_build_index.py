"""
Tests for build_index.py module
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import sys
import faiss

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_index import IndexBuilder


class TestIndexBuilder:
    """Test cases for IndexBuilder class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = IndexBuilder(self.temp_dir)
        
        # Create mock embeddings and labels
        self.create_mock_data()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_data(self):
        """Create mock text and audio embeddings for testing."""
        # Mock text embeddings
        text_labels = ["acoustic guitar", "electric piano", "analog synthesizer"]
        text_embs = np.random.randn(3, 512).astype(np.float32)
        
        np.save(Path(self.temp_dir) / "text_embs.npy", text_embs)
        with open(Path(self.temp_dir) / "text_labels.json", 'w') as f:
            json.dump(text_labels, f)
        
        # Mock audio embeddings (optional)
        audio_paths = ["audio1.wav", "audio2.wav"]
        audio_embs = np.random.randn(2, 512).astype(np.float32)
        
        np.save(Path(self.temp_dir) / "audio_embs.npy", audio_embs)
        with open(Path(self.temp_dir) / "audio_paths.json", 'w') as f:
            json.dump(audio_paths, f)
    
    def test_init(self):
        """Test IndexBuilder initialization."""
        assert self.builder.data_dir.exists()
        assert self.builder.text_db is not None
        assert self.builder.audio_db is not None
    
    def test_load_embeddings(self):
        """Test loading embeddings."""
        embeddings, labels, types = self.builder.load_embeddings()
        
        # Should have 5 total embeddings (3 text + 2 audio)
        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] == 512
        assert len(labels) == 5
        assert len(types) == 5
        
        # Check types
        assert types[:3] == ['text'] * 3
        assert types[3:] == ['audio'] * 2
        
        # Check labels
        assert labels[:3] == ["acoustic guitar", "electric piano", "analog synthesizer"]
        assert labels[3:] == ["audio1", "audio2"]  # Stems from paths
    
    def test_load_embeddings_text_only(self):
        """Test loading embeddings with only text data."""
        # Remove audio files
        (Path(self.temp_dir) / "audio_embs.npy").unlink()
        (Path(self.temp_dir) / "audio_paths.json").unlink()
        
        embeddings, labels, types = self.builder.load_embeddings()
        
        # Should have 3 text embeddings only
        assert embeddings.shape[0] == 3
        assert len(labels) == 3
        assert len(types) == 3
        assert all(t == 'text' for t in types)
    
    def test_build_faiss_index_flat(self):
        """Test building flat FAISS index."""
        embeddings = np.random.randn(10, 512).astype(np.float32)
        
        index = self.builder.build_faiss_index(embeddings, "flat")
        
        assert isinstance(index, faiss.IndexFlatL2)
        assert index.d == 512
        assert index.ntotal == 10
    
    def test_build_faiss_index_ivf(self):
        """Test building IVF FAISS index."""
        embeddings = np.random.randn(100, 512).astype(np.float32)
        
        index = self.builder.build_faiss_index(embeddings, "ivf")
        
        assert isinstance(index, faiss.IndexIVFFlat)
        assert index.d == 512
        assert index.ntotal == 100
    
    def test_build_faiss_index_hnsw(self):
        """Test building HNSW FAISS index."""
        embeddings = np.random.randn(50, 512).astype(np.float32)
        
        index = self.builder.build_faiss_index(embeddings, "hnsw")
        
        assert isinstance(index, faiss.IndexHNSWFlat)
        assert index.d == 512
        assert index.ntotal == 50
    
    def test_build_faiss_index_invalid(self):
        """Test building index with invalid type."""
        embeddings = np.random.randn(10, 512).astype(np.float32)
        
        with pytest.raises(ValueError):
            self.builder.build_faiss_index(embeddings, "invalid_type")
    
    def test_save_and_load_index(self):
        """Test saving and loading FAISS index."""
        # Create index
        embeddings = np.random.randn(10, 512).astype(np.float32)
        index = self.builder.build_faiss_index(embeddings, "flat")
        
        labels = [f"label_{i}" for i in range(10)]
        types = ['text'] * 5 + ['audio'] * 5
        metadata = {"test": "data"}
        
        # Save
        self.builder.save_index(index, labels, types, metadata)
        
        # Check files exist
        assert (Path(self.temp_dir) / "faiss.index").exists()
        assert (Path(self.temp_dir) / "index_metadata.json").exists()
        
        # Load
        loaded_index, loaded_labels, loaded_types, loaded_metadata = self.builder.load_index()
        
        # Check data
        assert loaded_index.d == index.d
        assert loaded_index.ntotal == index.ntotal
        assert loaded_labels == labels
        assert loaded_types == types
        assert loaded_metadata == metadata
    
    def test_load_index_not_found(self):
        """Test loading index when files don't exist."""
        with pytest.raises(FileNotFoundError):
            self.builder.load_index()
    
    def test_search_index(self):
        """Test searching the FAISS index."""
        # Create index
        embeddings = np.random.randn(10, 512).astype(np.float32)
        index = self.builder.build_faiss_index(embeddings, "flat")
        
        labels = [f"instrument_{i}" for i in range(10)]
        types = ['text'] * 10
        
        # Search with one of the original embeddings
        query = embeddings[0:1]  # First embedding as query
        results = self.builder.search_index(index, query, labels, types, k=5)
        
        # Check results
        assert len(results) == 5
        assert results[0]["rank"] == 1
        assert results[0]["label"] == "instrument_0"  # Should match itself
        assert results[0]["type"] == "text"
        assert results[0]["distance"] < 0.01  # Should be very close to itself
        assert results[0]["similarity"] > 0.99
    
    def test_search_index_1d_query(self):
        """Test searching with 1D query vector."""
        embeddings = np.random.randn(5, 512).astype(np.float32)
        index = self.builder.build_faiss_index(embeddings, "flat")
        
        labels = [f"label_{i}" for i in range(5)]
        types = ['text'] * 5
        
        # 1D query
        query = embeddings[0]  # 1D vector
        results = self.builder.search_index(index, query, labels, types, k=3)
        
        assert len(results) == 3
        assert all("rank" in r for r in results)
    
    def test_build_complete_index(self):
        """Test building complete index from all data."""
        index, labels, types = self.builder.build_complete_index("flat")
        
        # Check index
        assert isinstance(index, faiss.IndexFlatL2)
        assert index.ntotal == 5  # 3 text + 2 audio
        assert len(labels) == 5
        assert len(types) == 5
        
        # Check files were saved
        assert (Path(self.temp_dir) / "faiss.index").exists()
        assert (Path(self.temp_dir) / "index_metadata.json").exists()


class TestIndexBuilderIntegration:
    """Integration tests for IndexBuilder."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from embeddings to search."""
        builder = IndexBuilder(self.temp_dir)
        
        # Create mock text embeddings
        text_labels = ["guitar", "piano", "drums"]
        text_embs = np.random.randn(3, 128).astype(np.float32)
        
        np.save(Path(self.temp_dir) / "text_embs.npy", text_embs)
        with open(Path(self.temp_dir) / "text_labels.json", 'w') as f:
            json.dump(text_labels, f)
        
        # Build index
        index, labels, types = builder.build_complete_index("flat")
        
        # Test search
        query = text_embs[0:1]  # Use first embedding as query
        results = builder.search_index(index, query, labels, types, k=2)
        
        # Should find the guitar (first label) as top match
        assert results[0]["label"] == "guitar"
        assert results[0]["distance"] < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
