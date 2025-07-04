"""
Tests for synth_db.py module
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch
import faiss

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synth_db import SynthesizerDatabase


class TestSynthesizerDatabase:
    """Test cases for SynthesizerDatabase class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db = SynthesizerDatabase(self.temp_dir)
        
        # Create mock synthesizer data
        self.create_mock_synthesizer()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_synthesizer(self):
        """Create mock synthesizer data for testing."""
        synth_name = "TestSynth"
        synth_dir = self.db.soundfonts_dir / synth_name
        synth_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata
        metadata = {
            "synthesizer_name": synth_name,
            "soundfont_path": "test.sf2",
            "rendered_patches": 3,
            "patches": [
                {
                    "program": 1,
                    "bank": 0,
                    "preset": 0,
                    "name": "Test Patch 1",
                    "file_path": str(synth_dir / "patch_001.wav"),
                    "duration": 5.0
                },
                {
                    "program": 2,
                    "bank": 0,
                    "preset": 1,
                    "name": "Test Patch 2",
                    "file_path": str(synth_dir / "patch_002.wav"),
                    "duration": 5.0
                },
                {
                    "program": 3,
                    "bank": 0,
                    "preset": 2,
                    "name": "Test Patch 3",
                    "file_path": str(synth_dir / "patch_003.wav"),
                    "duration": 5.0
                }
            ]
        }
        
        with open(synth_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create dummy audio files
        for patch in metadata["patches"]:
            Path(patch["file_path"]).write_bytes(b"dummy audio data")
    
    def test_init(self):
        """Test SynthesizerDatabase initialization."""
        assert self.db.data_dir.exists()
        assert self.db.soundfonts_dir.exists()
        assert self.db.model is None
        assert self.db.sample_rate == 48000
    
    def test_list_synthesizers(self):
        """Test listing available synthesizers."""
        synthesizers = self.db.list_synthesizers()
        assert "TestSynth" in synthesizers
    
    def test_list_synthesizers_empty(self):
        """Test listing synthesizers when none exist."""
        # Remove test synthesizer
        shutil.rmtree(self.db.soundfonts_dir / "TestSynth")
        
        synthesizers = self.db.list_synthesizers()
        assert len(synthesizers) == 0
    
    def test_get_synthesizer_info(self):
        """Test getting synthesizer information."""
        info = self.db.get_synthesizer_info("TestSynth")
        
        assert info is not None
        assert info["synthesizer_name"] == "TestSynth"
        assert info["rendered_patches"] == 3
        assert len(info["patches"]) == 3
    
    def test_get_synthesizer_info_not_found(self):
        """Test getting info for non-existent synthesizer."""
        info = self.db.get_synthesizer_info("NonExistent")
        assert info is None
    
    def test_get_synthesizer_info_invalid_json(self):
        """Test getting info with corrupted metadata."""
        synth_dir = self.db.soundfonts_dir / "BadSynth"
        synth_dir.mkdir(exist_ok=True)
        
        # Write invalid JSON
        with open(synth_dir / "metadata.json", 'w') as f:
            f.write("invalid json")
        
        info = self.db.get_synthesizer_info("BadSynth")
        assert info is None
    
    @pytest.mark.skipif(True, reason="Requires CLAP model - slow test")
    def test_create_synthesizer_embeddings(self):
        """Test creating embeddings (requires CLAP model)."""
        # This would require CLAP model to be available
        result = self.db.create_synthesizer_embeddings("TestSynth")
        
        if result is not None:
            assert result["synthesizer_name"] == "TestSynth"
            assert result["total_patches"] > 0
            assert result["embedding_dimension"] > 0
    
    def test_create_synthesizer_embeddings_no_model(self):
        """Test creating embeddings when CLAP model not available."""
        with patch('synth_db.CLAP_Module', None):
            db = SynthesizerDatabase(self.temp_dir)
            result = db.create_synthesizer_embeddings("TestSynth")
            assert result is None
    
    def test_create_synthesizer_embeddings_not_found(self):
        """Test creating embeddings for non-existent synthesizer."""
        result = self.db.create_synthesizer_embeddings("NonExistent")
        assert result is None
    
    @patch.object(SynthesizerDatabase, 'load_model')
    def test_create_synthesizer_embeddings_mock(self, mock_load_model):
        """Test creating embeddings with mocked CLAP model."""
        # Mock the CLAP model
        mock_model = Mock()
        mock_embedding = np.random.randn(1, 512).astype(np.float32)
        mock_model.get_audio_embedding.return_value = mock_embedding
        self.db.model = mock_model
        
        # Mock audio loading
        with patch('synth_db.AudioDatabase') as mock_audio_db_class:
            mock_audio_db = Mock()
            mock_audio_db.load_audio.return_value = np.random.randn(48000)  # 1 second
            mock_audio_db_class.return_value = mock_audio_db
            
            result = self.db.create_synthesizer_embeddings("TestSynth")
            
            assert result is not None
            assert result["synthesizer_name"] == "TestSynth"
            assert result["total_patches"] == 3
            assert result["embedding_dimension"] == 512
    
    def test_build_synthesizer_index_no_embeddings(self):
        """Test building index when embeddings don't exist."""
        result = self.db.build_synthesizer_index("TestSynth")
        assert result is None
    
    def test_build_synthesizer_index_mock(self):
        """Test building index with mock embeddings."""
        synth_dir = self.db.soundfonts_dir / "TestSynth"
        
        # Create mock embeddings
        embeddings = np.random.randn(3, 512).astype(np.float32)
        np.save(synth_dir / "embeddings.npy", embeddings)
        
        # Create mock patches
        patches = [
            {"program": 1, "name": "Patch 1"},
            {"program": 2, "name": "Patch 2"},
            {"program": 3, "name": "Patch 3"}
        ]
        with open(synth_dir / "patches.json", 'w') as f:
            json.dump(patches, f)
        
        result = self.db.build_synthesizer_index("TestSynth", "flat")
        
        assert result is not None
        assert result["synthesizer_name"] == "TestSynth"
        assert result["total_vectors"] == 3
        assert result["dimension"] == 512
        assert result["index_type"] == "flat"
        
        # Check that files were created
        assert (synth_dir / "faiss.index").exists()
        assert (synth_dir / "index_metadata.json").exists()
    
    def test_build_synthesizer_index_ivf(self):
        """Test building IVF index."""
        synth_dir = self.db.soundfonts_dir / "TestSynth"
        
        # Create larger embeddings for IVF
        embeddings = np.random.randn(100, 512).astype(np.float32)
        np.save(synth_dir / "embeddings.npy", embeddings)
        
        patches = [{"program": i, "name": f"Patch {i}"} for i in range(100)]
        with open(synth_dir / "patches.json", 'w') as f:
            json.dump(patches, f)
        
        result = self.db.build_synthesizer_index("TestSynth", "ivf")
        
        assert result is not None
        assert result["index_type"] == "ivf"
    
    def test_build_synthesizer_index_invalid_type(self):
        """Test building index with invalid type."""
        synth_dir = self.db.soundfonts_dir / "TestSynth"
        
        embeddings = np.random.randn(3, 512).astype(np.float32)
        np.save(synth_dir / "embeddings.npy", embeddings)
        
        patches = [{"program": i, "name": f"Patch {i}"} for i in range(3)]
        with open(synth_dir / "patches.json", 'w') as f:
            json.dump(patches, f)
        
        with pytest.raises(ValueError):
            self.db.build_synthesizer_index("TestSynth", "invalid_type")
    
    def test_load_synthesizer_index_not_found(self):
        """Test loading index when files don't exist."""
        result = self.db.load_synthesizer_index("NonExistent")
        assert result is None
    
    def test_load_synthesizer_index_mock(self):
        """Test loading index with mock data."""
        synth_dir = self.db.soundfonts_dir / "TestSynth"
        
        # Create mock index
        embeddings = np.random.randn(3, 512).astype(np.float32)
        index = faiss.IndexFlatL2(512)
        index.add(embeddings)
        faiss.write_index(index, str(synth_dir / "faiss.index"))
        
        # Create mock patches and metadata
        patches = [{"program": i, "name": f"Patch {i}"} for i in range(3)]
        with open(synth_dir / "patches.json", 'w') as f:
            json.dump(patches, f)
        
        metadata = {"synthesizer_name": "TestSynth", "total_vectors": 3}
        with open(synth_dir / "index_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        result = self.db.load_synthesizer_index("TestSynth")
        
        assert result is not None
        index, patches_loaded, metadata_loaded = result
        assert index.ntotal == 3
        assert len(patches_loaded) == 3
        assert metadata_loaded["synthesizer_name"] == "TestSynth"
    
    def test_search_synthesizer_no_index(self):
        """Test searching when index doesn't exist."""
        query = np.random.randn(512).astype(np.float32)
        result = self.db.search_synthesizer("NonExistent", query)
        assert result is None
    
    def test_search_synthesizer_mock(self):
        """Test searching with mock index."""
        synth_dir = self.db.soundfonts_dir / "TestSynth"
        
        # Create mock index and data
        embeddings = np.random.randn(3, 512).astype(np.float32)
        index = faiss.IndexFlatL2(512)
        index.add(embeddings)
        faiss.write_index(index, str(synth_dir / "faiss.index"))
        
        patches = [
            {"program": 1, "bank": 0, "preset": 0, "name": "Patch 1", "file_path": "patch1.wav"},
            {"program": 2, "bank": 0, "preset": 1, "name": "Patch 2", "file_path": "patch2.wav"},
            {"program": 3, "bank": 0, "preset": 2, "name": "Patch 3", "file_path": "patch3.wav"}
        ]
        with open(synth_dir / "patches.json", 'w') as f:
            json.dump(patches, f)
        
        metadata = {"synthesizer_name": "TestSynth"}
        with open(synth_dir / "index_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Search with first embedding (should match itself)
        query = embeddings[0]
        results = self.db.search_synthesizer("TestSynth", query, k=2)
        
        assert results is not None
        assert len(results) == 2
        assert results[0]["synthesizer"] == "TestSynth"
        assert results[0]["program"] == 1
        assert results[0]["rank"] == 1
        assert results[0]["distance"] < 0.01  # Should be very close to itself


class TestSynthesizerDatabaseIntegration:
    """Integration tests for SynthesizerDatabase."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from metadata to search."""
        db = SynthesizerDatabase(self.temp_dir)
        
        # Create synthesizer with mock data
        synth_name = "TestWorkflow"
        synth_dir = db.soundfonts_dir / synth_name
        synth_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata
        metadata = {
            "synthesizer_name": synth_name,
            "rendered_patches": 2,
            "patches": [
                {"program": 1, "name": "Piano", "file_path": str(synth_dir / "piano.wav")},
                {"program": 2, "name": "Guitar", "file_path": str(synth_dir / "guitar.wav")}
            ]
        }
        with open(synth_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create dummy audio files
        for patch in metadata["patches"]:
            Path(patch["file_path"]).write_bytes(b"dummy")
        
        # Test listing
        synthesizers = db.list_synthesizers()
        assert synth_name in synthesizers
        
        # Test info
        info = db.get_synthesizer_info(synth_name)
        assert info["synthesizer_name"] == synth_name
        assert len(info["patches"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])
