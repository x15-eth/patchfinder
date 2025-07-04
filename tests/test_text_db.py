"""
Tests for text_db.py module
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_db import TextDatabase


class TestTextDatabase:
    """Test cases for TextDatabase class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db = TextDatabase(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test TextDatabase initialization."""
        assert self.db.data_dir.exists()
        assert self.db.model is None
    
    def test_get_instrument_labels(self):
        """Test instrument labels generation."""
        labels = self.db.get_instrument_labels()
        
        # Check we have a good number of labels
        assert len(labels) >= 200
        
        # Check for expected categories
        assert "acoustic guitar" in labels
        assert "electric guitar" in labels
        assert "Moog synthesizer" in labels
        assert "808 bass" in labels
        assert "analog pad" in labels
        
        # Check no duplicates
        assert len(labels) == len(set(labels))
        
        # Check all are strings
        assert all(isinstance(label, str) for label in labels)
    
    @pytest.mark.skipif(True, reason="Requires CLAP model - slow test")
    def test_create_embeddings(self):
        """Test embedding creation (requires CLAP model)."""
        labels = ["acoustic guitar", "electric piano", "analog synthesizer"]
        
        self.db.load_model()
        embeddings = self.db.create_embeddings(labels)
        
        # Check shape
        assert embeddings.shape[0] == len(labels)
        assert embeddings.shape[1] > 0  # Should have some embedding dimension
        
        # Check data type
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
        
        # Check not all zeros
        assert not np.allclose(embeddings, 0)
    
    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        # Create dummy embeddings
        labels = ["test1", "test2", "test3"]
        embeddings = np.random.randn(3, 512).astype(np.float32)
        
        # Save
        self.db.save_embeddings(embeddings, labels)
        
        # Check files exist
        assert (Path(self.temp_dir) / "text_embs.npy").exists()
        assert (Path(self.temp_dir) / "text_labels.json").exists()
        
        # Load
        loaded_embeddings, loaded_labels = self.db.load_embeddings()
        
        # Check data
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
        assert labels == loaded_labels
    
    def test_load_embeddings_not_found(self):
        """Test loading embeddings when files don't exist."""
        with pytest.raises(FileNotFoundError):
            self.db.load_embeddings()
    
    def test_create_text_database_mock(self):
        """Test create_text_database with mocked model."""
        # Mock the model and embedding creation
        original_load_model = self.db.load_model
        original_create_embeddings = self.db.create_embeddings
        
        def mock_load_model():
            self.db.model = "mock_model"
        
        def mock_create_embeddings(labels):
            return np.random.randn(len(labels), 512).astype(np.float32)
        
        self.db.load_model = mock_load_model
        self.db.create_embeddings = mock_create_embeddings
        
        try:
            embeddings, labels = self.db.create_text_database()
            
            # Check results
            assert len(embeddings) == len(labels)
            assert len(labels) >= 200
            assert embeddings.shape[1] == 512
            
            # Check files were saved
            assert (Path(self.temp_dir) / "text_embs.npy").exists()
            assert (Path(self.temp_dir) / "text_labels.json").exists()
            
        finally:
            # Restore original methods
            self.db.load_model = original_load_model
            self.db.create_embeddings = original_create_embeddings


class TestTextDatabaseIntegration:
    """Integration tests for TextDatabase."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_label_categories(self):
        """Test that labels cover expected instrument categories."""
        db = TextDatabase(self.temp_dir)
        labels = db.get_instrument_labels()
        
        # Convert to lowercase for easier checking
        labels_lower = [label.lower() for label in labels]
        
        # Check for major categories
        categories = {
            "strings": ["guitar", "violin", "bass", "cello"],
            "winds": ["flute", "saxophone", "trumpet", "clarinet"],
            "percussion": ["drums", "snare", "kick", "cymbal"],
            "keyboards": ["piano", "rhodes", "organ", "harpsichord"],
            "synthesizers": ["moog", "analog", "pad", "lead"],
            "electronic": ["808", "909", "electronic", "digital"]
        }
        
        for category, keywords in categories.items():
            found_keywords = []
            for keyword in keywords:
                if any(keyword in label for label in labels_lower):
                    found_keywords.append(keyword)
            
            # Should find at least half the keywords for each category
            assert len(found_keywords) >= len(keywords) // 2, \
                f"Category '{category}' underrepresented. Found: {found_keywords}"
    
    def test_label_quality(self):
        """Test quality of instrument labels."""
        db = TextDatabase(self.temp_dir)
        labels = db.get_instrument_labels()
        
        # Check for reasonable length
        for label in labels:
            assert 3 <= len(label) <= 50, f"Label length issue: '{label}'"
            assert label.strip() == label, f"Label has extra whitespace: '{label}'"
            assert not label.isupper(), f"Label is all uppercase: '{label}'"
            assert not label.islower() or " " in label, f"Single word should not be all lowercase: '{label}'"


if __name__ == "__main__":
    pytest.main([__file__])
