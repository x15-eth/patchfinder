"""
Tests for inference.py module
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import sys
import soundfile as sf
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import InstrumentIdentifier


class TestInstrumentIdentifier:
    """Test cases for InstrumentIdentifier class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.identifier = InstrumentIdentifier(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        
        # Create test audio file
        self.test_audio_file = self.create_test_audio()
        
        # Create mock index data
        self.create_mock_index_data()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_audio(self):
        """Create a test audio file."""
        duration = 3.0
        sample_rate = 48000
        samples = int(duration * sample_rate)
        
        # Create mono audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        test_file = Path(self.temp_dir) / "test_audio.wav"
        sf.write(test_file, audio, sample_rate)
        
        return str(test_file)
    
    def create_mock_index_data(self):
        """Create mock FAISS index and metadata."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock index metadata
        metadata = {
            "labels": ["acoustic guitar", "electric piano", "analog synthesizer"],
            "types": ["text", "text", "text"],
            "dimension": 512,
            "total_vectors": 3,
            "index_type": "IndexFlatL2",
            "metadata": {}
        }
        
        with open(self.data_dir / "index_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create a dummy index file (we'll mock the actual loading)
        (self.data_dir / "faiss.index").touch()
    
    def test_init(self):
        """Test InstrumentIdentifier initialization."""
        assert self.identifier.data_dir.exists()
        assert self.identifier.output_dir.exists()
        assert self.identifier.model is None
        assert self.identifier.index is None
        assert self.identifier.sample_rate == 48000
        assert self.identifier.chunk_duration == 5.0
    
    def test_chunk_audio(self):
        """Test audio chunking."""
        # Create test audio (10 seconds)
        duration = 10.0
        sample_rate = 48000
        audio = np.random.randn(int(duration * sample_rate))
        
        chunks = self.identifier.chunk_audio(audio, sample_rate)
        
        # Should have overlapping chunks
        expected_chunk_samples = int(5.0 * sample_rate)  # 5 second chunks
        
        assert len(chunks) > 1  # Should have multiple chunks
        assert all(len(chunk) == expected_chunk_samples for chunk in chunks)
    
    def test_chunk_audio_short(self):
        """Test chunking short audio."""
        # Create 2 second audio (shorter than chunk duration)
        duration = 2.0
        sample_rate = 48000
        audio = np.random.randn(int(duration * sample_rate))
        
        chunks = self.identifier.chunk_audio(audio, sample_rate)
        
        # Should have one chunk, padded to full length
        assert len(chunks) == 1
        assert len(chunks[0]) == int(5.0 * sample_rate)
    
    @patch('inference.InstrumentIdentifier.load_model')
    @patch('inference.InstrumentIdentifier.load_index')
    def test_analyze_audio_chunk(self, mock_load_index, mock_load_model):
        """Test analyzing a single audio chunk."""
        # Mock the model and index
        mock_model = Mock()
        mock_embedding = np.random.randn(1, 512).astype(np.float32)
        mock_model.get_audio_embedding.return_value = mock_embedding
        self.identifier.model = mock_model
        
        # Mock index search
        mock_index = Mock()
        mock_labels = ["guitar", "piano", "drums"]
        mock_types = ["text", "text", "text"]
        self.identifier.index = mock_index
        self.identifier.labels = mock_labels
        self.identifier.types = mock_types
        
        # Mock search results
        mock_search_results = [
            {"rank": 1, "label": "guitar", "type": "text", "distance": 0.1, "similarity": 0.9},
            {"rank": 2, "label": "piano", "type": "text", "distance": 0.3, "similarity": 0.7}
        ]
        
        with patch.object(self.identifier.index_builder, 'search_index', return_value=mock_search_results):
            # Test chunk analysis
            audio_chunk = np.random.randn(48000 * 5)  # 5 seconds
            results = self.identifier.analyze_audio_chunk(audio_chunk, k=5)
            
            assert len(results) == 2
            assert results[0]["label"] == "guitar"
            assert results[0]["similarity"] == 0.9
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        # Create mock results from multiple chunks
        results = [
            {"label": "guitar", "similarity": 0.9},
            {"label": "guitar", "similarity": 0.8},
            {"label": "piano", "similarity": 0.7},
            {"label": "guitar", "similarity": 0.85},
            {"label": "drums", "similarity": 0.6}
        ]
        
        aggregated = self.identifier.aggregate_results(results, top_n=3)
        
        # Check structure
        assert "top_matches" in aggregated
        assert "confidence_score" in aggregated
        assert "match_distribution" in aggregated
        
        # Guitar should be top (appears 3 times with high scores)
        top_matches = aggregated["top_matches"]
        assert len(top_matches) == 3
        assert top_matches[0]["label"] == "guitar"
        assert top_matches[0]["occurrences"] == 3
        
        # Confidence should be the top match's average score
        expected_guitar_avg = (0.9 + 0.8 + 0.85) / 3
        assert top_matches[0]["confidence"] == pytest.approx(expected_guitar_avg)
    
    @patch('inference.InstrumentIdentifier.load_model')
    @patch('inference.InstrumentIdentifier.load_index')
    def test_analyze_stem(self, mock_load_index, mock_load_model):
        """Test analyzing a single stem."""
        # Mock dependencies
        self.identifier.model = Mock()
        self.identifier.index = Mock()
        self.identifier.labels = ["guitar"]
        self.identifier.types = ["text"]
        
        # Mock chunk analysis
        mock_chunk_results = [
            {"label": "guitar", "similarity": 0.9},
            {"label": "guitar", "similarity": 0.8}
        ]
        
        with patch.object(self.identifier, 'analyze_audio_chunk', return_value=mock_chunk_results):
            result = self.identifier.analyze_stem(self.test_audio_file, "test_stem")
            
            assert "stem" in result
            assert "file" in result
            assert "duration" in result
            assert "top_matches" in result
            assert "confidence_score" in result
            
            assert result["stem"] == "test_stem"
            assert result["duration"] > 0
    
    def test_analyze_stem_file_not_found(self):
        """Test analyzing non-existent file."""
        result = self.identifier.analyze_stem("nonexistent.wav", "test")
        assert "error" in result
    
    def test_create_summary(self):
        """Test creating summary from stem results."""
        stem_results = {
            "vocals": {
                "confidence_score": 0.8,
                "top_matches": [{"label": "male vocals", "confidence": 0.8}],
                "duration": 30.0
            },
            "drums": {
                "confidence_score": 0.9,
                "top_matches": [{"label": "acoustic drums", "confidence": 0.9}],
                "duration": 30.0
            },
            "bass": {
                "error": "Failed to analyze"
            }
        }
        
        summary = self.identifier.create_summary(stem_results)
        
        assert summary["total_stems"] == 3
        assert summary["successful_analyses"] == 2
        assert summary["failed_analyses"] == 1
        assert summary["overall_confidence"] == pytest.approx(0.85)  # (0.8 + 0.9) / 2
        
        # Check top instruments
        top_instruments = summary["top_instruments"]
        assert len(top_instruments) == 2
        assert top_instruments[0]["instrument"] == "acoustic drums"  # Higher confidence
        assert top_instruments[1]["instrument"] == "male vocals"
    
    @patch('inference.InstrumentIdentifier.analyze_stem')
    def test_identify_instruments_no_separation(self, mock_analyze_stem):
        """Test instrument identification without separation."""
        mock_analyze_stem.return_value = {
            "stem": "full_mix",
            "confidence_score": 0.8,
            "top_matches": [{"label": "guitar", "confidence": 0.8}]
        }
        
        results = self.identifier.identify_instruments(
            self.test_audio_file,
            use_separation=False
        )
        
        assert results["use_separation"] is False
        assert "full_mix" in results["stems"]
        assert "summary" in results
        
        mock_analyze_stem.assert_called_once()
    
    @patch('inference.InstrumentIdentifier.analyze_stem')
    @patch('separate.AudioSeparator.separate_audio')
    def test_identify_instruments_with_separation(self, mock_separate, mock_analyze_stem):
        """Test instrument identification with separation."""
        # Mock separation results
        mock_separate.return_value = {
            "vocals": "/path/to/vocals.wav",
            "drums": "/path/to/drums.wav"
        }
        
        # Mock stem analysis
        mock_analyze_stem.return_value = {
            "stem": "test",
            "confidence_score": 0.8,
            "top_matches": [{"label": "test", "confidence": 0.8}]
        }
        
        results = self.identifier.identify_instruments(
            self.test_audio_file,
            use_separation=True,
            separation_model="4stems"
        )
        
        assert results["use_separation"] is True
        assert results["separation_model"] == "4stems"
        assert len(results["stems"]) == 2  # vocals and drums
        
        # Should have called analyze_stem for each separated stem
        assert mock_analyze_stem.call_count == 2
    
    def test_save_results(self):
        """Test saving results to JSON."""
        results = {"test": "data", "number": 123}
        output_file = Path(self.temp_dir) / "results.json"
        
        self.identifier.save_results(results, str(output_file))
        
        assert output_file.exists()
        
        with open(output_file) as f:
            loaded_results = json.load(f)
        
        assert loaded_results == results


class TestInstrumentIdentifierIntegration:
    """Integration tests for InstrumentIdentifier."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_chunking_edge_cases(self):
        """Test audio chunking with various edge cases."""
        identifier = InstrumentIdentifier(self.temp_dir)
        
        # Test very short audio
        short_audio = np.random.randn(1000)  # Very short
        chunks = identifier.chunk_audio(short_audio, 48000)
        assert len(chunks) == 1
        assert len(chunks[0]) == int(5.0 * 48000)  # Padded to full length
        
        # Test exact chunk length
        exact_audio = np.random.randn(int(5.0 * 48000))
        chunks = identifier.chunk_audio(exact_audio, 48000)
        assert len(chunks) == 1
        
        # Test slightly longer than chunk
        longer_audio = np.random.randn(int(5.1 * 48000))
        chunks = identifier.chunk_audio(longer_audio, 48000)
        assert len(chunks) == 2


if __name__ == "__main__":
    pytest.main([__file__])
