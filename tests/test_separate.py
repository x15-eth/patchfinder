"""
Tests for separate.py module
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from separate import AudioSeparator


class TestAudioSeparator:
    """Test cases for AudioSeparator class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.separator = AudioSeparator(str(self.output_dir))
        
        # Create test audio file
        self.test_audio_file = self.create_test_audio()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_audio(self):
        """Create a test audio file."""
        # Generate 2 seconds of test audio (stereo)
        duration = 2.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Create stereo audio with different content in each channel
        left_channel = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))  # 440 Hz
        right_channel = np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples))  # 880 Hz
        
        stereo_audio = np.column_stack([left_channel, right_channel])
        
        # Save to file
        test_file = Path(self.temp_dir) / "test_audio.wav"
        sf.write(test_file, stereo_audio, sample_rate)
        
        return str(test_file)
    
    def test_init(self):
        """Test AudioSeparator initialization."""
        assert self.separator.output_dir.exists()
        assert self.separator.temp_dir.exists()
        assert "4stems" in self.separator.models
        assert len(self.separator.stem_names["4stems"]) == 4
    
    def test_models_configuration(self):
        """Test model configurations."""
        assert "2stems" in self.separator.models
        assert "4stems" in self.separator.models
        assert "5stems" in self.separator.models
        
        assert self.separator.stem_names["2stems"] == ["vocals", "accompaniment"]
        assert self.separator.stem_names["4stems"] == ["vocals", "drums", "bass", "other"]
        assert self.separator.stem_names["5stems"] == ["vocals", "drums", "bass", "piano", "other"]
    
    def test_check_spleeter_installation(self):
        """Test Spleeter installation check."""
        # This will likely fail in CI, but that's expected
        result = self.separator.check_spleeter_installation()
        assert isinstance(result, bool)
    
    @pytest.mark.skipif(True, reason="Requires Spleeter installation - slow test")
    def test_separate_audio(self):
        """Test audio separation (requires Spleeter)."""
        if not self.separator.check_spleeter_installation():
            pytest.skip("Spleeter not installed")
        
        stem_files = self.separator.separate_audio(self.test_audio_file, "4stems")
        
        # Check that we got the expected stems
        expected_stems = ["vocals", "drums", "bass", "other"]
        for stem in expected_stems:
            assert stem in stem_files
            assert Path(stem_files[stem]).exists()
    
    def test_separate_audio_file_not_found(self):
        """Test separation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.separator.separate_audio("nonexistent.wav")
    
    def test_separate_audio_invalid_model(self):
        """Test separation with invalid model."""
        with pytest.raises(ValueError):
            self.separator.separate_audio(self.test_audio_file, "invalid_model")
    
    def test_convert_to_mono(self):
        """Test converting stems to mono."""
        # Create mock stereo files
        stem_files = {}
        for i, stem in enumerate(["vocals", "drums"]):
            # Create stereo audio
            stereo_audio = np.random.randn(16000, 2).astype(np.float32)
            stem_file = self.output_dir / f"{stem}.wav"
            stem_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(stem_file, stereo_audio, 16000)
            stem_files[stem] = str(stem_file)
        
        # Convert to mono
        mono_files = self.separator.convert_to_mono(stem_files)
        
        # Check results
        assert len(mono_files) == len(stem_files)
        for stem in stem_files:
            assert stem in mono_files
            mono_file = Path(mono_files[stem])
            assert mono_file.exists()
            
            # Check that it's actually mono
            audio, sr = sf.read(mono_file)
            assert audio.ndim == 1  # Should be 1D (mono)
    
    def test_get_stem_info(self):
        """Test getting stem information."""
        # Create mock audio files
        stem_files = {}
        for stem in ["vocals", "drums"]:
            # Create test audio
            audio = np.random.randn(32000).astype(np.float32)  # 2 seconds at 16kHz
            stem_file = self.output_dir / f"{stem}.wav"
            stem_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(stem_file, audio, 16000)
            stem_files[stem] = str(stem_file)
        
        # Get info
        stem_info = self.separator.get_stem_info(stem_files)
        
        # Check results
        assert len(stem_info) == len(stem_files)
        for stem in stem_files:
            assert stem in stem_info
            info = stem_info[stem]
            
            assert "file_path" in info
            assert "duration" in info
            assert "sample_rate" in info
            assert "channels" in info
            assert "rms_energy" in info
            assert "max_amplitude" in info
            
            assert info["duration"] == pytest.approx(2.0, rel=0.1)
            assert info["sample_rate"] == 16000
            assert info["channels"] == 1
    
    def test_get_stem_info_with_error(self):
        """Test getting stem info with corrupted file."""
        # Create a non-audio file
        bad_file = self.output_dir / "bad.wav"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("not audio data")
        
        stem_files = {"bad_stem": str(bad_file)}
        stem_info = self.separator.get_stem_info(stem_files)
        
        assert "bad_stem" in stem_info
        assert "error" in stem_info["bad_stem"]
    
    def test_cleanup_temp_files(self):
        """Test cleanup of temporary files."""
        # Create some temp files
        temp_file = self.separator.temp_dir / "test.txt"
        temp_file.write_text("test")
        assert temp_file.exists()
        
        # Cleanup
        self.separator.cleanup_temp_files()
        
        # Check that temp dir still exists but is empty
        assert self.separator.temp_dir.exists()
        assert not temp_file.exists()


class TestAudioSeparatorIntegration:
    """Integration tests for AudioSeparator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_workflow_without_spleeter(self):
        """Test workflow when Spleeter is not available."""
        separator = AudioSeparator(self.temp_dir)
        
        # Create test audio
        audio = np.random.randn(16000, 2).astype(np.float32)
        test_file = Path(self.temp_dir) / "test.wav"
        sf.write(test_file, audio, 16000)
        
        # Mock Spleeter check to return False
        original_check = separator.check_spleeter_installation
        separator.check_spleeter_installation = lambda: False
        
        try:
            with pytest.raises(RuntimeError, match="Spleeter not found"):
                separator.separate_audio(str(test_file))
        finally:
            separator.check_spleeter_installation = original_check
    
    def test_stem_names_consistency(self):
        """Test that stem names are consistent across models."""
        separator = AudioSeparator(self.temp_dir)
        
        # All models should have vocals
        for model in separator.stem_names:
            assert "vocals" in separator.stem_names[model]
        
        # 4stems and 5stems should have drums, bass
        for model in ["4stems", "5stems"]:
            stems = separator.stem_names[model]
            assert "drums" in stems
            assert "bass" in stems
        
        # 5stems should have piano
        assert "piano" in separator.stem_names["5stems"]


if __name__ == "__main__":
    pytest.main([__file__])
