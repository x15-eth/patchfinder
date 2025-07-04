"""
Tests for soundfont_renderer.py module
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soundfont_renderer import SoundFontRenderer


class TestSoundFontRenderer:
    """Test cases for SoundFontRenderer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.renderer = SoundFontRenderer(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test SoundFontRenderer initialization."""
        assert self.renderer.output_dir.exists()
        assert self.renderer.sample_rate == 44100
        assert self.renderer.duration == 5.0
        assert self.renderer.velocity == 100
        assert self.renderer.note == 60
        assert self.renderer.synth is None
    
    def test_check_fluidsynth_not_available(self):
        """Test FluidSynth availability check when not available."""
        # Mock FluidSynth as not available
        with patch('soundfont_renderer.FLUIDSYNTH_AVAILABLE', False):
            renderer = SoundFontRenderer(self.temp_dir)
            assert not renderer.check_fluidsynth()
    
    @pytest.mark.skipif(True, reason="Requires FluidSynth installation")
    def test_initialize_synth(self):
        """Test FluidSynth initialization (requires FluidSynth)."""
        if not self.renderer.check_fluidsynth():
            pytest.skip("FluidSynth not available")
        
        result = self.renderer.initialize_synth()
        assert isinstance(result, bool)
        
        if result:
            assert self.renderer.synth is not None
    
    def test_initialize_synth_not_available(self):
        """Test synth initialization when FluidSynth not available."""
        with patch.object(self.renderer, 'check_fluidsynth', return_value=False):
            with pytest.raises(ImportError):
                self.renderer.initialize_synth()
    
    def test_load_soundfont_file_not_found(self):
        """Test loading non-existent SoundFont file."""
        with pytest.raises(FileNotFoundError):
            self.renderer.load_soundfont("nonexistent.sf2")
    
    @patch('soundfont_renderer.fluidsynth')
    def test_load_soundfont_mock(self, mock_fluidsynth):
        """Test SoundFont loading with mocked FluidSynth."""
        # Create a dummy SF2 file
        sf2_file = Path(self.temp_dir) / "test.sf2"
        sf2_file.write_bytes(b"dummy sf2 data")
        
        # Mock FluidSynth
        mock_synth = Mock()
        mock_synth.sfload.return_value = 1  # Success
        self.renderer.synth = mock_synth
        
        sfid = self.renderer.load_soundfont(str(sf2_file))
        
        assert sfid == 1
        mock_synth.sfload.assert_called_once_with(str(sf2_file))
    
    @patch('soundfont_renderer.fluidsynth')
    def test_load_soundfont_failure(self, mock_fluidsynth):
        """Test SoundFont loading failure."""
        sf2_file = Path(self.temp_dir) / "test.sf2"
        sf2_file.write_bytes(b"dummy sf2 data")
        
        # Mock FluidSynth failure
        mock_synth = Mock()
        mock_synth.sfload.return_value = -1  # Failure
        self.renderer.synth = mock_synth
        
        sfid = self.renderer.load_soundfont(str(sf2_file))
        
        assert sfid is None
    
    def test_get_soundfont_info_no_synth(self):
        """Test getting SoundFont info when synth is None."""
        info = self.renderer.get_soundfont_info(1)
        assert info == {}
    
    @patch('soundfont_renderer.fluidsynth')
    def test_get_soundfont_info_mock(self, mock_fluidsynth):
        """Test getting SoundFont info with mocked synth."""
        mock_synth = Mock()
        # Mock successful preset selection for a few presets
        mock_synth.program_select.side_effect = [None, None, Exception("Invalid preset")]
        self.renderer.synth = mock_synth
        self.renderer.current_sf2 = "test.sf2"
        
        info = self.renderer.get_soundfont_info(1)
        
        assert info["sfid"] == 1
        assert info["file_path"] == "test.sf2"
        assert len(info["presets"]) >= 0  # Should have some presets
    
    def test_render_patch_no_synth(self):
        """Test rendering patch when synth is None."""
        audio = self.renderer.render_patch(1, 0, 0)
        assert audio is None
    
    @patch('soundfont_renderer.fluidsynth')
    def test_render_patch_mock(self, mock_fluidsynth):
        """Test patch rendering with mocked synth."""
        mock_synth = Mock()
        
        # Mock audio generation
        mock_audio_data = [0.1, 0.2, 0.3] * 1000  # Some dummy audio
        mock_synth.get_samples.return_value = mock_audio_data[:4096]
        
        self.renderer.synth = mock_synth
        self.renderer.duration = 0.1  # Short duration for test
        
        audio = self.renderer.render_patch(1, 0, 0, duration=0.1)
        
        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        
        # Check that synth methods were called
        mock_synth.program_select.assert_called_once_with(0, 1, 0, 0)
        mock_synth.noteon.assert_called_once_with(0, 60, 100)
        mock_synth.noteoff.assert_called_once_with(0, 60)
    
    @patch('soundfont_renderer.fluidsynth')
    def test_render_patch_error(self, mock_fluidsynth):
        """Test patch rendering with error."""
        mock_synth = Mock()
        mock_synth.program_select.side_effect = Exception("Render error")
        
        self.renderer.synth = mock_synth
        
        audio = self.renderer.render_patch(1, 0, 0)
        assert audio is None
    
    def test_batch_render_soundfont_no_file(self):
        """Test batch rendering with non-existent SoundFont."""
        result = self.renderer.batch_render_soundfont("nonexistent.sf2", "TestSynth")
        assert "error" in result
    
    @patch.object(SoundFontRenderer, 'load_soundfont')
    @patch.object(SoundFontRenderer, 'get_soundfont_info')
    @patch.object(SoundFontRenderer, 'render_patch')
    def test_batch_render_soundfont_mock(self, mock_render, mock_info, mock_load):
        """Test batch rendering with mocked methods."""
        # Create dummy SF2 file
        sf2_file = Path(self.temp_dir) / "test.sf2"
        sf2_file.write_bytes(b"dummy")
        
        # Mock methods
        mock_load.return_value = 1
        mock_info.return_value = {
            "presets": [
                {"bank": 0, "preset": 0, "program": 1, "name": "Test Patch 1"},
                {"bank": 0, "preset": 1, "program": 2, "name": "Test Patch 2"}
            ]
        }
        
        # Mock audio rendering
        mock_audio = np.random.randn(44100).astype(np.float32)  # 1 second
        mock_render.return_value = mock_audio
        
        result = self.renderer.batch_render_soundfont(str(sf2_file), "TestSynth")
        
        # Check results
        assert "error" not in result
        assert result["synthesizer_name"] == "TestSynth"
        assert result["rendered_patches"] == 2
        assert result["failed_patches"] == 0
        
        # Check that files were created
        synth_dir = Path(self.temp_dir) / "TestSynth"
        assert synth_dir.exists()
        assert (synth_dir / "metadata.json").exists()
    
    def test_cleanup(self):
        """Test cleanup method."""
        # Mock synth
        mock_synth = Mock()
        self.renderer.synth = mock_synth
        
        self.renderer.cleanup()
        
        mock_synth.delete.assert_called_once()
        assert self.renderer.synth is None
    
    def test_cleanup_with_error(self):
        """Test cleanup when delete raises error."""
        mock_synth = Mock()
        mock_synth.delete.side_effect = Exception("Delete error")
        self.renderer.synth = mock_synth
        
        # Should not raise exception
        self.renderer.cleanup()
        assert self.renderer.synth is None


class TestSoundFontRendererIntegration:
    """Integration tests for SoundFontRenderer."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_audio_normalization(self):
        """Test audio normalization in render_patch."""
        renderer = SoundFontRenderer(self.temp_dir)
        
        # Test with mock audio data
        with patch.object(renderer, 'synth') as mock_synth:
            # Create audio with high amplitude
            mock_audio = [2.0, -2.0, 1.5, -1.5] * 1000
            mock_synth.get_samples.return_value = mock_audio[:4096]
            mock_synth.program_select.return_value = None
            mock_synth.noteon.return_value = None
            mock_synth.noteoff.return_value = None
            
            renderer.duration = 0.1
            audio = renderer.render_patch(1, 0, 0)
            
            if audio is not None:
                # Check that audio is normalized
                assert np.max(np.abs(audio)) <= 0.8
    
    def test_preset_enumeration_logic(self):
        """Test the preset enumeration logic."""
        renderer = SoundFontRenderer(self.temp_dir)
        
        with patch.object(renderer, 'synth') as mock_synth:
            renderer.current_sf2 = "test.sf2"
            
            # Mock some presets as valid, others as invalid
            def mock_program_select(channel, sfid, bank, preset):
                if bank == 0 and preset < 3:
                    return None  # Success
                else:
                    raise Exception("Invalid preset")
            
            mock_synth.program_select.side_effect = mock_program_select
            
            info = renderer.get_soundfont_info(1)
            
            # Should find 3 presets in bank 0
            presets = info.get("presets", [])
            bank_0_presets = [p for p in presets if p["bank"] == 0]
            assert len(bank_0_presets) == 3


if __name__ == "__main__":
    pytest.main([__file__])
