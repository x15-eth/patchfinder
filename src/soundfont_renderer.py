"""
SoundFont Batch Renderer for Instrument & Patch Identifier

This module renders all patches from SoundFont (SF2) files using FluidSynth,
creating a comprehensive audio database for specific synthesizer models.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import sys
from tqdm import tqdm

try:
    import fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False
    print("FluidSynth not available. Install with: pip install pyfluidsynth")


class SoundFontRenderer:
    """Renders audio samples from SoundFont files using FluidSynth."""
    
    def __init__(self, output_dir: str = "data/soundfonts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = 44100
        self.duration = 5.0  # seconds per patch
        self.velocity = 100  # MIDI velocity
        self.note = 60  # Middle C (C4)
        
        self.synth = None
        self.current_sf2 = None
        
    def check_fluidsynth(self) -> bool:
        """Check if FluidSynth is available."""
        return FLUIDSYNTH_AVAILABLE
    
    def initialize_synth(self, driver: str = "alsa") -> bool:
        """
        Initialize FluidSynth synthesizer.
        
        Args:
            driver: Audio driver ("alsa", "coreaudio", "dsound", "pulseaudio")
            
        Returns:
            True if successful, False otherwise
        """
        if not self.check_fluidsynth():
            raise ImportError("FluidSynth not available. Install with: pip install pyfluidsynth")
        
        try:
            self.synth = fluidsynth.Synth(samplerate=self.sample_rate)
            
            # Try to start with specified driver, fallback to others
            drivers_to_try = [driver, "pulseaudio", "alsa", "coreaudio", "dsound", "file"]
            
            for drv in drivers_to_try:
                try:
                    self.synth.start(driver=drv)
                    print(f"FluidSynth started with {drv} driver")
                    return True
                except Exception as e:
                    print(f"Failed to start with {drv} driver: {e}")
                    continue
            
            print("Failed to start FluidSynth with any driver")
            return False
            
        except Exception as e:
            print(f"Error initializing FluidSynth: {e}")
            return False
    
    def load_soundfont(self, sf2_path: str) -> Optional[int]:
        """
        Load a SoundFont file.
        
        Args:
            sf2_path: Path to SF2 file
            
        Returns:
            SoundFont ID if successful, None otherwise
        """
        if not Path(sf2_path).exists():
            raise FileNotFoundError(f"SoundFont file not found: {sf2_path}")
        
        if self.synth is None:
            if not self.initialize_synth():
                return None
        
        try:
            sfid = self.synth.sfload(sf2_path)
            if sfid == -1:
                print(f"Failed to load SoundFont: {sf2_path}")
                return None
            
            self.current_sf2 = sf2_path
            print(f"Loaded SoundFont: {sf2_path} (ID: {sfid})")
            return sfid
            
        except Exception as e:
            print(f"Error loading SoundFont: {e}")
            return None
    
    def get_soundfont_info(self, sfid: int) -> Dict[str, Any]:
        """
        Get information about loaded SoundFont.
        
        Args:
            sfid: SoundFont ID
            
        Returns:
            Dictionary with SoundFont information
        """
        if self.synth is None:
            return {}
        
        try:
            # Get basic info
            info = {
                "sfid": sfid,
                "file_path": self.current_sf2,
                "presets": []
            }
            
            # Try to enumerate presets (this is FluidSynth version dependent)
            # We'll use a simple approach: try common preset numbers
            for bank in range(2):  # Usually banks 0 and 1
                for preset in range(128):  # MIDI presets 0-127
                    try:
                        # Try to select the preset
                        self.synth.program_select(0, sfid, bank, preset)
                        
                        # If no error, this preset exists
                        info["presets"].append({
                            "bank": bank,
                            "preset": preset,
                            "program": preset + 1,  # Program numbers are 1-based
                            "name": f"Bank {bank} Preset {preset + 1}"
                        })
                        
                    except Exception:
                        # Preset doesn't exist, continue
                        continue
            
            print(f"Found {len(info['presets'])} presets in SoundFont")
            return info
            
        except Exception as e:
            print(f"Error getting SoundFont info: {e}")
            return {"sfid": sfid, "presets": []}
    
    def render_patch(self, sfid: int, bank: int, preset: int, 
                    duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Render a single patch to audio.
        
        Args:
            sfid: SoundFont ID
            bank: MIDI bank number
            preset: MIDI preset number
            duration: Duration in seconds (uses default if None)
            
        Returns:
            Audio array or None if failed
        """
        if self.synth is None:
            return None
        
        render_duration = duration or self.duration
        
        try:
            # Select the patch
            self.synth.program_select(0, sfid, bank, preset)
            
            # Play a note
            self.synth.noteon(0, self.note, self.velocity)
            
            # Render audio
            samples = int(render_duration * self.sample_rate)
            audio_data = []
            
            # Render in chunks to avoid memory issues
            chunk_size = 4096
            chunks_needed = (samples + chunk_size - 1) // chunk_size
            
            for _ in range(chunks_needed):
                chunk = self.synth.get_samples(chunk_size)
                if isinstance(chunk, list):
                    audio_data.extend(chunk)
                else:
                    audio_data.extend(chunk.tolist())
            
            # Stop the note
            self.synth.noteoff(0, self.note)
            
            # Convert to numpy array and trim to exact length
            audio_array = np.array(audio_data[:samples], dtype=np.float32)
            
            # Normalize
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.8
            
            return audio_array
            
        except Exception as e:
            print(f"Error rendering patch {bank}:{preset}: {e}")
            return None
    
    def batch_render_soundfont(self, sf2_path: str, 
                              synthesizer_name: str,
                              max_presets: Optional[int] = None) -> Dict[str, Any]:
        """
        Batch render all patches from a SoundFont.
        
        Args:
            sf2_path: Path to SF2 file
            synthesizer_name: Name of the synthesizer (e.g., "PSR-2100")
            max_presets: Maximum number of presets to render (None for all)
            
        Returns:
            Dictionary with rendering results
        """
        print(f"Starting batch render of {sf2_path}")
        
        # Load SoundFont
        sfid = self.load_soundfont(sf2_path)
        if sfid is None:
            return {"error": "Failed to load SoundFont"}
        
        # Get SoundFont info
        sf_info = self.get_soundfont_info(sfid)
        presets = sf_info.get("presets", [])
        
        if not presets:
            return {"error": "No presets found in SoundFont"}
        
        # Limit presets if requested
        if max_presets:
            presets = presets[:max_presets]
        
        print(f"Rendering {len(presets)} presets...")
        
        # Create output directory for this synthesizer
        synth_dir = self.output_dir / synthesizer_name
        synth_dir.mkdir(exist_ok=True)
        
        # Render each preset
        rendered_patches = []
        failed_patches = []
        
        for i, preset_info in enumerate(tqdm(presets, desc="Rendering patches")):
            bank = preset_info["bank"]
            preset = preset_info["preset"]
            program = preset_info["program"]
            
            # Render audio
            audio = self.render_patch(sfid, bank, preset)
            
            if audio is not None:
                # Save audio file
                filename = f"patch_{program:03d}_bank{bank}_preset{preset}.wav"
                file_path = synth_dir / filename
                
                try:
                    sf.write(file_path, audio, self.sample_rate)
                    
                    rendered_patches.append({
                        "program": program,
                        "bank": bank,
                        "preset": preset,
                        "name": preset_info["name"],
                        "file_path": str(file_path),
                        "duration": len(audio) / self.sample_rate
                    })
                    
                except Exception as e:
                    print(f"Error saving patch {program}: {e}")
                    failed_patches.append(preset_info)
            else:
                failed_patches.append(preset_info)
        
        # Save metadata
        metadata = {
            "synthesizer_name": synthesizer_name,
            "soundfont_path": sf2_path,
            "total_presets": len(presets),
            "rendered_patches": len(rendered_patches),
            "failed_patches": len(failed_patches),
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "note": self.note,
            "velocity": self.velocity,
            "patches": rendered_patches
        }
        
        metadata_path = synth_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Batch rendering complete!")
        print(f"  Rendered: {len(rendered_patches)} patches")
        print(f"  Failed: {len(failed_patches)} patches")
        print(f"  Output directory: {synth_dir}")
        
        return metadata
    
    def cleanup(self):
        """Clean up FluidSynth resources."""
        if self.synth is not None:
            try:
                self.synth.delete()
            except:
                pass
            self.synth = None


def main():
    """Main function for batch rendering SoundFonts."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch render SoundFont patches")
    parser.add_argument("soundfont", help="Path to SF2 SoundFont file")
    parser.add_argument("--name", required=True, help="Synthesizer name (e.g., PSR-2100)")
    parser.add_argument("--max-presets", type=int, help="Maximum presets to render")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration per patch (seconds)")
    parser.add_argument("--note", type=int, default=60, help="MIDI note to play (60 = C4)")
    parser.add_argument("--velocity", type=int, default=100, help="MIDI velocity (0-127)")
    parser.add_argument("--output-dir", default="data/soundfonts", help="Output directory")
    parser.add_argument("--driver", default="alsa", help="Audio driver")
    
    args = parser.parse_args()
    
    # Create renderer
    renderer = SoundFontRenderer(args.output_dir)
    renderer.duration = args.duration
    renderer.note = args.note
    renderer.velocity = args.velocity
    
    try:
        # Initialize
        if not renderer.initialize_synth(args.driver):
            print("Failed to initialize FluidSynth")
            return 1
        
        # Render
        result = renderer.batch_render_soundfont(
            args.soundfont,
            args.name,
            args.max_presets
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        print(f"\nSuccess! Rendered {result['rendered_patches']} patches")
        print(f"Metadata saved to: {Path(args.output_dir) / args.name / 'metadata.json'}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        renderer.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())
