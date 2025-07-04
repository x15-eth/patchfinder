"""
Source Separation Module for Instrument & Patch Identifier

This module uses Spleeter to separate audio into stems (vocals, drums, bass, other)
for more accurate instrument identification.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import librosa
import soundfile as sf
import numpy as np


class AudioSeparator:
    """Handles audio source separation using Spleeter."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Spleeter model configurations
        self.models = {
            "2stems": "spleeter:2stems-16kHz",  # vocals, accompaniment
            "4stems": "spleeter:4stems-16kHz",  # vocals, drums, bass, other
            "5stems": "spleeter:5stems-16kHz"   # vocals, drums, bass, piano, other
        }
        
        self.stem_names = {
            "2stems": ["vocals", "accompaniment"],
            "4stems": ["vocals", "drums", "bass", "other"],
            "5stems": ["vocals", "drums", "bass", "piano", "other"]
        }
    
    def check_spleeter_installation(self) -> bool:
        """Check if Spleeter is properly installed."""
        try:
            result = subprocess.run(
                ["spleeter", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_spleeter_models(self, model_name: str = "4stems"):
        """
        Download Spleeter models if not already available.
        
        Args:
            model_name: Model to download ("2stems", "4stems", or "5stems")
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Downloading Spleeter {model_name} model...")
        
        # Create a dummy separation to trigger model download
        dummy_audio = np.zeros((16000, 2))  # 1 second of silence, stereo
        dummy_file = self.temp_dir / "dummy.wav"
        
        sf.write(dummy_file, dummy_audio, 16000)
        
        try:
            cmd = [
                "spleeter", "separate",
                "-p", self.models[model_name],
                "-o", str(self.temp_dir),
                str(dummy_file)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Spleeter {model_name} model downloaded successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {e}")
            raise
        finally:
            # Clean up dummy files
            if dummy_file.exists():
                dummy_file.unlink()
            dummy_output = self.temp_dir / "dummy"
            if dummy_output.exists():
                shutil.rmtree(dummy_output)
    
    def separate_audio(self, input_file: str, model_name: str = "4stems",
                      output_name: Optional[str] = None) -> Dict[str, str]:
        """
        Separate audio file into stems using Spleeter.
        
        Args:
            input_file: Path to input audio file
            model_name: Spleeter model to use ("2stems", "4stems", or "5stems")
            output_name: Custom output name (defaults to input filename)
            
        Returns:
            Dictionary mapping stem names to output file paths
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Check Spleeter installation
        if not self.check_spleeter_installation():
            raise RuntimeError("Spleeter not found. Please install with: pip install spleeter")
        
        # Determine output name
        if output_name is None:
            output_name = input_path.stem
        
        output_base = self.output_dir / output_name
        
        print(f"Separating {input_file} using {model_name} model...")
        
        try:
            # Run Spleeter
            cmd = [
                "spleeter", "separate",
                "-p", self.models[model_name],
                "-o", str(self.output_dir),
                str(input_path)
            ]
            
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print("Separation completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"Spleeter error: {e}")
            print(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            print("Spleeter timed out. File may be too large.")
            raise
        
        # Find output files
        stem_files = {}
        expected_output_dir = self.output_dir / input_path.stem
        
        if expected_output_dir.exists():
            for stem_name in self.stem_names[model_name]:
                stem_file = expected_output_dir / f"{stem_name}.wav"
                if stem_file.exists():
                    stem_files[stem_name] = str(stem_file)
                else:
                    print(f"Warning: Expected stem file not found: {stem_file}")
        else:
            print(f"Warning: Expected output directory not found: {expected_output_dir}")
        
        return stem_files
    
    def separate_with_fallback(self, input_file: str, 
                             preferred_model: str = "4stems") -> Dict[str, str]:
        """
        Separate audio with fallback to simpler models if needed.
        
        Args:
            input_file: Path to input audio file
            preferred_model: Preferred model to try first
            
        Returns:
            Dictionary mapping stem names to output file paths
        """
        models_to_try = [preferred_model, "4stems", "2stems"]
        
        for model in models_to_try:
            if model not in self.models:
                continue
                
            try:
                print(f"Trying separation with {model} model...")
                return self.separate_audio(input_file, model)
                
            except Exception as e:
                print(f"Failed with {model} model: {e}")
                if model == models_to_try[-1]:  # Last model
                    raise
                continue
        
        raise RuntimeError("All separation models failed")
    
    def convert_to_mono(self, stem_files: Dict[str, str]) -> Dict[str, str]:
        """
        Convert stereo stems to mono for easier processing.
        
        Args:
            stem_files: Dictionary of stem names to file paths
            
        Returns:
            Dictionary of stem names to mono file paths
        """
        mono_files = {}
        
        for stem_name, file_path in stem_files.items():
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=None, mono=True)
                
                # Save as mono
                mono_path = Path(file_path).with_suffix('.mono.wav')
                sf.write(mono_path, y, sr)
                
                mono_files[stem_name] = str(mono_path)
                print(f"Converted {stem_name} to mono: {mono_path}")
                
            except Exception as e:
                print(f"Error converting {stem_name} to mono: {e}")
                # Keep original file if conversion fails
                mono_files[stem_name] = file_path
        
        return mono_files
    
    def get_stem_info(self, stem_files: Dict[str, str]) -> Dict[str, Dict]:
        """
        Get information about separated stems.
        
        Args:
            stem_files: Dictionary of stem names to file paths
            
        Returns:
            Dictionary with stem information
        """
        stem_info = {}
        
        for stem_name, file_path in stem_files.items():
            try:
                # Load audio to get info
                y, sr = librosa.load(file_path, sr=None)
                
                stem_info[stem_name] = {
                    "file_path": file_path,
                    "duration": len(y) / sr,
                    "sample_rate": sr,
                    "channels": 1 if y.ndim == 1 else y.shape[0],
                    "rms_energy": float(np.sqrt(np.mean(y**2))),
                    "max_amplitude": float(np.max(np.abs(y)))
                }
                
            except Exception as e:
                print(f"Error analyzing {stem_name}: {e}")
                stem_info[stem_name] = {
                    "file_path": file_path,
                    "error": str(e)
                }
        
        return stem_info
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(exist_ok=True)


def main():
    """Main function for testing source separation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Separate audio into stems")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--model", choices=["2stems", "4stems", "5stems"], 
                       default="4stems", help="Spleeter model to use")
    parser.add_argument("--output-dir", default="output", 
                       help="Output directory for separated stems")
    parser.add_argument("--mono", action="store_true",
                       help="Convert outputs to mono")
    parser.add_argument("--info", action="store_true",
                       help="Show information about separated stems")
    
    args = parser.parse_args()
    
    separator = AudioSeparator(args.output_dir)
    
    try:
        print(f"Separating {args.input} with {args.model} model...")
        stem_files = separator.separate_audio(args.input, args.model)
        
        if args.mono:
            print("Converting to mono...")
            stem_files = separator.convert_to_mono(stem_files)
        
        print(f"\nSeparation complete! Generated {len(stem_files)} stems:")
        for stem_name, file_path in stem_files.items():
            print(f"  {stem_name}: {file_path}")
        
        if args.info:
            print("\nStem information:")
            stem_info = separator.get_stem_info(stem_files)
            for stem_name, info in stem_info.items():
                if "error" not in info:
                    print(f"  {stem_name}:")
                    print(f"    Duration: {info['duration']:.2f}s")
                    print(f"    Sample rate: {info['sample_rate']}Hz")
                    print(f"    RMS energy: {info['rms_energy']:.4f}")
                else:
                    print(f"  {stem_name}: Error - {info['error']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
