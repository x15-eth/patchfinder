"""
Audio Database Module for Instrument & Patch Identifier

This module processes audio datasets (NSynth, FSD50K, etc.) and creates audio embeddings
using the CLAP model for improved instrument identification.
"""

import json
import numpy as np
import os
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from tqdm import tqdm
import requests
import zipfile
import tarfile

# Add CLAP to path if it exists
clap_path = Path(__file__).parent.parent / "CLAP"
if clap_path.exists():
    sys.path.insert(0, str(clap_path))

try:
    from src.laion_clap import CLAP_Module
except ImportError:
    print("CLAP not found. Please run setup.sh/setup.bat first to install CLAP.")
    sys.exit(1)


class AudioDatabase:
    """Manages audio embeddings from public datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.audio_dir = self.data_dir / "audio_clips"
        self.audio_dir.mkdir(exist_ok=True)
        self.downloads_dir = self.data_dir / "downloads"
        self.downloads_dir.mkdir(exist_ok=True)
        self.model = None
        self.sample_rate = 48000  # CLAP expects 48kHz
        
    def load_model(self):
        """Load the CLAP model for audio embeddings."""
        if self.model is None:
            print("Loading CLAP model...")
            self.model = CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()  # Load pre-trained checkpoint
            print("CLAP model loaded successfully.")
    
    def download_nsynth_subset(self, subset_size: int = 1000) -> List[str]:
        """
        Download a subset of NSynth dataset for testing.
        Note: This is a simplified version. Full NSynth is very large.
        
        Args:
            subset_size: Number of samples to download
            
        Returns:
            List of downloaded file paths
        """
        print(f"Downloading NSynth subset ({subset_size} samples)...")
        
        # NSynth test set is smaller and good for our purposes
        nsynth_url = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
        nsynth_file = self.downloads_dir / "nsynth-test.jsonwav.tar.gz"
        
        if not nsynth_file.exists():
            print("Downloading NSynth test set...")
            response = requests.get(nsynth_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(nsynth_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        # Extract files
        extract_dir = self.downloads_dir / "nsynth-test"
        if not extract_dir.exists():
            print("Extracting NSynth files...")
            with tarfile.open(nsynth_file, 'r:gz') as tar:
                tar.extractall(self.downloads_dir)
        
        # Get audio files
        audio_files = list((extract_dir / "nsynth-test" / "audio").glob("*.wav"))[:subset_size]
        
        print(f"Found {len(audio_files)} NSynth audio files")
        return [str(f) for f in audio_files]
    
    def download_fsd50k_subset(self, subset_size: int = 500) -> List[str]:
        """
        Download a subset of FSD50K dataset.
        Note: This is a placeholder - FSD50K requires registration.
        
        Args:
            subset_size: Number of samples to download
            
        Returns:
            List of downloaded file paths
        """
        print("FSD50K requires manual download and registration.")
        print("Please visit: https://zenodo.org/record/4060432")
        print("For now, using local audio files if available...")
        
        # Look for any existing audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.audio_dir.glob(ext))
        
        return [str(f) for f in audio_files[:subset_size]]
    
    def load_audio(self, file_path: str, duration: Optional[float] = None) -> np.ndarray:
        """
        Load audio file with librosa at the correct sample rate for CLAP.
        
        Args:
            file_path: Path to audio file
            duration: Optional duration to load (in seconds)
            
        Returns:
            Audio array normalized to [-1, 1]
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration, mono=True)
            return y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def create_audio_embeddings(self, audio_files: List[str], 
                              chunk_duration: float = 5.0) -> Tuple[np.ndarray, List[str]]:
        """
        Create audio embeddings for a list of audio files.
        
        Args:
            audio_files: List of audio file paths
            chunk_duration: Duration of each audio chunk in seconds
            
        Returns:
            Tuple of (embeddings array, file paths list)
        """
        if self.model is None:
            self.load_model()
        
        embeddings_list = []
        valid_files = []
        
        print(f"Creating embeddings for {len(audio_files)} audio files...")
        
        for file_path in tqdm(audio_files, desc="Processing audio files"):
            # Load audio
            audio = self.load_audio(file_path, duration=chunk_duration)
            if audio is None:
                continue
            
            # Ensure minimum length
            min_samples = int(1.0 * self.sample_rate)  # 1 second minimum
            if len(audio) < min_samples:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, min_samples - len(audio)))
            
            try:
                # Get audio embedding
                # CLAP expects audio as a list of arrays
                embedding = self.model.get_audio_embedding([audio])
                embeddings_list.append(embedding[0])  # Extract single embedding
                valid_files.append(file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if embeddings_list:
            embeddings = np.vstack(embeddings_list)
            print(f"Created {len(embeddings)} audio embeddings with shape: {embeddings.shape}")
            return embeddings, valid_files
        else:
            print("No valid audio embeddings created")
            return np.array([]), []
    
    def save_audio_embeddings(self, embeddings: np.ndarray, file_paths: List[str]):
        """
        Save audio embeddings and file paths to disk.
        
        Args:
            embeddings: Audio embeddings array
            file_paths: Corresponding audio file paths
        """
        if len(embeddings) == 0:
            print("No embeddings to save")
            return
            
        # Save embeddings
        emb_path = self.data_dir / "audio_embs.npy"
        np.save(emb_path, embeddings)
        print(f"Saved audio embeddings to {emb_path}")
        
        # Save file paths
        paths_file = self.data_dir / "audio_paths.json"
        with open(paths_file, 'w') as f:
            json.dump(file_paths, f, indent=2)
        print(f"Saved audio paths to {paths_file}")
    
    def load_audio_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load audio embeddings and file paths from disk.
        
        Returns:
            Tuple of (embeddings, file_paths)
        """
        emb_path = self.data_dir / "audio_embs.npy"
        paths_file = self.data_dir / "audio_paths.json"
        
        if not emb_path.exists() or not paths_file.exists():
            print("Audio embeddings not found. Run create_audio_database() first.")
            return np.array([]), []
        
        embeddings = np.load(emb_path)
        with open(paths_file, 'r') as f:
            file_paths = json.load(f)
            
        return embeddings, file_paths
    
    def create_audio_database(self, dataset: str = "local", subset_size: int = 1000):
        """
        Create and save the complete audio database.
        
        Args:
            dataset: Dataset to use ("nsynth", "fsd50k", or "local")
            subset_size: Number of samples to process
        """
        print(f"Creating audio database from {dataset} dataset...")
        
        if dataset == "nsynth":
            audio_files = self.download_nsynth_subset(subset_size)
        elif dataset == "fsd50k":
            audio_files = self.download_fsd50k_subset(subset_size)
        else:  # local
            print("Using local audio files...")
            audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(self.audio_dir.glob(ext))
            audio_files = [str(f) for f in audio_files[:subset_size]]
        
        if not audio_files:
            print("No audio files found. Please add some audio files to data/audio_clips/")
            print("Or try downloading a dataset with dataset='nsynth'")
            return np.array([]), []
        
        embeddings, valid_files = self.create_audio_embeddings(audio_files)
        
        if len(embeddings) > 0:
            self.save_audio_embeddings(embeddings, valid_files)
            print(f"Audio database created with {len(embeddings)} samples")
        
        return embeddings, valid_files


def main():
    """Main function to create the audio database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create audio database for Instrument & Patch Identifier")
    parser.add_argument("--dataset", choices=["local", "nsynth", "fsd50k"], default="local",
                       help="Dataset to use for audio embeddings")
    parser.add_argument("--subset-size", type=int, default=1000,
                       help="Number of audio samples to process")
    
    args = parser.parse_args()
    
    print("Creating audio database for Instrument & Patch Identifier...")
    
    db = AudioDatabase()
    embeddings, files = db.create_audio_database(args.dataset, args.subset_size)
    
    if len(embeddings) > 0:
        print(f"\nAudio database created successfully!")
        print(f"- {len(embeddings)} audio samples")
        print(f"- Embedding dimension: {embeddings.shape[1]}")
        print(f"- Files saved in 'data/' directory")
    else:
        print("\nNo audio database created. Please add audio files or try a different dataset.")


if __name__ == "__main__":
    main()
