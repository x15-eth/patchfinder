"""
Text Database Module for Instrument & Patch Identifier

This module creates text embeddings for a comprehensive list of instruments and synth patches
using the CLAP (Contrastive Language-Audio Pre-training) model.
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add CLAP to path if it exists
clap_path = Path(__file__).parent.parent / "CLAP"
if clap_path.exists():
    sys.path.insert(0, str(clap_path))

try:
    from src.laion_clap import CLAP_Module
except ImportError:
    print("CLAP not found. Please run setup.sh/setup.bat first to install CLAP.")
    sys.exit(1)


class TextDatabase:
    """Manages text embeddings for instrument and patch labels."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.model = None
        
    def load_model(self):
        """Load the CLAP model for text embeddings."""
        if self.model is None:
            print("Loading CLAP model...")
            self.model = CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()  # Load pre-trained checkpoint
            print("CLAP model loaded successfully.")
    
    def get_instrument_labels(self) -> List[str]:
        """
        Returns a comprehensive list of 200+ instrument and synth patch names.
        """
        labels = [
            # Acoustic Instruments - Strings
            "acoustic guitar", "electric guitar", "classical guitar", "12-string guitar",
            "bass guitar", "electric bass", "upright bass", "fretless bass", "slap bass",
            "violin", "viola", "cello", "double bass", "fiddle",
            "banjo", "mandolin", "ukulele", "sitar", "harp", "dulcimer",
            
            # Acoustic Instruments - Winds
            "flute", "piccolo", "clarinet", "oboe", "bassoon", "english horn",
            "saxophone", "alto saxophone", "tenor saxophone", "soprano saxophone", "baritone saxophone",
            "trumpet", "trombone", "french horn", "tuba", "cornet", "flugelhorn",
            "harmonica", "accordion", "bagpipes", "recorder", "pan flute",
            
            # Acoustic Instruments - Percussion
            "acoustic drums", "drum kit", "snare drum", "kick drum", "hi-hat", "crash cymbal",
            "ride cymbal", "tom-tom", "floor tom", "timpani", "xylophone", "marimba",
            "vibraphone", "glockenspiel", "triangle", "tambourine", "cowbell", "woodblock",
            "bongos", "congas", "djembe", "tabla", "cajon",
            
            # Piano & Keyboards
            "grand piano", "upright piano", "electric piano", "Rhodes piano", "Wurlitzer",
            "Fender Rhodes", "warm Rhodes", "bright Rhodes", "vintage electric piano",
            "harpsichord", "clavinet", "organ", "Hammond organ", "church organ", "pipe organ",
            
            # Synthesizers - Classic
            "Moog synthesizer", "Minimoog", "Moog bass", "Moog lead", "analog synthesizer",
            "Roland Jupiter-8", "Roland Juno-106", "Roland SH-101", "Yamaha DX7",
            "Oberheim OB-6", "Sequential Prophet-5", "ARP Odyssey", "Korg MS-20",
            
            # Synthesizer Patches - Bass
            "analog bass", "sub bass", "deep bass", "acid bass", "wobble bass",
            "reese bass", "FM bass", "distorted bass", "filtered bass", "pluck bass",
            "808 bass", "909 bass", "TB-303 bass", "Moog deep bass",
            
            # Synthesizer Patches - Leads
            "analog lead", "saw lead", "square lead", "sync lead", "acid lead",
            "distorted lead", "filtered lead", "arpeggio lead", "solo lead",
            "vintage lead", "modern lead", "aggressive lead", "smooth lead",
            
            # Synthesizer Patches - Pads
            "warm pad", "lush pad", "ambient pad", "string pad", "choir pad",
            "analog pad", "digital pad", "evolving pad", "atmospheric pad",
            "JP-8000 saw pad", "Juno pad", "vintage pad", "modern pad",
            "sweeping pad", "filtered pad", "reverb pad",
            
            # Synthesizer Patches - Plucks & Arps
            "pluck synth", "analog pluck", "bell pluck", "mallet pluck",
            "arpeggio", "sequenced arp", "fast arp", "slow arp", "gated arp",
            "vintage arp", "modern arp", "filtered arp",
            
            # Electronic Drums & Percussion
            "electronic drums", "drum machine", "808 drums", "909 drums", "707 drums",
            "trap drums", "hip hop drums", "techno drums", "house drums",
            "electronic kick", "electronic snare", "electronic hi-hat",
            "808 kick", "808 snare", "909 kick", "909 snare", "909 hi-hat",
            "clap", "electronic clap", "reverse snare", "gated snare",
            
            # Vocal & Choir
            "male vocals", "female vocals", "choir", "vocal harmony", "vocal lead",
            "background vocals", "gospel choir", "children's choir", "operatic vocals",
            "whispered vocals", "falsetto", "vocal pad", "vocoder", "talk box",
            
            # World Instruments
            "didgeridoo", "shakuhachi", "erhu", "guzheng", "koto", "shamisen",
            "oud", "balalaika", "bouzouki", "charango", "cuatro",
            "steel drums", "gamelan", "mbira", "kalimba", "hang drum",
            
            # Sound Effects & Textures
            "white noise", "pink noise", "vinyl crackle", "tape hiss", "rain",
            "wind", "ocean waves", "thunder", "fire crackling", "footsteps",
            "door slam", "glass break", "metal clang", "wood knock",
            "ambient texture", "drone", "field recording", "nature sounds",
            
            # Modern Electronic
            "dubstep bass", "future bass", "trap snare", "drill hi-hat",
            "lo-fi drums", "chillwave synth", "vaporwave pad", "synthwave lead",
            "retrowave bass", "outrun synth", "cyberpunk pad",
            
            # Ethnic & Traditional
            "didgeridoo", "native flute", "tribal drums", "ethnic percussion",
            "middle eastern oud", "indian tabla", "african djembe", "irish fiddle",
            "scottish bagpipes", "chinese erhu", "japanese koto", "gamelan gong",
            
            # Orchestral Sections
            "string section", "violin section", "viola section", "cello section",
            "brass section", "trumpet section", "trombone section", "horn section",
            "woodwind section", "flute section", "clarinet section", "oboe section",
            "full orchestra", "chamber orchestra", "symphony orchestra"
        ]
        
        return labels
    
    def create_embeddings(self, labels: List[str]) -> np.ndarray:
        """
        Create text embeddings for the given labels using CLAP.
        
        Args:
            labels: List of text labels to embed
            
        Returns:
            numpy array of embeddings with shape (n_labels, embedding_dim)
        """
        if self.model is None:
            self.load_model()
            
        print(f"Creating embeddings for {len(labels)} text labels...")
        
        # CLAP expects text in a specific format
        text_data = labels
        
        # Get text embeddings
        embeddings = self.model.get_text_embedding(text_data)
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, labels: List[str]):
        """
        Save embeddings and labels to disk.
        
        Args:
            embeddings: Text embeddings array
            labels: Corresponding text labels
        """
        # Save embeddings
        emb_path = self.data_dir / "text_embs.npy"
        np.save(emb_path, embeddings)
        print(f"Saved text embeddings to {emb_path}")
        
        # Save labels
        labels_path = self.data_dir / "text_labels.json"
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"Saved text labels to {labels_path}")
    
    def load_embeddings(self) -> tuple[np.ndarray, List[str]]:
        """
        Load embeddings and labels from disk.
        
        Returns:
            Tuple of (embeddings, labels)
        """
        emb_path = self.data_dir / "text_embs.npy"
        labels_path = self.data_dir / "text_labels.json"
        
        if not emb_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Text embeddings not found. Run create_text_database() first.")
        
        embeddings = np.load(emb_path)
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            
        return embeddings, labels
    
    def create_text_database(self):
        """Create and save the complete text database."""
        labels = self.get_instrument_labels()
        embeddings = self.create_embeddings(labels)
        self.save_embeddings(embeddings, labels)
        
        print(f"Text database created with {len(labels)} labels")
        return embeddings, labels


def main():
    """Main function to create the text database."""
    print("Creating text database for Instrument & Patch Identifier...")
    
    db = TextDatabase()
    embeddings, labels = db.create_text_database()
    
    print(f"\nDatabase created successfully!")
    print(f"- {len(labels)} instrument/patch labels")
    print(f"- Embedding dimension: {embeddings.shape[1]}")
    print(f"- Files saved in 'data/' directory")


if __name__ == "__main__":
    main()
