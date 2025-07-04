#!/usr/bin/env python3
"""
SoundFont Workflow Example for Instrument & Patch Identifier

This script demonstrates the complete workflow for creating synthesizer-specific
databases using SoundFont files and using them for targeted identification.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soundfont_renderer import SoundFontRenderer
from synth_db import SynthesizerDatabase
from inference import InstrumentIdentifier


def demo_soundfont_rendering():
    """Demonstrate SoundFont rendering workflow."""
    print("=== SoundFont Rendering Demo ===")
    
    # Example SoundFont file (you'll need to provide your own)
    sf2_file = "path/to/your/PSR-2100.sf2"
    synthesizer_name = "PSR-2100"
    
    if not Path(sf2_file).exists():
        print(f"Please provide a valid SoundFont file path.")
        print(f"Current path: {sf2_file}")
        print("\nTo get SoundFont files:")
        print("1. Search for 'PSR-2100.sf2' or 'Yamaha PSR-2100 SoundFont'")
        print("2. Download from synthesizer communities or forums")
        print("3. Convert from other formats using tools like Polyphone")
        return False
    
    try:
        # Create renderer
        renderer = SoundFontRenderer("data/soundfonts")
        
        # Check FluidSynth availability
        if not renderer.check_fluidsynth():
            print("FluidSynth not available. Please install:")
            print("  pip install pyfluidsynth")
            print("  # Also install system FluidSynth package")
            return False
        
        # Render patches (limit to 10 for demo)
        print(f"Rendering patches from {sf2_file}...")
        result = renderer.batch_render_soundfont(
            sf2_file,
            synthesizer_name,
            max_presets=10  # Limit for demo
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return False
        
        print(f"Successfully rendered {result['rendered_patches']} patches")
        print(f"Output directory: data/soundfonts/{synthesizer_name}")
        
        return True
        
    except Exception as e:
        print(f"Error during rendering: {e}")
        return False
    
    finally:
        renderer.cleanup()


def demo_synthesizer_database():
    """Demonstrate synthesizer database creation."""
    print("\n=== Synthesizer Database Demo ===")
    
    synthesizer_name = "PSR-2100"
    
    try:
        # Create database manager
        synth_db = SynthesizerDatabase("data")
        
        # Check if synthesizer exists
        synthesizers = synth_db.list_synthesizers()
        if synthesizer_name not in synthesizers:
            print(f"Synthesizer '{synthesizer_name}' not found.")
            print("Available synthesizers:", synthesizers)
            print("Please run demo_soundfont_rendering() first.")
            return False
        
        # Get synthesizer info
        info = synth_db.get_synthesizer_info(synthesizer_name)
        print(f"Synthesizer: {info['synthesizer_name']}")
        print(f"Patches: {info['rendered_patches']}")
        print(f"SoundFont: {info['soundfont_path']}")
        
        # Create embeddings
        print("Creating CLAP embeddings...")
        embed_result = synth_db.create_synthesizer_embeddings(synthesizer_name)
        
        if embed_result is None:
            print("Failed to create embeddings. CLAP model may not be available.")
            return False
        
        print(f"Created embeddings for {embed_result['total_patches']} patches")
        print(f"Embedding dimension: {embed_result['embedding_dimension']}")
        
        # Build index
        print("Building FAISS index...")
        index_result = synth_db.build_synthesizer_index(synthesizer_name, "flat")
        
        if index_result is None:
            print("Failed to build index.")
            return False
        
        print(f"Built index with {index_result['total_vectors']} vectors")
        
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def demo_synthesizer_identification():
    """Demonstrate synthesizer-specific identification."""
    print("\n=== Synthesizer-Specific Identification Demo ===")
    
    # Example audio file (you'll need to provide your own)
    audio_file = "path/to/your/audio.mp3"
    synthesizer_name = "PSR-2100"
    
    if not Path(audio_file).exists():
        print(f"Please provide a valid audio file path.")
        print(f"Current path: {audio_file}")
        return False
    
    try:
        # Create identifier
        identifier = InstrumentIdentifier()
        
        # Check if synthesizer is available
        available_synths = identifier.list_available_synthesizers()
        if synthesizer_name not in available_synths:
            print(f"Synthesizer '{synthesizer_name}' not available.")
            print("Available synthesizers:", available_synths)
            print("Please run demo_synthesizer_database() first.")
            return False
        
        # Run identification with synthesizer filter
        print(f"Analyzing {audio_file} with {synthesizer_name} filter...")
        results = identifier.identify_instruments(
            audio_file,
            use_separation=True,
            separation_model="4stems",
            synthesizer_filter=synthesizer_name
        )
        
        # Display results
        print_synthesizer_results(results)
        
        return True
        
    except Exception as e:
        print(f"Error during identification: {e}")
        return False


def demo_comparison():
    """Demonstrate comparison between general and synthesizer-specific identification."""
    print("\n=== Comparison Demo ===")
    
    audio_file = "path/to/your/audio.mp3"
    synthesizer_name = "PSR-2100"
    
    if not Path(audio_file).exists():
        print(f"Please provide a valid audio file path.")
        return False
    
    try:
        identifier = InstrumentIdentifier()
        
        # General identification
        print("Running general identification...")
        general_results = identifier.identify_instruments(
            audio_file,
            use_separation=False  # Faster for demo
        )
        
        # Synthesizer-specific identification
        if synthesizer_name in identifier.list_available_synthesizers():
            print(f"Running {synthesizer_name}-specific identification...")
            synth_results = identifier.identify_instruments(
                audio_file,
                use_separation=False,
                synthesizer_filter=synthesizer_name
            )
            
            # Compare results
            print("\n=== COMPARISON RESULTS ===")
            print("\nGeneral Database:")
            for match in general_results["summary"]["top_instruments"][:3]:
                print(f"  {match['instrument']} (confidence: {match['confidence']:.3f})")
            
            print(f"\n{synthesizer_name} Database:")
            for match in synth_results["summary"]["top_instruments"][:3]:
                print(f"  {match['instrument']} (confidence: {match['confidence']:.3f})")
        
        else:
            print(f"Synthesizer '{synthesizer_name}' not available for comparison.")
        
        return True
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False


def print_synthesizer_results(results):
    """Print formatted synthesizer identification results."""
    summary = results["summary"]
    
    print(f"\nResults for: {results['input_file']}")
    print(f"Synthesizer filter: {results['synthesizer_filter']}")
    print(f"Overall confidence: {summary['overall_confidence']:.3f}")
    
    print(f"\nTop patches detected:")
    for i, match in enumerate(summary["top_instruments"][:5], 1):
        print(f"  {i}. {match['instrument']} ({match['stem']}) - confidence: {match['confidence']:.3f}")
    
    if results['use_separation']:
        print(f"\nDetailed results by stem:")
        for stem_name, stem_summary in summary["stem_summaries"].items():
            if stem_summary["status"] == "success":
                print(f"  {stem_name}: {stem_summary['top_instrument']} "
                      f"(confidence: {stem_summary['confidence']:.3f})")
            else:
                print(f"  {stem_name}: Failed - {stem_summary['error']}")


def demo_batch_synthesizer_setup():
    """Demonstrate setting up multiple synthesizers."""
    print("\n=== Batch Synthesizer Setup Demo ===")
    
    # Example synthesizers and their SoundFont files
    synthesizers = {
        "PSR-2100": "path/to/PSR-2100.sf2",
        "Korg-Triton": "path/to/Triton.sf2",
        "Roland-JV1000": "path/to/JV1000.sf2"
    }
    
    renderer = SoundFontRenderer("data/soundfonts")
    synth_db = SynthesizerDatabase("data")
    
    if not renderer.check_fluidsynth():
        print("FluidSynth not available. Skipping batch setup.")
        return False
    
    for synth_name, sf2_file in synthesizers.items():
        if not Path(sf2_file).exists():
            print(f"Skipping {synth_name}: SoundFont file not found")
            continue
        
        try:
            print(f"\nProcessing {synth_name}...")
            
            # Render patches
            result = renderer.batch_render_soundfont(
                sf2_file, synth_name, max_presets=50  # Limit for demo
            )
            
            if "error" in result:
                print(f"  Rendering failed: {result['error']}")
                continue
            
            print(f"  Rendered {result['rendered_patches']} patches")
            
            # Create embeddings
            embed_result = synth_db.create_synthesizer_embeddings(synth_name)
            if embed_result:
                print(f"  Created embeddings for {embed_result['total_patches']} patches")
            
            # Build index
            index_result = synth_db.build_synthesizer_index(synth_name)
            if index_result:
                print(f"  Built index with {index_result['total_vectors']} vectors")
            
        except Exception as e:
            print(f"  Error processing {synth_name}: {e}")
    
    renderer.cleanup()
    
    # List all available synthesizers
    available = synth_db.list_synthesizers()
    print(f"\nSetup complete! Available synthesizers: {available}")
    
    return True


def main():
    """Run all SoundFont workflow demos."""
    print("🎹 SoundFont Workflow Demo for Instrument & Patch Identifier")
    print("=" * 60)
    
    print("\nThis demo shows how to:")
    print("1. Render patches from SoundFont files")
    print("2. Create synthesizer-specific databases")
    print("3. Use synthesizer filtering for identification")
    print("4. Compare general vs. synthesizer-specific results")
    
    # Check dependencies
    try:
        import fluidsynth
        print("\n✓ FluidSynth available")
    except ImportError:
        print("\n⚠️  FluidSynth not available. Install with: pip install pyfluidsynth")
        print("   Also install system FluidSynth package.")
    
    # Run demos
    print("\n" + "=" * 60)
    
    # Note: These demos require actual SoundFont files and audio files
    # Update the file paths in the functions above to use your own files
    
    print("\nTo run these demos:")
    print("1. Update file paths in the demo functions")
    print("2. Ensure you have SoundFont (.sf2) files")
    print("3. Install FluidSynth: pip install pyfluidsynth")
    print("4. Run individual demo functions")
    
    print("\nExample usage:")
    print("  demo_soundfont_rendering()")
    print("  demo_synthesizer_database()")
    print("  demo_synthesizer_identification()")
    print("  demo_comparison()")


if __name__ == "__main__":
    main()
