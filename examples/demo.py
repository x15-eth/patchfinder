#!/usr/bin/env python3
"""
Demo script for Instrument & Patch Identifier

This script demonstrates how to use the instrument identification system
with various configuration options.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import InstrumentIdentifier


def demo_basic_usage():
    """Demonstrate basic instrument identification."""
    print("=== Basic Usage Demo ===")
    
    # Initialize the identifier
    identifier = InstrumentIdentifier()
    
    # Example audio file (you'll need to provide your own)
    audio_file = "path/to/your/audio.mp3"
    
    if not Path(audio_file).exists():
        print(f"Please provide a valid audio file path in the demo script.")
        print(f"Current path: {audio_file}")
        return
    
    try:
        # Run identification with source separation
        print(f"Analyzing: {audio_file}")
        results = identifier.identify_instruments(
            audio_file,
            use_separation=True,
            separation_model="4stems"
        )
        
        # Display results
        print_results(results)
        
    except Exception as e:
        print(f"Error: {e}")


def demo_no_separation():
    """Demonstrate analysis without source separation (faster)."""
    print("\n=== No Separation Demo ===")
    
    identifier = InstrumentIdentifier()
    audio_file = "path/to/your/audio.mp3"
    
    if not Path(audio_file).exists():
        print(f"Please provide a valid audio file path.")
        return
    
    try:
        # Run without separation (faster)
        print(f"Analyzing full mix: {audio_file}")
        results = identifier.identify_instruments(
            audio_file,
            use_separation=False
        )
        
        print_results(results)
        
    except Exception as e:
        print(f"Error: {e}")


def demo_batch_processing():
    """Demonstrate batch processing of multiple files."""
    print("\n=== Batch Processing Demo ===")
    
    identifier = InstrumentIdentifier()
    
    # List of audio files to process
    audio_files = [
        "path/to/song1.mp3",
        "path/to/song2.wav",
        "path/to/song3.flac"
    ]
    
    results_summary = []
    
    for audio_file in audio_files:
        if not Path(audio_file).exists():
            print(f"Skipping {audio_file} (not found)")
            continue
        
        try:
            print(f"\nProcessing: {audio_file}")
            results = identifier.identify_instruments(audio_file)
            
            # Extract summary info
            summary = {
                "file": audio_file,
                "confidence": results["summary"]["overall_confidence"],
                "top_instrument": results["summary"]["top_instruments"][0] if results["summary"]["top_instruments"] else None
            }
            results_summary.append(summary)
            
            print(f"  Top instrument: {summary['top_instrument']['instrument'] if summary['top_instrument'] else 'None'}")
            print(f"  Confidence: {summary['confidence']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Print batch summary
    print(f"\n=== Batch Summary ===")
    for summary in results_summary:
        print(f"{summary['file']}: {summary['top_instrument']['instrument'] if summary['top_instrument'] else 'Unknown'} "
              f"({summary['confidence']:.3f})")


def demo_custom_analysis():
    """Demonstrate custom analysis with specific parameters."""
    print("\n=== Custom Analysis Demo ===")
    
    identifier = InstrumentIdentifier()
    
    # Customize chunk duration for different analysis granularity
    identifier.chunk_duration = 3.0  # 3-second chunks instead of 5
    identifier.overlap = 0.25  # 25% overlap instead of 50%
    
    audio_file = "path/to/your/audio.mp3"
    
    if not Path(audio_file).exists():
        print(f"Please provide a valid audio file path.")
        return
    
    try:
        print(f"Custom analysis with 3s chunks, 25% overlap")
        results = identifier.identify_instruments(audio_file)
        
        # Show detailed chunk analysis
        for stem_name, stem_result in results["stems"].items():
            if "error" not in stem_result:
                print(f"\n{stem_name.upper()}:")
                print(f"  Chunks analyzed: {stem_result.get('chunks_analyzed', 'N/A')}")
                print(f"  Duration: {stem_result['duration']:.2f}s")
                print(f"  Top 3 matches:")
                for match in stem_result["top_matches"][:3]:
                    print(f"    {match['rank']}. {match['label']} - {match['confidence']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")


def print_results(results):
    """Print formatted results."""
    summary = results["summary"]
    
    print(f"\nResults for: {results['input_file']}")
    print(f"Source separation: {'Yes' if results['use_separation'] else 'No'}")
    print(f"Overall confidence: {summary['overall_confidence']:.3f}")
    print(f"Stems analyzed: {summary['successful_analyses']}/{summary['total_stems']}")
    
    print(f"\nTop instruments detected:")
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


def save_results_demo():
    """Demonstrate saving results to file."""
    print("\n=== Save Results Demo ===")
    
    identifier = InstrumentIdentifier()
    audio_file = "path/to/your/audio.mp3"
    
    if not Path(audio_file).exists():
        print(f"Please provide a valid audio file path.")
        return
    
    try:
        results = identifier.identify_instruments(audio_file)
        
        # Save to JSON
        output_file = "analysis_results.json"
        identifier.save_results(results, output_file)
        print(f"Results saved to: {output_file}")
        
        # Save summary only
        summary_file = "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results["summary"], f, indent=2)
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all demos."""
    print("🎵 Instrument & Patch Identifier - Demo Script")
    print("=" * 50)
    
    # Check if system is set up
    try:
        from text_db import TextDatabase
        from build_index import IndexBuilder
        
        # Check if index exists
        data_dir = Path("data")
        if not (data_dir / "faiss.index").exists():
            print("⚠️  Index not found. Please run setup first:")
            print("   python src/text_db.py")
            print("   python src/build_index.py")
            return
        
    except ImportError as e:
        print(f"⚠️  Setup incomplete: {e}")
        print("Please run setup.sh or setup.bat first.")
        return
    
    # Run demos
    demo_basic_usage()
    demo_no_separation()
    demo_batch_processing()
    demo_custom_analysis()
    save_results_demo()
    
    print("\n" + "=" * 50)
    print("Demo complete! 🎉")
    print("\nTo use with your own audio files:")
    print("1. Update the audio_file paths in this script")
    print("2. Run: python examples/demo.py")


if __name__ == "__main__":
    main()
