"""
Inference Pipeline for Instrument & Patch Identifier

This module performs the complete inference pipeline:
1. Source separation (optional)
2. Audio chunking and embedding
3. Nearest neighbor search
4. Result aggregation and ranking
"""

import json
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
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

from separate import AudioSeparator
from build_index import IndexBuilder


class InstrumentIdentifier:
    """Main inference pipeline for instrument and patch identification."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.index = None
        self.labels = None
        self.types = None
        self.metadata = None
        
        self.separator = AudioSeparator(output_dir)
        self.index_builder = IndexBuilder(data_dir)
        
        self.sample_rate = 48000  # CLAP expects 48kHz
        self.chunk_duration = 5.0  # seconds
        self.overlap = 0.5  # 50% overlap between chunks
    
    def load_model(self):
        """Load the CLAP model."""
        if self.model is None:
            print("Loading CLAP model...")
            self.model = CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()
            print("CLAP model loaded successfully.")
    
    def load_index(self):
        """Load the FAISS index and metadata."""
        if self.index is None:
            print("Loading FAISS index...")
            self.index, self.labels, self.types, self.metadata = self.index_builder.load_index()
            print(f"Index loaded with {self.index.ntotal} vectors")
    
    def chunk_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Split audio into overlapping chunks for analysis.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(self.chunk_duration * sr)
        hop_samples = int(chunk_samples * (1 - self.overlap))
        
        chunks = []
        start = 0
        
        while start + chunk_samples <= len(audio):
            chunk = audio[start:start + chunk_samples]
            chunks.append(chunk)
            start += hop_samples
        
        # Add final chunk if there's remaining audio
        if start < len(audio):
            final_chunk = audio[start:]
            # Pad if too short
            if len(final_chunk) < chunk_samples:
                padding = chunk_samples - len(final_chunk)
                final_chunk = np.pad(final_chunk, (0, padding), mode='constant')
            chunks.append(final_chunk)
        
        return chunks
    
    def analyze_audio_chunk(self, audio_chunk: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Analyze a single audio chunk and return top matches.
        
        Args:
            audio_chunk: Audio chunk to analyze
            k: Number of top matches to return
            
        Returns:
            List of match results
        """
        if self.model is None:
            self.load_model()
        if self.index is None:
            self.load_index()
        
        # Get audio embedding
        embedding = self.model.get_audio_embedding([audio_chunk])
        
        # Search index
        results = self.index_builder.search_index(
            self.index, embedding[0], self.labels, self.types, k=k
        )
        
        return results
    
    def analyze_stem(self, audio_file: str, stem_name: str = "full") -> Dict[str, Any]:
        """
        Analyze a single audio file/stem.
        
        Args:
            audio_file: Path to audio file
            stem_name: Name of the stem (for labeling)
            
        Returns:
            Analysis results for the stem
        """
        print(f"Analyzing {stem_name}: {audio_file}")
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            return {"error": str(e)}
        
        if len(audio) == 0:
            return {"error": "Empty audio file"}
        
        # Chunk audio
        chunks = self.chunk_audio(audio, sr)
        print(f"Split into {len(chunks)} chunks")
        
        # Analyze each chunk
        all_results = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_results = self.analyze_audio_chunk(chunk, k=10)
                all_results.extend(chunk_results)
            except Exception as e:
                print(f"Error analyzing chunk {i}: {e}")
                continue
        
        if not all_results:
            return {"error": "No valid analysis results"}
        
        # Aggregate results
        aggregated = self.aggregate_results(all_results)
        
        return {
            "stem": stem_name,
            "file": audio_file,
            "duration": len(audio) / sr,
            "chunks_analyzed": len(chunks),
            "top_matches": aggregated["top_matches"],
            "confidence_score": aggregated["confidence_score"],
            "match_distribution": aggregated["match_distribution"]
        }
    
    def aggregate_results(self, results: List[Dict[str, Any]], 
                         top_n: int = 5) -> Dict[str, Any]:
        """
        Aggregate results from multiple chunks.
        
        Args:
            results: List of search results from all chunks
            top_n: Number of top results to return
            
        Returns:
            Aggregated results
        """
        # Count label occurrences weighted by similarity
        label_scores = defaultdict(float)
        label_counts = defaultdict(int)
        
        for result in results:
            label = result["label"]
            similarity = result["similarity"]
            
            label_scores[label] += similarity
            label_counts[label] += 1
        
        # Calculate average scores
        label_avg_scores = {}
        for label in label_scores:
            label_avg_scores[label] = label_scores[label] / label_counts[label]
        
        # Sort by average score
        sorted_labels = sorted(
            label_avg_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Create top matches
        top_matches = []
        for i, (label, avg_score) in enumerate(sorted_labels[:top_n]):
            top_matches.append({
                "rank": i + 1,
                "label": label,
                "confidence": avg_score,
                "occurrences": label_counts[label],
                "total_score": label_scores[label]
            })
        
        # Calculate overall confidence
        if top_matches:
            confidence_score = top_matches[0]["confidence"]
        else:
            confidence_score = 0.0
        
        # Create match distribution
        match_distribution = dict(sorted_labels[:10])  # Top 10 for distribution
        
        return {
            "top_matches": top_matches,
            "confidence_score": confidence_score,
            "match_distribution": match_distribution
        }
    
    def identify_instruments(self, input_file: str, 
                           use_separation: bool = True,
                           separation_model: str = "4stems") -> Dict[str, Any]:
        """
        Complete instrument identification pipeline.
        
        Args:
            input_file: Path to input audio file
            use_separation: Whether to use source separation
            separation_model: Spleeter model to use
            
        Returns:
            Complete analysis results
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"Starting instrument identification for: {input_file}")
        
        results = {
            "input_file": str(input_path),
            "use_separation": use_separation,
            "separation_model": separation_model if use_separation else None,
            "stems": {}
        }
        
        if use_separation:
            try:
                print("Performing source separation...")
                stem_files = self.separator.separate_audio(input_file, separation_model)
                
                # Analyze each stem
                for stem_name, stem_file in stem_files.items():
                    stem_result = self.analyze_stem(stem_file, stem_name)
                    results["stems"][stem_name] = stem_result
                
            except Exception as e:
                print(f"Source separation failed: {e}")
                print("Falling back to full mix analysis...")
                use_separation = False
        
        if not use_separation:
            # Analyze full mix
            full_result = self.analyze_stem(input_file, "full_mix")
            results["stems"]["full_mix"] = full_result
        
        # Add summary
        results["summary"] = self.create_summary(results["stems"])

        return results

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    def create_summary(self, stem_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of all stem analysis results.
        
        Args:
            stem_results: Results from all stems
            
        Returns:
            Summary information
        """
        summary = {
            "total_stems": len(stem_results),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "overall_confidence": 0.0,
            "top_instruments": [],
            "stem_summaries": {}
        }
        
        all_confidences = []
        all_top_matches = []
        
        for stem_name, result in stem_results.items():
            if "error" in result:
                summary["failed_analyses"] += 1
                summary["stem_summaries"][stem_name] = {"status": "failed", "error": result["error"]}
            else:
                summary["successful_analyses"] += 1
                confidence = result.get("confidence_score", 0.0)
                all_confidences.append(confidence)
                
                # Get top match for this stem
                top_matches = result.get("top_matches", [])
                if top_matches:
                    top_match = top_matches[0]
                    all_top_matches.append({
                        "stem": stem_name,
                        "instrument": top_match["label"],
                        "confidence": top_match["confidence"]
                    })
                    
                    summary["stem_summaries"][stem_name] = {
                        "status": "success",
                        "top_instrument": top_match["label"],
                        "confidence": top_match["confidence"],
                        "duration": result.get("duration", 0)
                    }
        
        # Calculate overall confidence
        if all_confidences:
            summary["overall_confidence"] = np.mean(all_confidences)
        
        # Sort top instruments by confidence
        summary["top_instruments"] = sorted(
            all_top_matches, 
            key=lambda x: x["confidence"], 
            reverse=True
        )[:5]
        
        return summary


def main():
    """Main function for testing the inference pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Identify instruments in audio")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--no-separation", action="store_true",
                       help="Skip source separation")
    parser.add_argument("--model", choices=["2stems", "4stems", "5stems"],
                       default="4stems", help="Separation model")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    identifier = InstrumentIdentifier()
    
    try:
        results = identifier.identify_instruments(
            args.input,
            use_separation=not args.no_separation,
            separation_model=args.model
        )
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        # Print summary
        summary = results["summary"]
        print(f"\n=== Instrument Identification Results ===")
        print(f"Input: {results['input_file']}")
        print(f"Separation: {'Yes' if results['use_separation'] else 'No'}")
        print(f"Overall confidence: {summary['overall_confidence']:.3f}")
        print(f"Stems analyzed: {summary['successful_analyses']}/{summary['total_stems']}")
        
        print(f"\nTop instruments detected:")
        for i, match in enumerate(summary["top_instruments"], 1):
            print(f"  {i}. {match['instrument']} ({match['stem']}) - "
                  f"confidence: {match['confidence']:.3f}")
        
        if args.verbose:
            print(f"\nDetailed results by stem:")
            for stem_name, stem_summary in summary["stem_summaries"].items():
                if stem_summary["status"] == "success":
                    print(f"  {stem_name}: {stem_summary['top_instrument']} "
                          f"(confidence: {stem_summary['confidence']:.3f})")
                else:
                    print(f"  {stem_name}: Failed - {stem_summary['error']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
