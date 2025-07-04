# 🎵 Instrument & Patch Identifier

An AI-powered tool that identifies instruments and synthesizer patches in audio using zero-shot audio/text embedding and nearest-neighbor search. Built with CLAP (Contrastive Language-Audio Pre-training) and FAISS for fast, accurate instrument recognition.

## ✨ Features

- **Zero-shot identification**: No training required - works out of the box
- **Source separation**: Automatically separates audio into vocals, drums, bass, and other instruments
- **200+ instrument labels**: Comprehensive database of acoustic instruments and synth patches
- **SoundFont support**: Batch-render patches from SF2 files for exact timbral matches
- **Synthesizer-specific filtering**: Target specific keyboard models (PSR-2100, Korg Triton, etc.)
- **Fast search**: FAISS-powered nearest neighbor search for real-time performance
- **Multiple interfaces**: Command-line tool and web interface
- **Extensible**: Easy to add new instruments or audio samples

## 🚀 Quick Start

### 1. Setup

```bash
# Clone and setup (Linux/Mac)
git clone <repository-url>
cd instrument_patch_identifier
chmod +x setup.sh
./setup.sh

# Windows
git clone <repository-url>
cd instrument_patch_identifier
setup.bat
```

### 2. Build the database

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

# Create text embeddings database
python src/text_db.py

# Build search index
python src/build_index.py
```

### 3. Analyze audio

```bash
# Command line interface
python src/ui.py cli your_song.mp3

# Web interface
python src/ui.py web
# Then open http://localhost:5000 in your browser
```

## 📁 Project Structure

```
instrument_patch_identifier/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.sh / setup.bat     # Setup scripts
├── data/                     # Data storage
│   ├── text_embs.npy        # Text embeddings
│   ├── text_labels.json     # Instrument labels
│   ├── audio_embs.npy       # Audio embeddings (optional)
│   ├── faiss.index          # Search index
│   └── audio_clips/         # Audio samples (optional)
├── src/                      # Source code
│   ├── text_db.py           # Text embedding creation
│   ├── audio_db.py          # Audio embedding creation
│   ├── soundfont_renderer.py # SoundFont batch rendering
│   ├── synth_db.py          # Synthesizer-specific databases
│   ├── build_index.py       # FAISS index building
│   ├── separate.py          # Audio source separation
│   ├── inference.py         # Main inference pipeline
│   └── ui.py                # User interfaces
├── tests/                    # Unit tests
└── output/                   # Separated audio output
```

## 🔧 Usage Examples

### Command Line Interface

```bash
# Basic usage
python src/ui.py cli song.mp3

# Skip source separation (faster)
python src/ui.py cli song.mp3 --no-separation

# Use different separation model
python src/ui.py cli song.mp3 --model 2stems

# Filter by specific synthesizer
python src/ui.py cli song.mp3 --synthesizer PSR-2100

# List available synthesizers
python src/ui.py cli --list-synthesizers

# Save results to JSON
python src/ui.py cli song.mp3 --output results.json --verbose
```

### Web Interface

```bash
# Start web server
python src/ui.py web

# Custom host/port
python src/ui.py web --host 0.0.0.0 --port 8080

# Debug mode
python src/ui.py web --debug
```

### Python API

```python
from src.inference import InstrumentIdentifier

# Initialize
identifier = InstrumentIdentifier()

# Analyze audio
results = identifier.identify_instruments(
    "song.mp3",
    use_separation=True,
    separation_model="4stems",
    synthesizer_filter="PSR-2100"  # Optional: filter by synthesizer
)

# Print top instruments
for match in results["summary"]["top_instruments"]:
    print(f"{match['instrument']} ({match['stem']}) - {match['confidence']:.3f}")
```

## 📊 Output Format

The tool outputs JSON with detailed analysis:

```json
{
  "input_file": "song.mp3",
  "use_separation": true,
  "separation_model": "4stems",
  "stems": {
    "vocals": {
      "top_matches": [
        {"label": "male vocals", "confidence": 0.892, "rank": 1},
        {"label": "female vocals", "confidence": 0.743, "rank": 2}
      ],
      "confidence_score": 0.892,
      "duration": 180.5
    },
    "drums": {
      "top_matches": [
        {"label": "acoustic drums", "confidence": 0.934, "rank": 1},
        {"label": "808 drums", "confidence": 0.621, "rank": 2}
      ]
    }
  },
  "summary": {
    "overall_confidence": 0.847,
    "top_instruments": [
      {"instrument": "acoustic drums", "stem": "drums", "confidence": 0.934},
      {"instrument": "male vocals", "stem": "vocals", "confidence": 0.892}
    ]
  }
}
```

## 🎹 SoundFont Workflow (Synthesizer-Specific Identification)

For the most accurate timbral matches, you can create synthesizer-specific databases using SoundFont files:

### Step 1: Get SoundFont Files

Download SF2 files for your target synthesizers:
- **PSR-2100**: Search for "PSR-2100.sf2" or "Yamaha PSR-2100 SoundFont"
- **Korg Triton**: Look for "Triton.sf2" or "Korg Triton SoundFont"
- **Roland JV-1000**: Find "JV1000.sf2" files
- Many keyboard patches are available online in SF2 format

### Step 2: Batch Render Patches

```bash
# Install FluidSynth (required for SoundFont rendering)
# Ubuntu/Debian: sudo apt-get install fluidsynth
# macOS: brew install fluidsynth
# Windows: Download from https://www.fluidsynth.org/

# Install Python FluidSynth bindings
pip install pyfluidsynth

# Render all patches from a SoundFont
python src/soundfont_renderer.py PSR-2100.sf2 --name PSR-2100

# Render with custom settings
python src/soundfont_renderer.py Triton.sf2 --name Korg-Triton \
    --duration 3.0 --note 60 --velocity 100 --max-presets 200
```

### Step 3: Create Synthesizer Database

```bash
# Create embeddings for the rendered patches
python src/synth_db.py embed PSR-2100

# Build FAISS index
python src/synth_db.py index PSR-2100

# List available synthesizers
python src/synth_db.py list
```

### Step 4: Use Synthesizer-Specific Identification

```bash
# Analyze with PSR-2100 filter
python src/ui.py cli song.mp3 --synthesizer PSR-2100

# Web interface will automatically show available synthesizers
python src/ui.py web
```

### Why This Works Better

- **Exact timbral matches**: Uses the actual synthesizer's sound engine
- **No physical keyboard needed**: Renders patches programmatically
- **Complete coverage**: Captures all 200+ patches in minutes
- **Consistent conditions**: Same note, velocity, and duration for all patches

## 🛠️ Advanced Usage

### Adding Custom Instruments

1. **Edit text labels** in `src/text_db.py`:
```python
def get_instrument_labels(self):
    labels = [
        # ... existing labels ...
        "your custom instrument",
        "another synth patch"
    ]
    return labels
```

2. **Rebuild the database**:
```bash
python src/text_db.py
python src/build_index.py
```

### Adding Audio Samples

1. **Add audio files** to `data/audio_clips/`
2. **Create audio embeddings**:
```bash
python src/audio_db.py --dataset local
```
3. **Rebuild index**:
```bash
python src/build_index.py
```

### Using Different Datasets

```bash
# Use NSynth dataset (requires download)
python src/audio_db.py --dataset nsynth --subset-size 1000

# Use FSD50K dataset (requires manual download)
python src/audio_db.py --dataset fsd50k --subset-size 500
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_text_db.py

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for faster inference)
- ~2GB disk space for models and data
- FFmpeg (for audio processing)

### Key Dependencies

- **torch**: Deep learning framework
- **CLAP**: Audio-text embedding model
- **faiss-cpu**: Fast similarity search
- **librosa**: Audio processing
- **spleeter**: Source separation
- **pyfluidsynth**: SoundFont rendering (optional)
- **flask**: Web interface

## 🔍 How It Works

1. **Text Embeddings**: Creates embeddings for 200+ instrument/patch names using CLAP
2. **Audio Processing**: Loads and chunks input audio into 5-second segments
3. **Source Separation**: Uses Spleeter to separate into vocals, drums, bass, other
4. **Audio Embeddings**: Converts each audio chunk to embeddings using CLAP
5. **Similarity Search**: Uses FAISS to find nearest text embeddings
6. **Aggregation**: Combines results across chunks and ranks by confidence

## 🎯 Performance Tips

- **GPU acceleration**: Install `torch` with CUDA support for faster inference
- **Index optimization**: Use IVF or HNSW indices for large datasets:
  ```bash
  python src/build_index.py --index-type ivf
  ```
- **Skip separation**: Use `--no-separation` for faster analysis of full mixes
- **Chunk size**: Adjust `chunk_duration` in `inference.py` for different trade-offs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LAION-AI](https://github.com/LAION-AI/CLAP) for the CLAP model
- [Deezer](https://github.com/deezer/spleeter) for Spleeter
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS
- NSynth and FSD50K datasets for audio samples

## 🐛 Troubleshooting

### Common Issues

**CLAP model not found**:
```bash
# Ensure CLAP is properly installed
cd CLAP && pip install -e .
```

**Spleeter errors**:
```bash
# Install TensorFlow if needed
pip install tensorflow
```

**Memory issues**:
- Reduce chunk size or use smaller audio files
- Use CPU-only versions of dependencies

**Web interface not loading**:
```bash
# Create templates
python src/ui.py setup
```

**FluidSynth errors**:
```bash
# Install FluidSynth system package first
# Ubuntu: sudo apt-get install fluidsynth
# macOS: brew install fluidsynth
# Then install Python bindings
pip install pyfluidsynth
```

**SoundFont rendering fails**:
- Ensure SF2 file is valid and not corrupted
- Try different audio drivers: `--driver pulseaudio` or `--driver alsa`
- Check FluidSynth installation: `fluidsynth --version`

For more issues, check the [Issues](https://github.com/x15-eth/patchfinder/issues) page.
