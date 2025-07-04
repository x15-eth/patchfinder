# Changelog

All notable changes to the Instrument & Patch Identifier project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-04

### Added
- Initial release of Instrument & Patch Identifier
- CLAP-based zero-shot instrument identification
- Source separation using Spleeter (2, 4, and 5 stem models)
- FAISS-powered fast similarity search
- Comprehensive database of 200+ instrument and synth patch labels
- Command-line interface for batch processing
- Web interface for interactive use
- Comprehensive test suite with pytest
- Support for multiple audio formats (WAV, MP3, FLAC, M4A)
- Extensible architecture for adding custom instruments
- Optional audio dataset integration (NSynth, FSD50K)
- Detailed JSON output with confidence scores
- Cross-platform support (Windows, Linux, macOS)

### Features
- Text embedding database with acoustic and electronic instruments
- Audio chunking with configurable overlap for robust analysis
- Result aggregation across multiple audio chunks
- Confidence scoring and ranking system
- Modular design for easy extension and customization
- Comprehensive error handling and logging
- Memory-efficient processing for large audio files

### Documentation
- Complete setup instructions for all platforms
- Usage examples for CLI and web interfaces
- API documentation for Python integration
- Troubleshooting guide for common issues
- Contributing guidelines for developers
