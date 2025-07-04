#!/bin/bash

# Instrument & Patch Identifier Setup Script
echo "Setting up Instrument & Patch Identifier..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$python_version >= 3.10" | bc -l) -eq 0 ]]; then
    echo "Error: Python 3.10+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Clone and install CLAP
echo "Cloning and installing CLAP..."
if [ ! -d "CLAP" ]; then
    git clone https://github.com/LAION-AI/CLAP.git
fi
cd CLAP
pip install -e .
cd ..

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/audio_clips
mkdir -p data/downloads
mkdir -p output
mkdir -p temp

# Download pre-trained CLAP model (this will happen automatically on first use)
echo "CLAP model will be downloaded automatically on first use."

echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To get started, run:"
echo "python src/text_db.py  # Build text embeddings"
echo "python src/build_index.py  # Build search index"
echo "python src/ui.py --input your_song.mp3  # Analyze a song"
