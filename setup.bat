@echo off
REM Instrument & Patch Identifier Setup Script for Windows
echo Setting up Instrument & Patch Identifier...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python dependencies...
pip install -r requirements.txt

REM Clone and install CLAP
echo Cloning and installing CLAP...
if not exist "CLAP" (
    git clone https://github.com/LAION-AI/CLAP.git
)
cd CLAP
pip install -e .
cd ..

REM Create necessary directories
echo Creating data directories...
if not exist "data\audio_clips" mkdir data\audio_clips
if not exist "data\downloads" mkdir data\downloads
if not exist "output" mkdir output
if not exist "temp" mkdir temp

echo Setup complete!
echo.
echo To activate the environment, run:
echo venv\Scripts\activate.bat
echo.
echo To get started, run:
echo python src\text_db.py  # Build text embeddings
echo python src\build_index.py  # Build search index
echo python src\ui.py --input your_song.mp3  # Analyze a song
pause
