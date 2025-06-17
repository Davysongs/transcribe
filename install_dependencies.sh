#!/bin/bash

echo "Installing system dependencies for audio processing..."

# For Debian/Ubuntu systems
if [ -x "$(command -v apt-get)" ]; then
    echo "Detected Debian/Ubuntu system"
    sudo apt-get update
    sudo apt-get install -y libsndfile1 libsndfile1-dev ffmpeg liblzma-dev
    echo "Ubuntu/Debian dependencies installed successfully!"
fi

# For CentOS/RHEL systems
if [ -x "$(command -v yum)" ]; then
    echo "Detected CentOS/RHEL system"
    sudo yum install -y libsndfile ffmpeg xz-devel
    echo "CentOS/RHEL dependencies installed successfully!"
fi

# For macOS
if [ -x "$(command -v brew)" ]; then
    echo "Detected macOS system"
    brew install libsndfile ffmpeg
    echo "macOS dependencies installed successfully!"
fi

echo ""
echo "Installing/Reinstalling Python dependencies..."
pip install --force-reinstall pydub librosa soundfile

echo ""
echo "Dependencies installation completed."
echo ""
echo "To verify installation, run:"
echo "ffmpeg -version"
echo "ffprobe -version"
echo ""
echo "If you're on Windows, please install ffmpeg manually:"
echo "1. Download from: https://ffmpeg.org/download.html"
echo "2. Add to system PATH"
echo "3. Or use: winget install ffmpeg"
