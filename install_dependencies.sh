#!/bin/bash

echo "Installing system dependencies for audio processing..."

# For Debian/Ubuntu systems
if [ -x "$(command -v apt-get)" ]; then
    sudo apt-get update
    sudo apt-get install -y libsndfile1 ffmpeg
fi

# For CentOS/RHEL systems
if [ -x "$(command -v yum)" ]; then
    sudo yum install -y libsndfile ffmpeg
fi

# For macOS
if [ -x "$(command -v brew)" ]; then
    brew install libsndfile ffmpeg
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Dependencies installation completed."
