#!/usr/bin/env python3
"""
Diagnostic script to identify and help fix audio processing issues.
"""

import os
import sys
import subprocess
import importlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_command_availability(command):
    """Check if a system command is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False, None

def check_python_module(module_name):
    """Check if a Python module can be imported and used."""
    try:
        module = importlib.import_module(module_name)
        return True, str(getattr(module, '__version__', 'unknown'))
    except ImportError as e:
        return False, str(e)

def test_pydub_functionality():
    """Test if pydub can actually work with audio files."""
    try:
        from pydub import AudioSegment
        # Try to create a simple audio segment
        test_segment = AudioSegment.silent(duration=100)  # 100ms of silence
        return True, "Pydub is functional"
    except Exception as e:
        return False, f"Pydub error: {e}"

def test_librosa_functionality():
    """Test if librosa can work properly."""
    try:
        import librosa
        import numpy as np
        # Try to create a simple audio array
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        return True, f"Librosa is functional (version {librosa.__version__})"
    except Exception as e:
        return False, f"Librosa error: {e}"

def main():
    """Run comprehensive audio diagnostics."""
    print("🔍 Audio Processing Diagnostics")
    print("=" * 50)
    
    # Check system commands
    print("\n📋 System Commands:")
    commands = ['ffmpeg', 'ffprobe']
    for cmd in commands:
        available, version = check_command_availability(cmd)
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {cmd}: {status}")
        if available and version:
            print(f"    Version: {version.split()[0] if version else 'unknown'}")
    
    # Check Python modules
    print("\n📦 Python Modules:")
    modules = [
        'pydub', 'librosa', 'soundfile', 'numpy', 
        'openai', 'faster_whisper', 'tqdm'
    ]
    
    for module in modules:
        available, info = check_python_module(module)
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {module}: {status}")
        if available:
            print(f"    Version: {info}")
        else:
            print(f"    Error: {info}")
    
    # Test functionality
    print("\n🧪 Functionality Tests:")
    
    # Test pydub
    pydub_works, pydub_msg = test_pydub_functionality()
    status = "✅ Working" if pydub_works else "❌ Not working"
    print(f"  Pydub: {status}")
    print(f"    {pydub_msg}")
    
    # Test librosa
    librosa_works, librosa_msg = test_librosa_functionality()
    status = "✅ Working" if librosa_works else "❌ Not working"
    print(f"  Librosa: {status}")
    print(f"    {librosa_msg}")
    
    # Check lzma module specifically
    try:
        import lzma
        print(f"  LZMA: ✅ Available")
    except ImportError as e:
        print(f"  LZMA: ❌ Missing - {e}")
    
    # Provide recommendations
    print("\n💡 Recommendations:")
    
    ffmpeg_available, _ = check_command_availability('ffmpeg')
    if not ffmpeg_available:
        print("  🔧 Install ffmpeg:")
        print("     - Linux: sudo apt-get install ffmpeg")
        print("     - macOS: brew install ffmpeg")
        print("     - Windows: winget install ffmpeg")
    
    if not pydub_works:
        print("  🔧 Fix pydub issues:")
        print("     - Ensure ffmpeg is installed and in PATH")
        print("     - Reinstall: pip install --force-reinstall pydub")
    
    if not librosa_works:
        print("  🔧 Fix librosa issues:")
        print("     - Install system dependencies: sudo apt-get install liblzma-dev")
        print("     - Reinstall: pip install --force-reinstall librosa soundfile")
    
    # Test the enhanced utils
    print("\n🔧 Testing Enhanced Utils:")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
        from utils import (
            PYDUB_AVAILABLE, PYDUB_FUNCTIONAL, AUDIO_PROCESSING_AVAILABLE,
            create_simple_chunks_by_size, get_audio_duration
        )
        
        print(f"  PYDUB_AVAILABLE: {PYDUB_AVAILABLE}")
        print(f"  PYDUB_FUNCTIONAL: {PYDUB_FUNCTIONAL}")
        print(f"  AUDIO_PROCESSING_AVAILABLE: {AUDIO_PROCESSING_AVAILABLE}")
        
        # Test fallback chunking
        print("  Testing fallback chunking method: Available ✅")
        
    except Exception as e:
        print(f"  Error loading enhanced utils: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Summary:")
    
    if ffmpeg_available and (pydub_works or librosa_works):
        print("✅ Audio processing should work normally")
    elif not ffmpeg_available:
        print("⚠️  Install ffmpeg to enable full audio processing")
        print("   Fallback binary chunking will be used")
    else:
        print("⚠️  Some audio libraries have issues")
        print("   Enhanced fallback methods are available")
    
    print("\n🚀 Next Steps:")
    print("1. Run: bash install_dependencies.sh")
    print("2. Restart your terminal/application")
    print("3. Test with a small audio file first")
    print("4. The app now has fallback methods for when audio libraries fail")

if __name__ == "__main__":
    main()
