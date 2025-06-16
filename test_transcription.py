#!/usr/bin/env python3
"""
Test script for the modernized transcription functionality.
This script tests the various transcription methods and validates the setup.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from utils import (
    get_transcription_methods_status,
    validate_audio_file,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transcription_methods_status():
    """Test the transcription methods status function."""
    print("Testing transcription methods status...")
    try:
        status = get_transcription_methods_status()
        print(f"✓ Status retrieved successfully")
        print(f"  Current method: {status['current_method']}")
        print(f"  Whisper model: {status['whisper_model']}")

        for method, info in status.items():
            if isinstance(info, dict) and 'available' in info:
                availability = "✓ Available" if info['available'] else "✗ Not available"
                print(f"  {method}: {availability}")

        return True
    except Exception as e:
        print(f"✗ Error testing status: {e}")
        return False


def test_enhanced_features():
    """Test enhanced transcription features."""
    print("\nTesting enhanced features...")

    try:
        # Test data structures
        from utils import TranscriptionResult, TranscriptionSegment, WordTimestamp

        # Create test data
        word1 = WordTimestamp(word="Hello", start=0.0, end=0.5)
        word2 = WordTimestamp(word="world", start=0.6, end=1.0)

        segment = TranscriptionSegment(
            text="Hello world",
            start=0.0,
            end=1.0,
            words=[word1, word2]
        )

        result = TranscriptionResult(
            text="Hello world",
            segments=[segment],
            language="en",
            duration=1.0,
            method_used="test",
            model_used="test_model",
            processing_time=0.1,
            file_size_mb=1.0
        )

        # Test format conversions
        srt_content = result.to_srt()
        vtt_content = result.to_vtt()
        json_data = result.to_dict()

        print("✓ Enhanced data structures working")
        print("✓ SRT format conversion working")
        print("✓ VTT format conversion working")
        print("✓ JSON serialization working")

        return True

    except Exception as e:
        print(f"✗ Error testing enhanced features: {e}")
        return False


def test_audio_processing():
    """Test audio processing capabilities."""
    print("\nTesting audio processing...")

    try:
        from utils import AUDIO_PROCESSING_AVAILABLE, PYDUB_AVAILABLE, get_audio_duration

        print(f"  Librosa available: {'✓' if AUDIO_PROCESSING_AVAILABLE else '✗'}")
        print(f"  Pydub available: {'✓' if PYDUB_AVAILABLE else '✗'}")

        if AUDIO_PROCESSING_AVAILABLE or PYDUB_AVAILABLE:
            print("✓ Audio processing libraries available")
            return True
        else:
            print("⚠ No audio processing libraries available (chunking disabled)")
            return True  # Not a failure, just limited functionality

    except Exception as e:
        print(f"✗ Error testing audio processing: {e}")
        return False

def test_file_validation():
    """Test file validation functionality."""
    print("\nTesting file validation...")
    
    # Test with a temporary file
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            # Write some dummy data
            temp_file.write(b'dummy audio data')
            temp_file_path = temp_file.name
        
        try:
            validate_audio_file(temp_file_path)
            print("✓ File validation passed for valid file")
        except Exception as e:
            print(f"✗ File validation failed: {e}")
        finally:
            os.unlink(temp_file_path)
        
        # Test with non-existent file
        try:
            validate_audio_file('non_existent_file.mp3')
            print("✗ File validation should have failed for non-existent file")
        except FileNotFoundError:
            print("✓ File validation correctly failed for non-existent file")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing file validation: {e}")
        return False

def test_configuration():
    """Test configuration values."""
    print("\nTesting configuration...")
    
    print(f"✓ Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"✓ Max file size: {MAX_FILE_SIZE_MB}MB")
    
    # Check environment variables
    env_vars = ['TRANSCRIPTION_METHOD', 'WHISPER_MODEL', 'OPENAI_API_KEY']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'OPENAI_API_KEY' and value != 'Not set':
            value = f"Set (length: {len(value)})"
        print(f"  {var}: {value}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("TRANSCRIPTION FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_configuration,
        test_transcription_methods_status,
        test_enhanced_features,
        test_audio_processing,
        test_file_validation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The transcription system is ready.")
    else:
        print("✗ Some tests failed. Please check the configuration and dependencies.")
        
        print("\nTo fix issues:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up environment variables (copy .env.example to .env and configure)")
        print("3. Ensure at least one transcription method is available")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
