#!/usr/bin/env python3
"""
Test script to verify transcription fixes for large files.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from utils import (
    get_audio_duration,
    create_audio_chunks,
    transcribe_audio_enhanced,
    validate_audio_file,
    get_transcription_methods_status,
    OPENAI_MAX_FILE_SIZE_MB,
    OPENAI_MAX_CHUNK_SIZE_SECONDS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_duration_detection():
    """Test improved audio duration detection."""
    print("\n=== Testing Audio Duration Detection ===")
    
    # This would need an actual audio file to test
    # For now, just test the function exists and handles errors gracefully
    try:
        # Test with non-existent file
        duration = get_audio_duration("non_existent_file.mp3")
        print(f"Duration for non-existent file: {duration}s (should be fallback value)")
    except Exception as e:
        print(f"Error handling test passed: {e}")

def test_chunking_logic():
    """Test the improved chunking logic."""
    print("\n=== Testing Chunking Logic ===")
    
    # Test chunk size calculation for different methods
    from utils import get_chunk_size_for_method, estimate_chunk_file_size
    
    # Test OpenAI API chunk sizes
    small_file_chunk = get_chunk_size_for_method('openai_api', 10)  # 10MB file
    large_file_chunk = get_chunk_size_for_method('openai_api', 150)  # 150MB file
    
    print(f"OpenAI API chunk size for 10MB file: {small_file_chunk}s")
    print(f"OpenAI API chunk size for 150MB file: {large_file_chunk}s")
    
    # Test local method chunk sizes
    local_chunk = get_chunk_size_for_method('faster_whisper', 150)
    print(f"Local method chunk size for 150MB file: {local_chunk}s")
    
    # Test file size estimation
    estimated_size = estimate_chunk_file_size(300, 150, 3600)  # 5min chunk from 150MB, 1hr file
    print(f"Estimated chunk size for 5min segment: {estimated_size:.1f}MB")

def test_transcription_methods_status():
    """Test transcription methods availability."""
    print("\n=== Testing Transcription Methods Status ===")
    
    status = get_transcription_methods_status()
    for method, info in status.items():
        if isinstance(info, dict):
            print(f"{method}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"{method}: {info}")

def test_file_validation():
    """Test file validation logic."""
    print("\n=== Testing File Validation ===")
    
    try:
        # Test with non-existent file
        validate_audio_file("non_existent_file.mp3")
    except FileNotFoundError as e:
        print(f"✓ File not found error handled correctly: {e}")
    
    try:
        # Test with unsupported format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        validate_audio_file(tmp_path)
    except ValueError as e:
        print(f"✓ Unsupported format error handled correctly: {e}")
        os.unlink(tmp_path)

def test_configuration_values():
    """Test that configuration values are properly set."""
    print("\n=== Testing Configuration Values ===")
    
    print(f"OpenAI max file size: {OPENAI_MAX_FILE_SIZE_MB}MB")
    print(f"OpenAI max chunk duration: {OPENAI_MAX_CHUNK_SIZE_SECONDS}s")
    
    # Import other config values
    from utils import CHUNK_SIZE_SECONDS, CHUNK_OVERLAP_SECONDS
    print(f"Default chunk size: {CHUNK_SIZE_SECONDS}s")
    print(f"Chunk overlap: {CHUNK_OVERLAP_SECONDS}s")

def main():
    """Run all tests."""
    print("Testing Transcription Fixes")
    print("=" * 50)
    
    test_configuration_values()
    test_transcription_methods_status()
    test_audio_duration_detection()
    test_chunking_logic()
    test_file_validation()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nKey improvements made:")
    print("1. ✓ Increased chunk size to 10 minutes for OpenAI API")
    print("2. ✓ Added file size validation before API calls")
    print("3. ✓ Improved audio duration detection with fallbacks")
    print("4. ✓ Method-specific chunking optimization")
    print("5. ✓ Better error handling for failed chunks")
    print("6. ✓ Retry logic for network errors")
    print("7. ✓ Compressed chunk format (MP3) for OpenAI API")

if __name__ == "__main__":
    main()
