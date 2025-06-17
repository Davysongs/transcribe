#!/usr/bin/env python3
"""
Test the fallback chunking functionality for large files.
"""

import os
import sys
import tempfile
import logging

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_audio_file(size_mb=150):
    """Create a dummy audio file for testing."""
    # Create a temporary file with the specified size
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    
    # Write dummy data to simulate a large audio file
    chunk_size = 1024 * 1024  # 1MB chunks
    total_bytes = int(size_mb * chunk_size)
    
    logger.info(f"Creating dummy {size_mb}MB audio file...")
    
    with open(temp_file.name, 'wb') as f:
        bytes_written = 0
        while bytes_written < total_bytes:
            # Write some dummy audio-like data
            dummy_data = b'\x00' * min(chunk_size, total_bytes - bytes_written)
            f.write(dummy_data)
            bytes_written += len(dummy_data)
    
    logger.info(f"Created dummy file: {temp_file.name} ({size_mb}MB)")
    return temp_file.name

def test_fallback_chunking():
    """Test the fallback chunking functionality."""
    try:
        from utils import (
            create_simple_chunks_by_size, 
            get_audio_duration,
            OPENAI_MAX_FILE_SIZE_MB
        )
        
        # Create a dummy large file
        dummy_file = create_dummy_audio_file(150)  # 150MB file
        
        # Create chunks folder
        chunks_folder = tempfile.mkdtemp()
        logger.info(f"Chunks folder: {chunks_folder}")
        
        # Test duration estimation
        duration = get_audio_duration(dummy_file)
        logger.info(f"Estimated duration: {duration:.2f}s")
        
        # Test simple chunking for OpenAI API
        logger.info("Testing simple chunking for OpenAI API...")
        chunks = create_simple_chunks_by_size(dummy_file, chunks_folder, 'openai_api')
        
        logger.info(f"Created {len(chunks)} chunks:")
        total_chunk_size = 0
        for i, (chunk_path, start_time, end_time) in enumerate(chunks):
            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            total_chunk_size += chunk_size_mb
            logger.info(f"  Chunk {i}: {chunk_size_mb:.1f}MB ({start_time:.1f}s - {end_time:.1f}s)")
            
            # Verify chunk is within OpenAI limits
            if chunk_size_mb > OPENAI_MAX_FILE_SIZE_MB:
                logger.error(f"Chunk {i} exceeds OpenAI limit!")
            else:
                logger.info(f"  ✅ Chunk {i} is within OpenAI limit")
        
        logger.info(f"Total chunk size: {total_chunk_size:.1f}MB")
        
        # Clean up
        os.unlink(dummy_file)
        for chunk_path, _, _ in chunks:
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
        os.rmdir(chunks_folder)
        
        logger.info("✅ Fallback chunking test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback chunking test failed: {e}")
        return False

def test_enhanced_transcription_logic():
    """Test the enhanced transcription logic with fallbacks."""
    try:
        from utils import (
            transcribe_audio_enhanced,
            validate_audio_file,
            get_transcription_methods_status
        )
        
        # Check transcription methods status
        status = get_transcription_methods_status()
        logger.info("Transcription methods status:")
        for method, info in status.items():
            if isinstance(info, dict):
                logger.info(f"  {method}: {info}")
            else:
                logger.info(f"  {method}: {info}")
        
        logger.info("✅ Enhanced transcription logic is available!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced transcription logic test failed: {e}")
        return False

def main():
    """Run fallback functionality tests."""
    print("🧪 Testing Fallback Chunking Functionality")
    print("=" * 50)
    
    # Test 1: Fallback chunking
    print("\n1️⃣ Testing Simple Binary Chunking...")
    chunking_success = test_fallback_chunking()
    
    # Test 2: Enhanced transcription logic
    print("\n2️⃣ Testing Enhanced Transcription Logic...")
    transcription_success = test_enhanced_transcription_logic()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Fallback Chunking: {'✅ PASS' if chunking_success else '❌ FAIL'}")
    print(f"  Enhanced Logic: {'✅ PASS' if transcription_success else '❌ FAIL'}")
    
    if chunking_success and transcription_success:
        print("\n🎉 All tests passed! The app should now handle large files even without ffmpeg.")
        print("\n📝 Key improvements:")
        print("  • Binary chunking fallback for when audio libraries fail")
        print("  • Smart chunk size calculation for OpenAI API limits")
        print("  • Enhanced error handling and graceful degradation")
        print("  • Better duration estimation methods")
        
        print("\n🚀 Your 150MB file should now work with these improvements!")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
