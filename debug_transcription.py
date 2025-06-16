#!/usr/bin/env python3
"""
Debug script to test transcription and file saving.
"""

import os
import sys
import tempfile
import logging

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_test_file():
    """Create a simple test audio file."""
    try:
        import wave
        import struct
        
        # Create a temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Write a simple WAV file (1 second of silence)
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(44100)  # 44.1kHz
            
            # Write 1 second of silence
            for _ in range(44100):
                wav_file.writeframes(struct.pack('<h', 0))
        
        logger.info(f"Created test audio file: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to create test audio file: {e}")
        return None

def test_basic_transcription():
    """Test basic transcription functionality."""
    print("=" * 60)
    print("TESTING BASIC TRANSCRIPTION")
    print("=" * 60)
    
    try:
        from utils import transcribe_audio
        
        # Create test directories
        transcription_folder = os.path.join(os.getcwd(), 'test_transcriptions')
        os.makedirs(transcription_folder, exist_ok=True)
        
        # Create test audio file
        audio_file = create_simple_test_file()
        if not audio_file:
            print("✗ Could not create test audio file")
            return False
        
        try:
            print(f"Testing transcription with file: {audio_file}")
            print(f"Transcription folder: {transcription_folder}")
            
            # Test transcription
            text, transcription_path, transcription_filename = transcribe_audio(
                audio_file, 
                transcription_folder
            )
            
            print(f"✓ Transcription completed")
            print(f"  Text: {text[:100]}...")
            print(f"  File: {transcription_filename}")
            print(f"  Path: {transcription_path}")
            
            # Check if files were created
            print(f"\nChecking created files in: {transcription_folder}")
            created_files = os.listdir(transcription_folder)
            print(f"Created files: {created_files}")
            
            for file in created_files:
                file_path = os.path.join(transcription_folder, file)
                file_size = os.path.getsize(file_path)
                print(f"  - {file}: {file_size} bytes")
            
            return True
            
        finally:
            # Clean up
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
            
            # Clean up test transcription folder
            import shutil
            if os.path.exists(transcription_folder):
                shutil.rmtree(transcription_folder)
    
    except Exception as e:
        print(f"✗ Error testing basic transcription: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_transcription():
    """Test enhanced transcription functionality."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED TRANSCRIPTION")
    print("=" * 60)
    
    try:
        from utils import transcribe_audio_enhanced
        
        # Create test directories
        transcription_folder = os.path.join(os.getcwd(), 'test_transcriptions_enhanced')
        chunks_folder = os.path.join(os.getcwd(), 'test_chunks')
        os.makedirs(transcription_folder, exist_ok=True)
        os.makedirs(chunks_folder, exist_ok=True)
        
        # Create test audio file
        audio_file = create_simple_test_file()
        if not audio_file:
            print("✗ Could not create test audio file")
            return False
        
        try:
            print(f"Testing enhanced transcription with file: {audio_file}")
            print(f"Transcription folder: {transcription_folder}")
            print(f"Chunks folder: {chunks_folder}")
            
            # Test enhanced transcription
            result = transcribe_audio_enhanced(
                audio_file, 
                transcription_folder,
                chunks_folder
            )
            
            print(f"✓ Enhanced transcription completed")
            print(f"  Text: {result.text[:100]}...")
            print(f"  Method: {result.method_used}")
            print(f"  Model: {result.model_used}")
            print(f"  Duration: {result.duration}")
            print(f"  Segments: {len(result.segments) if result.segments else 0}")
            
            # Check if files were created
            print(f"\nChecking created files in: {transcription_folder}")
            created_files = os.listdir(transcription_folder)
            print(f"Created files: {created_files}")
            
            for file in created_files:
                file_path = os.path.join(transcription_folder, file)
                file_size = os.path.getsize(file_path)
                print(f"  - {file}: {file_size} bytes")
            
            return True
            
        finally:
            # Clean up
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
            
            # Clean up test folders
            import shutil
            for folder in [transcription_folder, chunks_folder]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
    
    except Exception as e:
        print(f"✗ Error testing enhanced transcription: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_transcription_folder():
    """Check the actual transcription folder used by the app."""
    print("\n" + "=" * 60)
    print("CHECKING ACTUAL TRANSCRIPTION FOLDER")
    print("=" * 60)
    
    transcription_folder = os.path.join(os.getcwd(), 'transcriptions')
    print(f"Transcription folder: {transcription_folder}")
    print(f"Exists: {os.path.exists(transcription_folder)}")
    
    if os.path.exists(transcription_folder):
        files = os.listdir(transcription_folder)
        print(f"Files in folder: {files}")
        
        for file in files:
            file_path = os.path.join(transcription_folder, file)
            file_size = os.path.getsize(file_path)
            file_modified = os.path.getmtime(file_path)
            print(f"  - {file}: {file_size} bytes, modified: {file_modified}")
    else:
        print("Transcription folder does not exist")

def main():
    """Run all debug tests."""
    print("TRANSCRIPTION DEBUG TESTS")
    
    # Check current transcription folder
    check_transcription_folder()
    
    # Test basic transcription
    basic_success = test_basic_transcription()
    
    # Test enhanced transcription
    enhanced_success = test_enhanced_transcription()
    
    # Final summary
    print("\n" + "=" * 60)
    print("DEBUG TEST SUMMARY")
    print("=" * 60)
    print(f"Basic transcription: {'✓ PASS' if basic_success else '✗ FAIL'}")
    print(f"Enhanced transcription: {'✓ PASS' if enhanced_success else '✗ FAIL'}")
    
    if basic_success and enhanced_success:
        print("✓ All tests passed! Transcription is working correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.")

if __name__ == '__main__':
    main()
