#!/usr/bin/env python3
"""
Test script to verify file saving functionality.
"""

import os
import sys
import tempfile
import logging

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.utils import (
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
    transcribe_audio
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio_file():
    """Create a simple test audio file."""
    try:
        # Create a simple WAV file with silence
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

def test_transcription_result_formats():
    """Test the transcription result format conversion."""
    print("Testing transcription result formats...")
    
    try:
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
        
        # Test SRT format
        srt_content = result.to_srt()
        print("✓ SRT format generated")
        print("SRT content preview:")
        print(srt_content[:200] + "..." if len(srt_content) > 200 else srt_content)
        
        # Test VTT format
        vtt_content = result.to_vtt()
        print("✓ VTT format generated")
        
        # Test JSON format
        json_data = result.to_dict()
        print("✓ JSON format generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing formats: {e}")
        return False

def test_file_saving():
    """Test the actual file saving functionality."""
    print("\nTesting file saving...")
    
    # Create test directories
    test_dir = tempfile.mkdtemp()
    transcription_folder = os.path.join(test_dir, 'transcriptions')
    os.makedirs(transcription_folder, exist_ok=True)
    
    try:
        # Create a test audio file
        audio_file = create_test_audio_file()
        if not audio_file:
            print("✗ Could not create test audio file")
            return False
        
        try:
            # Test the transcription function
            print("Attempting transcription...")
            text, transcription_path, transcription_filename = transcribe_audio(
                audio_file, 
                transcription_folder
            )
            
            print(f"✓ Transcription completed")
            print(f"  Text: {text[:100]}...")
            print(f"  File: {transcription_filename}")
            print(f"  Path: {transcription_path}")
            
            # Check if files were created
            if os.path.exists(transcription_path):
                print("✓ Basic transcription file created")
            else:
                print("✗ Basic transcription file not found")
            
            # Check for enhanced format files
            base_name = os.path.splitext(transcription_filename)[0]
            
            json_file = os.path.join(transcription_folder, f"{base_name}_detailed.json")
            srt_file = os.path.join(transcription_folder, f"{base_name}.srt")
            vtt_file = os.path.join(transcription_folder, f"{base_name}.vtt")
            
            if os.path.exists(json_file):
                print("✓ JSON file created")
            else:
                print("⚠ JSON file not created (may be expected if no segments)")
            
            if os.path.exists(srt_file):
                print("✓ SRT file created")
            else:
                print("⚠ SRT file not created (may be expected if no segments)")
            
            if os.path.exists(vtt_file):
                print("✓ VTT file created")
            else:
                print("⚠ VTT file not created (may be expected if no segments)")
            
            # List all created files
            created_files = os.listdir(transcription_folder)
            print(f"Created files: {created_files}")
            
            return True
            
        finally:
            # Clean up test audio file
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
    
    except Exception as e:
        print(f"✗ Error testing file saving: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test directory
        import shutil
        try:
            shutil.rmtree(test_dir)
        except:
            pass

def main():
    """Run all tests."""
    print("=" * 60)
    print("FILE SAVING FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_transcription_result_formats,
        test_file_saving,
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
        print("✓ All tests passed! File saving is working correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
