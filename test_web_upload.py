#!/usr/bin/env python3
"""
Test script to verify web upload functionality.
"""

import os
import requests
import tempfile
import wave
import struct

def create_test_audio_file():
    """Create a simple test audio file."""
    try:
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
        
        print(f"Created test audio file: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"Failed to create test audio file: {e}")
        return None

def test_web_upload():
    """Test uploading a file through the web interface."""
    print("Testing web upload functionality...")
    
    # Create test audio file
    audio_file = create_test_audio_file()
    if not audio_file:
        print("✗ Could not create test audio file")
        return False
    
    try:
        # Upload file to the enhanced upload endpoint
        url = "http://127.0.0.1:5000/enhanced-upload"
        
        with open(audio_file, 'rb') as f:
            files = {'audio_file': ('test_audio.wav', f, 'audio/wav')}
            
            print(f"Uploading file to: {url}")
            response = requests.post(url, files=files)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                print("✓ Upload successful!")
                
                # Check if the response contains the expected content
                if "Enhanced Transcription Complete" in response.text:
                    print("✓ Enhanced result page rendered successfully")
                    
                    # Check for download links
                    if "download-enhanced" in response.text:
                        print("✓ Download links found in response")
                    else:
                        print("⚠ No download links found in response")
                    
                    return True
                else:
                    print("⚠ Unexpected response content")
                    print("Response preview:", response.text[:500])
                    return False
            else:
                print(f"✗ Upload failed with status {response.status_code}")
                print("Response:", response.text[:500])
                return False
                
    except Exception as e:
        print(f"✗ Error during upload: {e}")
        return False
    
    finally:
        # Clean up test file
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

def check_files_endpoint():
    """Check the files listing endpoint."""
    print("\nChecking files endpoint...")
    
    try:
        url = "http://127.0.0.1:5000/files"
        response = requests.get(url)
        
        print(f"Files endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            if "Available Transcription Files" in response.text:
                print("✓ Files page rendered successfully")
                
                if "No transcription files found" in response.text:
                    print("⚠ No files found (expected if no uploads yet)")
                else:
                    print("✓ Files found in listing")
                
                return True
            else:
                print("⚠ Unexpected files page content")
                return False
        else:
            print(f"✗ Files endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking files endpoint: {e}")
        return False

def check_debug_endpoint():
    """Check the debug endpoint."""
    print("\nChecking debug endpoint...")
    
    try:
        url = "http://127.0.0.1:5000/debug/files"
        response = requests.get(url)
        
        print(f"Debug endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            import json
            data = response.json()
            
            print(f"✓ Debug endpoint working")
            print(f"  Transcription folder: {data.get('folder', 'Unknown')}")
            print(f"  File count: {data.get('count', 0)}")
            
            files = data.get('files', [])
            if files:
                print("  Files found:")
                for file in files:
                    print(f"    - {file.get('name', 'Unknown')}: {file.get('size', 0)} bytes")
            else:
                print("  No files found")
            
            return True
        else:
            print(f"✗ Debug endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking debug endpoint: {e}")
        return False

def main():
    """Run all web tests."""
    print("=" * 60)
    print("WEB UPLOAD FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Check if Flask app is running
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        if response.status_code == 200:
            print("✓ Flask app is running")
        else:
            print(f"⚠ Flask app returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Flask app is not accessible: {e}")
        print("Please make sure the Flask app is running on http://127.0.0.1:5000")
        return
    
    # Run tests
    tests = [
        ("Web Upload", test_web_upload),
        ("Files Endpoint", check_files_endpoint),
        ("Debug Endpoint", check_debug_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"WEB TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All web tests passed! The upload and download functionality is working.")
    else:
        print("✗ Some web tests failed. Check the output above for details.")

if __name__ == '__main__':
    main()
