# Modern Audio Transcription Flask App

A modernized Flask web application for transcribing audio files to text using multiple state-of-the-art transcription methods including OpenAI's Whisper API, faster-whisper, and the original openai-whisper package.

## 🚀 Features

### Core Functionality
- **Multiple Transcription Methods**: OpenAI API, faster-whisper (local), openai-whisper (local)
- **Automatic Fallback**: If one method fails, automatically tries the next available method
- **Enhanced File Support**: MP3, WAV, M4A, FLAC, OGG, WebM, MP4
- **Smart File Validation**: File size and format validation before processing
- **Progress Tracking**: Real-time progress indicators during transcription

### User Interface
- **Modern Responsive Design**: Clean, mobile-friendly interface
- **Drag & Drop Upload**: Drag files directly onto the upload area
- **Real-time Feedback**: File size display, validation messages, progress bars
- **Service Status Display**: Shows which transcription methods are available
- **Enhanced Error Handling**: Clear, user-friendly error messages

### Technical Improvements
- **Environment-based Configuration**: Flexible configuration via environment variables
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Robust Error Handling**: Graceful handling of API failures, network issues, and file problems
- **Security Enhancements**: Secure file handling and validation

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Optional: OpenAI API key for cloud-based transcription
- Optional: CUDA-compatible GPU for faster local processing

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Davysongs/transcribe.git
cd transcribe
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your preferred settings
# At minimum, set your OpenAI API key if using the API method
```

### 5. Test Installation
```bash
python test_transcription.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI API Configuration (required for cloud transcription)
OPENAI_API_KEY=your_openai_api_key_here

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=development

# Transcription Configuration
# Options: 'openai_api', 'faster_whisper', 'openai_whisper', 'auto'
TRANSCRIPTION_METHOD=openai_api

# Local Whisper Model (for local methods)
# Options: 'tiny', 'base', 'small', 'medium', 'large', 'turbo'
WHISPER_MODEL=base

# File Upload Configuration
MAX_FILE_SIZE_MB=25
```

### Transcription Methods

1. **OpenAI API** (Recommended for production)
   - Requires: OpenAI API key
   - Pros: Fast, accurate, always up-to-date, no local compute needed
   - Cons: Costs per usage, requires internet

2. **faster-whisper** (Recommended for local processing)
   - Requires: `pip install faster-whisper`
   - Pros: Much faster than original whisper, lower memory usage
   - Cons: Requires local compute resources

3. **openai-whisper** (Original implementation)
   - Requires: `pip install openai-whisper`
   - Pros: Official implementation, works offline
   - Cons: Slower, higher memory usage

## 🚀 Usage

### 1. Start the Application
```bash
python main.py
```

### 2. Access the Web Interface
Open your browser and navigate to `http://localhost:5000`

### 3. Upload and Transcribe
1. **Upload**: Click "Choose an audio file" or drag & drop a file
2. **Validate**: The app will show file size and validate format
3. **Transcribe**: Click "Transcribe" to start processing
4. **Download**: View results and download the transcription file

### 4. Monitor Progress
- Real-time progress bar during transcription
- Service status indicators show available methods
- Clear error messages if issues occur

## 📁 File Structure

```
transcribe/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # Web routes and handlers
│   ├── utils.py             # Transcription utilities
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Enhanced styling
│   │   └── js/
│   │       └── scripts.js   # Interactive features
│   └── templates/
│       ├── index.html       # Upload interface
│       └── result.html      # Results display
├── uploads/                 # Temporary file storage
├── transcriptions/          # Saved transcriptions
├── .env.example            # Environment template
├── test_transcription.py   # Test suite
├── main.py                 # Application entry point
└── requirements.txt        # Dependencies
```

## 🧪 Testing

### Run the Test Suite
```bash
python test_transcription.py
```

The test suite validates:
- Configuration settings
- Transcription method availability
- File validation functionality
- Environment variable setup

### Manual Testing
1. Test with different audio formats
2. Try various file sizes
3. Test drag & drop functionality
4. Verify error handling with invalid files

## 🔧 Troubleshooting

### Common Issues

**"No transcription methods available"**
- Install at least one transcription package
- Set up OpenAI API key if using API method
- Check environment variables

**"File size exceeds limit"**
- Reduce file size or increase `MAX_FILE_SIZE_MB`
- Use audio compression tools

**"OpenAI API key not found"**
- Set `OPENAI_API_KEY` in your `.env` file
- Verify the API key is valid

**Slow transcription**
- Use OpenAI API for faster processing
- Try faster-whisper for local processing
- Use smaller Whisper models (tiny, base)

### Performance Optimization

1. **For Cloud Processing**: Use OpenAI API
2. **For Local Processing**:
   - Use faster-whisper instead of openai-whisper
   - Use GPU if available
   - Choose appropriate model size
3. **For Large Files**: Consider chunking (future enhancement)

## 🔒 Security Considerations

- Files are automatically cleaned up after processing
- Secure filename handling prevents path traversal
- File size and format validation
- Environment-based configuration keeps secrets secure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the Whisper models and API
- The faster-whisper team for performance optimizations
- Flask community for the excellent web framework