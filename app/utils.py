import os
import uuid
import logging
import time
from typing import Tuple, Optional
from werkzeug.utils import secure_filename

# Import transcription libraries with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available. Install with: pip install openai")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("Faster-whisper package not available. Install with: pip install faster-whisper")

# For now, disable openai-whisper due to installation issues
# This can be re-enabled once the package installation is fixed
WHISPER_AVAILABLE = False
logging.warning("OpenAI-whisper package disabled due to installation issues. Use OpenAI API or faster-whisper instead.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm', 'mp4'}

# Configuration
TRANSCRIPTION_METHOD = os.environ.get('TRANSCRIPTION_METHOD', 'openai_api')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', '25'))

def allowed_file(filename):
    """Check if the file extension is in the allowed list."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_upload(file, upload_folder):
    """Save an uploaded file with a secure filename."""
    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid collisions
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path, original_filename
    return None, None

def transcribe_with_openai_api(file_path: str) -> str:
    """Transcribe audio using OpenAI's Whisper API."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not available. Install with: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        logger.info(f"Transcribing file with OpenAI API: {file_path}")

        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        logger.info("OpenAI API transcription completed successfully")
        return transcript

    except Exception as e:
        logger.error(f"OpenAI API transcription failed: {str(e)}")
        raise


def transcribe_with_faster_whisper(file_path: str) -> str:
    """Transcribe audio using faster-whisper (local processing)."""
    if not FASTER_WHISPER_AVAILABLE:
        raise ImportError("Faster-whisper package not available. Install with: pip install faster-whisper")

    try:
        logger.info(f"Loading faster-whisper model: {WHISPER_MODEL}")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

        logger.info(f"Transcribing file with faster-whisper: {file_path}")
        segments, info = model.transcribe(file_path, beam_size=5)

        # Combine all segments into a single text
        transcription = " ".join([segment.text for segment in segments])

        logger.info(f"Faster-whisper transcription completed. Language: {info.language}")
        return transcription

    except Exception as e:
        logger.error(f"Faster-whisper transcription failed: {str(e)}")
        raise


def transcribe_with_openai_whisper(file_path: str) -> str:
    """Transcribe audio using openai-whisper (local processing)."""
    # Currently disabled due to installation issues
    raise ImportError("OpenAI-whisper package is currently disabled due to installation issues. Use OpenAI API or faster-whisper instead.")


def transcribe_audio(file_path: str, transcription_folder: str) -> Tuple[str, str, str]:
    """
    Transcribe audio file using the configured method with fallbacks.

    Returns:
        Tuple of (transcription_text, transcription_path, transcription_filename)
    """
    # Validate file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)")

    transcription_text = None
    method_used = None

    # Try transcription methods in order of preference
    methods = [
        ('openai_api', transcribe_with_openai_api),
        ('faster_whisper', transcribe_with_faster_whisper),
        ('openai_whisper', transcribe_with_openai_whisper)
    ]

    # If a specific method is configured, try it first
    if TRANSCRIPTION_METHOD != 'auto':
        preferred_method = next((m for m in methods if m[0] == TRANSCRIPTION_METHOD), None)
        if preferred_method:
            methods.insert(0, preferred_method)
            methods = [preferred_method] + [m for m in methods if m[0] != TRANSCRIPTION_METHOD]

    for method_name, method_func in methods:
        try:
            logger.info(f"Attempting transcription with method: {method_name}")
            transcription_text = method_func(file_path)
            method_used = method_name
            break
        except Exception as e:
            logger.warning(f"Transcription method {method_name} failed: {str(e)}")
            continue

    if transcription_text is None:
        raise RuntimeError("All transcription methods failed. Please check your configuration and dependencies.")

    # Create a unique filename for the transcription
    base_filename = os.path.basename(file_path)
    transcription_filename = f"{os.path.splitext(base_filename)[0]}.txt"
    transcription_path = os.path.join(transcription_folder, transcription_filename)

    # Save transcription to file
    with open(transcription_path, 'w', encoding='utf-8') as f:
        f.write(f"# Transcribed using: {method_used}\n")
        f.write(f"# Original file: {base_filename}\n")
        f.write(f"# Transcription date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(transcription_text)

    logger.info(f"Transcription saved to: {transcription_path} (method: {method_used})")
    return transcription_text, transcription_path, transcription_filename


def validate_audio_file(file_path: str) -> bool:
    """
    Validate audio file format and size.

    Returns:
        True if file is valid, raises exception otherwise
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)")

    # Check file extension
    file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
    if file_extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")

    return True


def get_transcription_methods_status() -> dict:
    """
    Get the availability status of all transcription methods.

    Returns:
        Dictionary with method availability and configuration
    """
    return {
        'openai_api': {
            'available': OPENAI_AVAILABLE and bool(OPENAI_API_KEY),
            'configured': bool(OPENAI_API_KEY),
            'package_installed': OPENAI_AVAILABLE
        },
        'faster_whisper': {
            'available': FASTER_WHISPER_AVAILABLE,
            'configured': True,
            'package_installed': FASTER_WHISPER_AVAILABLE
        },
        'openai_whisper': {
            'available': WHISPER_AVAILABLE,
            'configured': True,
            'package_installed': WHISPER_AVAILABLE
        },
        'current_method': TRANSCRIPTION_METHOD,
        'whisper_model': WHISPER_MODEL
    }
