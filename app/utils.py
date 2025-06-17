import os
import uuid
import logging
import time
import json
import hashlib
import math
from typing import Tuple, Optional, List, Dict, Any
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

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

# Import audio processing libraries
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logging.warning("Audio processing libraries not available. Install with: pip install librosa soundfile")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("Pydub not available. Install with: pip install pydub")

try:
    import diskcache as dc
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    logging.warning("Diskcache not available. Install with: pip install diskcache")

try:
    from tqdm import tqdm
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    logging.warning("Tqdm not available. Install with: pip install tqdm")

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
MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', '200'))

# Audio processing configuration
CHUNK_SIZE_SECONDS = int(os.environ.get('CHUNK_SIZE_SECONDS', '600'))  # 10 minutes for OpenAI API
CHUNK_OVERLAP_SECONDS = int(os.environ.get('CHUNK_OVERLAP_SECONDS', '5'))
ENABLE_TIMESTAMPS = os.environ.get('ENABLE_TIMESTAMPS', 'true').lower() == 'true'
ENABLE_WORD_TIMESTAMPS = os.environ.get('ENABLE_WORD_TIMESTAMPS', 'false').lower() == 'true'

# OpenAI API specific limits
OPENAI_MAX_FILE_SIZE_MB = 25  # OpenAI API limit
OPENAI_MAX_CHUNK_SIZE_SECONDS = 600  # 10 minutes recommended for quality

# Performance configuration
PARALLEL_PROCESSING = os.environ.get('PARALLEL_PROCESSING', 'true').lower() == 'true'
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '4'))
ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'true').lower() == 'true'
CACHE_DURATION_HOURS = int(os.environ.get('CACHE_DURATION_HOURS', '24'))

# Model selection
AUTO_MODEL_SELECTION = os.environ.get('AUTO_MODEL_SELECTION', 'true').lower() == 'true'
SMALL_FILE_THRESHOLD_MB = int(os.environ.get('SMALL_FILE_THRESHOLD_MB', '50'))
LARGE_FILE_MODEL = os.environ.get('LARGE_FILE_MODEL', 'base')
SMALL_FILE_MODEL = os.environ.get('SMALL_FILE_MODEL', 'small')

# Initialize cache if available
cache = None
if CACHING_AVAILABLE and ENABLE_CACHING:
    try:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache = dc.Cache(cache_dir, size_limit=1024**3)  # 1GB cache limit
    except Exception as e:
        logging.warning(f"Failed to initialize cache: {e}")
        cache = None

# Data structures for enhanced transcription
@dataclass
class WordTimestamp:
    """Represents a word with its timestamp information."""
    word: str
    start: float
    end: float
    confidence: Optional[float] = None

@dataclass
class TranscriptionSegment:
    """Represents a segment of transcription with timestamps."""
    text: str
    start: float
    end: float
    words: Optional[List[WordTimestamp]] = None
    confidence: Optional[float] = None

@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    text: str
    segments: List[TranscriptionSegment]
    language: Optional[str] = None
    duration: Optional[float] = None
    method_used: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    file_size_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        srt_content = []
        for i, segment in enumerate(self.segments, 1):
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{segment.text}\n")
        return "\n".join(srt_content)

    def to_vtt(self) -> str:
        """Convert to WebVTT format."""
        vtt_content = ["WEBVTT\n"]
        for segment in self.segments:
            start_time = self._seconds_to_vtt_time(segment.start)
            end_time = self._seconds_to_vtt_time(segment.end)
            vtt_content.append(f"{start_time} --> {end_time}\n{segment.text}\n")
        return "\n".join(vtt_content)

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

def allowed_file(filename):
    """Check if the file extension is in the allowed list."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds with improved error handling."""
    try:
        # Try pydub first as it's more reliable for various formats
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(file_path)
                duration = len(audio) / 1000.0  # Convert milliseconds to seconds
                logger.debug(f"Audio duration (pydub): {duration:.2f}s")
                return duration
            except Exception as e:
                logger.warning(f"Pydub failed to get duration: {e}")

        # Fallback to librosa with better error handling
        if AUDIO_PROCESSING_AVAILABLE:
            try:
                # Try with different backends
                duration = librosa.get_duration(path=file_path)
                logger.debug(f"Audio duration (librosa): {duration:.2f}s")
                return duration
            except Exception as e:
                logger.warning(f"Librosa failed to get duration: {e}")

                # Try loading the file first to check if it's readable
                try:
                    y, sr = librosa.load(file_path, sr=None, duration=1.0)  # Load just 1 second
                    # If successful, estimate full duration from file size
                    file_size = os.path.getsize(file_path)
                    sample_size = len(y) * 4  # Approximate bytes per sample
                    estimated_duration = file_size / sample_size
                    logger.debug(f"Audio duration (estimated): {estimated_duration:.2f}s")
                    return estimated_duration
                except Exception as e2:
                    logger.warning(f"Librosa file loading also failed: {e2}")

        # Final fallback: estimate based on file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        # More conservative estimate: 1MB ≈ 45 seconds for compressed audio
        estimated_duration = file_size_mb * 45
        logger.warning(f"Using file size estimation for duration: {estimated_duration:.2f}s")
        return estimated_duration

    except Exception as e:
        logger.error(f"Could not determine audio duration: {e}")
        # Return a reasonable default based on file size
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return file_size_mb * 45  # Conservative estimate
        except:
            return 300.0  # 5 minutes default


def get_chunk_size_for_method(method: str, file_size_mb: float) -> int:
    """Determine optimal chunk size based on transcription method and file size."""
    if method == 'openai_api':
        # OpenAI API has a 25MB limit, so we need smaller chunks for large files
        if file_size_mb > 100:
            return 300  # 5 minutes for very large files
        elif file_size_mb > 50:
            return 480  # 8 minutes for large files
        else:
            return OPENAI_MAX_CHUNK_SIZE_SECONDS  # 10 minutes for smaller files
    else:
        # Local methods can handle larger chunks
        return CHUNK_SIZE_SECONDS


def estimate_chunk_file_size(duration_seconds: float, original_file_size_mb: float, original_duration: float) -> float:
    """Estimate the file size of a chunk based on duration."""
    if original_duration > 0:
        size_per_second = original_file_size_mb / original_duration
        return duration_seconds * size_per_second
    else:
        # Fallback estimate: ~0.1MB per second for compressed audio
        return duration_seconds * 0.1


def create_audio_chunks(file_path: str, chunk_folder: str, method: str = 'openai_api') -> List[Tuple[str, float, float]]:
    """
    Split audio file into chunks for processing with method-specific optimization.

    Returns:
        List of tuples (chunk_file_path, start_time, end_time)
    """
    if not AUDIO_PROCESSING_AVAILABLE and not PYDUB_AVAILABLE:
        logger.warning("No audio processing library available. Cannot chunk audio.")
        return [(file_path, 0.0, get_audio_duration(file_path))]

    try:
        duration = get_audio_duration(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Get method-specific chunk size
        chunk_size = get_chunk_size_for_method(method, file_size_mb)

        # For OpenAI API, ensure chunks don't exceed size limit
        if method == 'openai_api':
            # Estimate chunk size and adjust if needed
            estimated_chunk_size = estimate_chunk_file_size(chunk_size, file_size_mb, duration)
            while estimated_chunk_size > OPENAI_MAX_FILE_SIZE_MB and chunk_size > 60:
                chunk_size = int(chunk_size * 0.8)  # Reduce by 20%
                estimated_chunk_size = estimate_chunk_file_size(chunk_size, file_size_mb, duration)

            logger.info(f"Using chunk size: {chunk_size}s (estimated {estimated_chunk_size:.1f}MB per chunk)")

        if duration <= chunk_size and file_size_mb <= OPENAI_MAX_FILE_SIZE_MB:
            # File is small enough, no need to chunk
            logger.info(f"File is small enough ({duration:.1f}s, {file_size_mb:.1f}MB), no chunking needed")
            return [(file_path, 0.0, duration)]

        chunks = []
        chunk_start = 0.0
        chunk_index = 0

        logger.info(f"Splitting audio file into chunks (duration: {duration:.1f}s, size: {file_size_mb:.1f}MB)")

        if PYDUB_AVAILABLE:
            # Use pydub for chunking (more reliable for various formats)
            try:
                audio = AudioSegment.from_file(file_path)

                while chunk_start < duration:
                    chunk_end = min(chunk_start + chunk_size, duration)

                    # Extract chunk
                    start_ms = int(chunk_start * 1000)
                    end_ms = int(chunk_end * 1000)
                    chunk_audio = audio[start_ms:end_ms]

                    # Save chunk as MP3 to reduce file size for OpenAI API
                    if method == 'openai_api':
                        chunk_filename = f"chunk_{chunk_index:03d}.mp3"
                        chunk_path = os.path.join(chunk_folder, chunk_filename)
                        chunk_audio.export(chunk_path, format="mp3", bitrate="128k")
                    else:
                        chunk_filename = f"chunk_{chunk_index:03d}.wav"
                        chunk_path = os.path.join(chunk_folder, chunk_filename)
                        chunk_audio.export(chunk_path, format="wav")

                    # Verify chunk size for OpenAI API
                    if method == 'openai_api':
                        chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                        if chunk_size_mb > OPENAI_MAX_FILE_SIZE_MB:
                            logger.warning(f"Chunk {chunk_index} is {chunk_size_mb:.1f}MB, exceeds OpenAI limit")
                            # Try to re-export with lower quality
                            chunk_audio.export(chunk_path, format="mp3", bitrate="64k")
                            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                            logger.info(f"Re-exported chunk {chunk_index} as {chunk_size_mb:.1f}MB")

                    chunks.append((chunk_path, chunk_start, chunk_end))

                    # Move to next chunk with overlap
                    chunk_start += chunk_size - CHUNK_OVERLAP_SECONDS
                    chunk_index += 1

            except Exception as e:
                logger.error(f"Pydub chunking failed: {e}")
                raise

        elif AUDIO_PROCESSING_AVAILABLE:
            # Use librosa for chunking
            try:
                y, sr = librosa.load(file_path, sr=None)

                while chunk_start < duration:
                    chunk_end = min(chunk_start + chunk_size, duration)

                    # Extract chunk
                    start_sample = int(chunk_start * sr)
                    end_sample = int(chunk_end * sr)
                    chunk_audio = y[start_sample:end_sample]

                    # Save chunk
                    chunk_filename = f"chunk_{chunk_index:03d}.wav"
                    chunk_path = os.path.join(chunk_folder, chunk_filename)
                    sf.write(chunk_path, chunk_audio, sr)

                    chunks.append((chunk_path, chunk_start, chunk_end))

                    # Move to next chunk with overlap
                    chunk_start += chunk_size - CHUNK_OVERLAP_SECONDS
                    chunk_index += 1

            except Exception as e:
                logger.error(f"Librosa chunking failed: {e}")
                raise

        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks

    except Exception as e:
        logger.error(f"Failed to create audio chunks: {e}")
        # Fallback to processing the whole file if it's small enough
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if method == 'openai_api' and file_size_mb > OPENAI_MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large ({file_size_mb:.1f}MB) for OpenAI API and chunking failed")
        return [(file_path, 0.0, get_audio_duration(file_path))]

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

def transcribe_with_openai_api(file_path: str, enable_timestamps: bool = True) -> TranscriptionResult:
    """Transcribe audio using OpenAI's Whisper API with enhanced features and size validation."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not available. Install with: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    # Validate file size before attempting transcription
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > OPENAI_MAX_FILE_SIZE_MB:
        raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds OpenAI API limit ({OPENAI_MAX_FILE_SIZE_MB}MB). Use chunking instead.")

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        logger.info(f"Transcribing file with OpenAI API: {file_path} ({file_size_mb:.1f}MB)")
        start_time = time.time()

        # Get file info
        duration = get_audio_duration(file_path)

        with open(file_path, "rb") as audio_file:
            if enable_timestamps and ENABLE_TIMESTAMPS:
                # Request detailed response with timestamps
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"] + (["word"] if ENABLE_WORD_TIMESTAMPS else [])
                )

                # Parse response into our format
                segments = []
                for segment_data in response.segments:
                    words = []
                    if hasattr(segment_data, 'words') and segment_data.words:
                        words = [
                            WordTimestamp(
                                word=word.word,
                                start=word.start,
                                end=word.end
                            ) for word in segment_data.words
                        ]

                    segments.append(TranscriptionSegment(
                        text=segment_data.text,
                        start=segment_data.start,
                        end=segment_data.end,
                        words=words if words else None
                    ))

                result = TranscriptionResult(
                    text=response.text,
                    segments=segments,
                    language=response.language,
                    duration=duration,
                    method_used="openai_api",
                    model_used="whisper-1",
                    processing_time=time.time() - start_time,
                    file_size_mb=file_size_mb
                )
            else:
                # Simple text response
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )

                # Create a single segment for the entire transcription
                segments = [TranscriptionSegment(
                    text=transcript,
                    start=0.0,
                    end=duration
                )]

                result = TranscriptionResult(
                    text=transcript,
                    segments=segments,
                    duration=duration,
                    method_used="openai_api",
                    model_used="whisper-1",
                    processing_time=time.time() - start_time,
                    file_size_mb=file_size_mb
                )

        logger.info(f"OpenAI API transcription completed in {result.processing_time:.2f}s")
        return result

    except Exception as e:
        error_msg = str(e)
        if "413" in error_msg or "Maximum content size limit" in error_msg:
            logger.error(f"OpenAI API file size limit exceeded: {error_msg}")
            raise ValueError(f"File too large for OpenAI API. File size: {file_size_mb:.1f}MB, API limit: {OPENAI_MAX_FILE_SIZE_MB}MB")
        else:
            logger.error(f"OpenAI API transcription failed: {error_msg}")
            raise


def transcribe_with_faster_whisper(file_path: str, model_name: str = None) -> TranscriptionResult:
    """Transcribe audio using faster-whisper (local processing) with enhanced features."""
    if not FASTER_WHISPER_AVAILABLE:
        raise ImportError("Faster-whisper package not available. Install with: pip install faster-whisper")

    try:
        model_to_use = model_name or WHISPER_MODEL
        logger.info(f"Loading faster-whisper model: {model_to_use}")

        # Optimize model settings based on file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 100:  # Large file
            model = WhisperModel(model_to_use, device="cpu", compute_type="int8")
        else:  # Smaller file, can use better quality
            model = WhisperModel(model_to_use, device="cpu", compute_type="float16")

        logger.info(f"Transcribing file with faster-whisper: {file_path}")
        start_time = time.time()
        duration = get_audio_duration(file_path)

        # Configure transcription options
        transcribe_options = {
            "beam_size": 5,
            "language": None,  # Auto-detect
            "condition_on_previous_text": False,
            "word_timestamps": ENABLE_WORD_TIMESTAMPS
        }

        segments_iter, info = model.transcribe(file_path, **transcribe_options)

        # Process segments
        segments = []
        full_text_parts = []

        for segment in segments_iter:
            words = []
            if ENABLE_WORD_TIMESTAMPS and hasattr(segment, 'words') and segment.words:
                words = [
                    WordTimestamp(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        confidence=getattr(word, 'probability', None)
                    ) for word in segment.words
                ]

            segments.append(TranscriptionSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                words=words if words else None,
                confidence=getattr(segment, 'avg_logprob', None)
            ))

            full_text_parts.append(segment.text)

        full_text = " ".join(full_text_parts)
        processing_time = time.time() - start_time

        result = TranscriptionResult(
            text=full_text,
            segments=segments,
            language=info.language,
            duration=duration,
            method_used="faster_whisper",
            model_used=model_to_use,
            processing_time=processing_time,
            file_size_mb=file_size_mb
        )

        logger.info(f"Faster-whisper transcription completed in {processing_time:.2f}s. Language: {info.language}")
        return result

    except Exception as e:
        logger.error(f"Faster-whisper transcription failed: {str(e)}")
        raise


def transcribe_with_openai_whisper(file_path: str) -> TranscriptionResult:
    """Transcribe audio using openai-whisper (local processing)."""
    # Currently disabled due to installation issues
    raise ImportError("OpenAI-whisper package is currently disabled due to installation issues. Use OpenAI API or faster-whisper instead.")


def get_cache_key(file_path: str, method: str, model: str) -> str:
    """Generate a cache key for transcription results."""
    # Create hash based on file content and parameters
    hasher = hashlib.md5()

    # Add file content hash
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    # Add parameters
    hasher.update(f"{method}_{model}_{ENABLE_TIMESTAMPS}_{ENABLE_WORD_TIMESTAMPS}".encode())

    return hasher.hexdigest()


def get_optimal_model(file_size_mb: float) -> str:
    """Select optimal model based on file size and configuration."""
    if not AUTO_MODEL_SELECTION:
        return WHISPER_MODEL

    if file_size_mb <= SMALL_FILE_THRESHOLD_MB:
        return SMALL_FILE_MODEL
    else:
        return LARGE_FILE_MODEL


def transcribe_chunk(chunk_info: Tuple[str, float, float], method: str, model: str, retry_count: int = 0) -> TranscriptionResult:
    """Transcribe a single audio chunk with retry logic."""
    chunk_path, start_offset, end_offset = chunk_info
    max_retries = 2

    try:
        # Validate chunk file size for OpenAI API
        if method == 'openai_api':
            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            if chunk_size_mb > OPENAI_MAX_FILE_SIZE_MB:
                raise ValueError(f"Chunk file size ({chunk_size_mb:.1f}MB) exceeds OpenAI API limit ({OPENAI_MAX_FILE_SIZE_MB}MB)")

        logger.debug(f"Transcribing chunk: {chunk_path} ({start_offset:.1f}s - {end_offset:.1f}s)")

        if method == 'openai_api':
            result = transcribe_with_openai_api(chunk_path, enable_timestamps=ENABLE_TIMESTAMPS)
        elif method == 'faster_whisper':
            result = transcribe_with_faster_whisper(chunk_path, model_name=model)
        else:
            raise ValueError(f"Unsupported transcription method: {method}")

        # Adjust timestamps to account for chunk offset
        if result.segments:
            for segment in result.segments:
                segment.start += start_offset
                segment.end += start_offset
                if segment.words:
                    for word in segment.words:
                        word.start += start_offset
                        word.end += start_offset

        logger.debug(f"Successfully transcribed chunk: {chunk_path}")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to transcribe chunk {chunk_path}: {error_msg}")

        # Retry logic for certain errors
        if retry_count < max_retries:
            if "413" in error_msg or "Maximum content size limit" in error_msg:
                logger.info(f"Chunk too large, attempting to re-create with smaller size (retry {retry_count + 1})")
                # Try to recreate chunk with smaller duration if possible
                # This would require additional logic to re-chunk the specific segment
                pass
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                logger.info(f"Network error, retrying chunk {chunk_path} (retry {retry_count + 1})")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return transcribe_chunk(chunk_info, method, model, retry_count + 1)

        raise


def transcribe_audio_enhanced(file_path: str, transcription_folder: str, chunks_folder: str) -> TranscriptionResult:
    """
    Enhanced transcription with chunking, timestamps, and parallel processing.

    Returns:
        TranscriptionResult with detailed information
    """
    start_time = time.time()
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # Validate file size
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)")

    # Check cache first
    cache_key = None
    if cache and ENABLE_CACHING:
        optimal_model = get_optimal_model(file_size_mb)
        cache_key = get_cache_key(file_path, TRANSCRIPTION_METHOD, optimal_model)

        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Using cached transcription result")
            return TranscriptionResult(**cached_result)

    # Determine transcription method
    methods = [
        ('openai_api', transcribe_with_openai_api),
        ('faster_whisper', transcribe_with_faster_whisper),
        ('openai_whisper', transcribe_with_openai_whisper)
    ]

    # Filter available methods
    available_methods = []
    for method_name, method_func in methods:
        if method_name == 'openai_api' and OPENAI_AVAILABLE and OPENAI_API_KEY:
            available_methods.append((method_name, method_func))
        elif method_name == 'faster_whisper' and FASTER_WHISPER_AVAILABLE:
            available_methods.append((method_name, method_func))
        elif method_name == 'openai_whisper' and WHISPER_AVAILABLE:
            available_methods.append((method_name, method_func))

    if not available_methods:
        raise RuntimeError("No transcription methods available. Please check your configuration and dependencies.")

    # Use preferred method if specified
    if TRANSCRIPTION_METHOD != 'auto':
        preferred_method = next((m for m in available_methods if m[0] == TRANSCRIPTION_METHOD), None)
        if preferred_method:
            available_methods = [preferred_method]

    # Get optimal model
    optimal_model = get_optimal_model(file_size_mb)

    result = None
    for method_name, method_func in available_methods:
        try:
            logger.info(f"Attempting transcription with method: {method_name}, model: {optimal_model}")

            # Determine if we need chunking
            duration = get_audio_duration(file_path)

            # For OpenAI API, check both duration and file size limits
            if method_name == 'openai_api':
                needs_chunking = (
                    file_size_mb > OPENAI_MAX_FILE_SIZE_MB or  # Exceeds API size limit
                    duration > OPENAI_MAX_CHUNK_SIZE_SECONDS  # Exceeds recommended duration
                )
            else:
                # For local methods, only chunk very large files
                needs_chunking = (
                    duration > CHUNK_SIZE_SECONDS * 2 or  # File is significantly longer than chunk size
                    file_size_mb > 200  # Very large file that might cause memory issues
                )

            if needs_chunking and (AUDIO_PROCESSING_AVAILABLE or PYDUB_AVAILABLE):
                logger.info(f"File duration: {duration:.1f}s, size: {file_size_mb:.1f}MB - using chunked processing")
                result = transcribe_with_chunking(file_path, chunks_folder, method_name, optimal_model)
            else:
                logger.info(f"Using direct transcription (duration: {duration:.1f}s, size: {file_size_mb:.1f}MB)")
                if method_name == 'openai_api':
                    result = transcribe_with_openai_api(file_path, enable_timestamps=ENABLE_TIMESTAMPS)
                elif method_name == 'faster_whisper':
                    result = transcribe_with_faster_whisper(file_path, model_name=optimal_model)
                else:
                    result = method_func(file_path)

            break

        except Exception as e:
            logger.warning(f"Transcription method {method_name} failed: {str(e)}")
            continue

    if result is None:
        raise RuntimeError("All transcription methods failed. Please check your configuration and dependencies.")

    # Update processing time
    result.processing_time = time.time() - start_time

    # Cache the result
    if cache and ENABLE_CACHING and cache_key:
        try:
            cache.set(cache_key, result.to_dict(), expire=CACHE_DURATION_HOURS * 3600)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    logger.info(f"Transcription completed in {result.processing_time:.2f}s using {result.method_used}")
    return result


def transcribe_with_chunking(file_path: str, chunks_folder: str, method: str, model: str) -> TranscriptionResult:
    """Transcribe audio file using chunking for large files."""
    start_time = time.time()

    # Create chunks with method-specific optimization
    chunks = create_audio_chunks(file_path, chunks_folder, method)
    logger.info(f"Processing {len(chunks)} chunks using {method}")

    all_segments = []
    all_text_parts = []

    if PARALLEL_PROCESSING and len(chunks) > 1:
        # Parallel processing
        logger.info(f"Using parallel processing with {min(MAX_WORKERS, len(chunks))} workers")

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as executor:
            # Submit all chunk transcription tasks
            future_to_chunk = {
                executor.submit(transcribe_chunk, chunk_info, method, model): chunk_info
                for chunk_info in chunks
            }

            # Process completed tasks
            if PROGRESS_AVAILABLE:
                futures = list(future_to_chunk.keys())
                for future in tqdm(as_completed(futures), total=len(futures), desc="Transcribing chunks"):
                    chunk_info = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        all_segments.extend(chunk_result.segments)
                        all_text_parts.append(chunk_result.text)
                        logger.debug(f"Completed chunk {chunk_info[0]} ({chunk_info[1]:.1f}s - {chunk_info[2]:.1f}s)")
                    except Exception as e:
                        logger.error(f"Chunk {chunk_info[0]} failed: {e}")
                        # Add placeholder text for failed chunk to maintain timing
                        chunk_duration = chunk_info[2] - chunk_info[1]
                        placeholder_text = f"[TRANSCRIPTION FAILED FOR {chunk_duration:.1f}s SEGMENT]"
                        all_text_parts.append(placeholder_text)

                        # Add placeholder segment
                        placeholder_segment = TranscriptionSegment(
                            text=placeholder_text,
                            start=chunk_info[1],
                            end=chunk_info[2]
                        )
                        all_segments.append(placeholder_segment)
            else:
                for future in as_completed(future_to_chunk):
                    chunk_info = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        all_segments.extend(chunk_result.segments)
                        all_text_parts.append(chunk_result.text)
                        logger.debug(f"Completed chunk {chunk_info[0]} ({chunk_info[1]:.1f}s - {chunk_info[2]:.1f}s)")
                    except Exception as e:
                        logger.error(f"Chunk {chunk_info[0]} failed: {e}")
                        # Add placeholder text for failed chunk to maintain timing
                        chunk_duration = chunk_info[2] - chunk_info[1]
                        placeholder_text = f"[TRANSCRIPTION FAILED FOR {chunk_duration:.1f}s SEGMENT]"
                        all_text_parts.append(placeholder_text)

                        # Add placeholder segment
                        placeholder_segment = TranscriptionSegment(
                            text=placeholder_text,
                            start=chunk_info[1],
                            end=chunk_info[2]
                        )
                        all_segments.append(placeholder_segment)
    else:
        # Sequential processing
        logger.info("Using sequential processing")

        chunk_iterator = tqdm(chunks, desc="Transcribing chunks") if PROGRESS_AVAILABLE else chunks

        for chunk_info in chunk_iterator:
            try:
                chunk_result = transcribe_chunk(chunk_info, method, model)
                all_segments.extend(chunk_result.segments)
                all_text_parts.append(chunk_result.text)
                logger.debug(f"Completed chunk {chunk_info[0]} ({chunk_info[1]:.1f}s - {chunk_info[2]:.1f}s)")
            except Exception as e:
                logger.error(f"Chunk {chunk_info[0]} failed: {e}")
                # Add placeholder text for failed chunk to maintain timing
                chunk_duration = chunk_info[2] - chunk_info[1]
                placeholder_text = f"[TRANSCRIPTION FAILED FOR {chunk_duration:.1f}s SEGMENT]"
                all_text_parts.append(placeholder_text)

                # Add placeholder segment
                placeholder_segment = TranscriptionSegment(
                    text=placeholder_text,
                    start=chunk_info[1],
                    end=chunk_info[2]
                )
                all_segments.append(placeholder_segment)

    # Clean up chunk files
    for chunk_path, _, _ in chunks:
        try:
            if chunk_path != file_path and os.path.exists(chunk_path):
                os.remove(chunk_path)
        except Exception as e:
            logger.warning(f"Failed to clean up chunk file {chunk_path}: {e}")

    # Sort segments by start time
    all_segments.sort(key=lambda s: s.start)

    # Combine text
    full_text = " ".join(all_text_parts)

    # Get file info
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    duration = get_audio_duration(file_path)

    # Create final result
    result = TranscriptionResult(
        text=full_text,
        segments=all_segments,
        duration=duration,
        method_used=method,
        model_used=model,
        processing_time=time.time() - start_time,
        file_size_mb=file_size_mb
    )

    return result


def transcribe_audio(file_path: str, transcription_folder: str) -> Tuple[str, str, str]:
    """
    Backward-compatible transcription function.

    Returns:
        Tuple of (transcription_text, transcription_path, transcription_filename)
    """
    # Create chunks folder
    chunks_folder = os.path.join(os.path.dirname(transcription_folder), 'chunks')
    os.makedirs(chunks_folder, exist_ok=True)

    try:
        # Use enhanced transcription
        result = transcribe_audio_enhanced(file_path, transcription_folder, chunks_folder)

        # Save results in multiple formats
        base_filename = os.path.basename(file_path)
        base_name = os.path.splitext(base_filename)[0]

        # Save basic text transcription
        transcription_filename = f"{base_name}.txt"
        transcription_path = os.path.join(transcription_folder, transcription_filename)

        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(f"# Transcribed using: {result.method_used}\n")
            f.write(f"# Model: {result.model_used}\n")
            f.write(f"# Original file: {base_filename}\n")
            f.write(f"# Duration: {result.duration:.2f}s\n")
            f.write(f"# Processing time: {result.processing_time:.2f}s\n")
            f.write(f"# Language: {result.language}\n")
            f.write(f"# Transcription date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(result.text)

        # Save enhanced formats if timestamps are available
        if result.segments and len(result.segments) > 0:
            try:
                # Save JSON with full details
                json_filename = f"{base_name}_detailed.json"
                json_path = os.path.join(transcription_folder, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"Saved JSON file: {json_path}")

                # Save SRT subtitle format
                srt_filename = f"{base_name}.srt"
                srt_path = os.path.join(transcription_folder, srt_filename)
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(result.to_srt())
                logger.info(f"Saved SRT file: {srt_path}")

                # Save VTT subtitle format
                vtt_filename = f"{base_name}.vtt"
                vtt_path = os.path.join(transcription_folder, vtt_filename)
                with open(vtt_path, 'w', encoding='utf-8') as f:
                    f.write(result.to_vtt())
                logger.info(f"Saved VTT file: {vtt_path}")

            except Exception as e:
                logger.error(f"Failed to save enhanced formats: {e}")
        else:
            logger.info("No segments available, skipping enhanced format export")

        logger.info(f"Transcription saved to: {transcription_path}")
        return result.text, transcription_path, transcription_filename

    except Exception as e:
        logger.error(f"Enhanced transcription failed: {e}")
        raise


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
