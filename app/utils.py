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
CHUNK_SIZE_SECONDS = int(os.environ.get('CHUNK_SIZE_SECONDS', '30'))
CHUNK_OVERLAP_SECONDS = int(os.environ.get('CHUNK_OVERLAP_SECONDS', '2'))
ENABLE_TIMESTAMPS = os.environ.get('ENABLE_TIMESTAMPS', 'true').lower() == 'true'
ENABLE_WORD_TIMESTAMPS = os.environ.get('ENABLE_WORD_TIMESTAMPS', 'false').lower() == 'true'

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
    """Get the duration of an audio file in seconds."""
    try:
        if AUDIO_PROCESSING_AVAILABLE:
            duration = librosa.get_duration(path=file_path)
            return duration
        elif PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        else:
            # Fallback: estimate based on file size (very rough)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            # Rough estimate: 1MB ≈ 1 minute for compressed audio
            return file_size_mb * 60
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


def create_audio_chunks(file_path: str, chunk_folder: str) -> List[Tuple[str, float, float]]:
    """
    Split audio file into chunks for processing.

    Returns:
        List of tuples (chunk_file_path, start_time, end_time)
    """
    if not AUDIO_PROCESSING_AVAILABLE and not PYDUB_AVAILABLE:
        logger.warning("No audio processing library available. Cannot chunk audio.")
        return [(file_path, 0.0, get_audio_duration(file_path))]

    try:
        duration = get_audio_duration(file_path)
        if duration <= CHUNK_SIZE_SECONDS:
            # File is small enough, no need to chunk
            return [(file_path, 0.0, duration)]

        chunks = []
        chunk_start = 0.0
        chunk_index = 0

        logger.info(f"Splitting audio file into chunks (duration: {duration:.1f}s)")

        if PYDUB_AVAILABLE:
            # Use pydub for chunking (more reliable for various formats)
            audio = AudioSegment.from_file(file_path)

            while chunk_start < duration:
                chunk_end = min(chunk_start + CHUNK_SIZE_SECONDS, duration)

                # Extract chunk
                start_ms = int(chunk_start * 1000)
                end_ms = int(chunk_end * 1000)
                chunk_audio = audio[start_ms:end_ms]

                # Save chunk
                chunk_filename = f"chunk_{chunk_index:03d}.wav"
                chunk_path = os.path.join(chunk_folder, chunk_filename)
                chunk_audio.export(chunk_path, format="wav")

                chunks.append((chunk_path, chunk_start, chunk_end))

                # Move to next chunk with overlap
                chunk_start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
                chunk_index += 1

        elif AUDIO_PROCESSING_AVAILABLE:
            # Use librosa for chunking
            y, sr = librosa.load(file_path, sr=None)

            while chunk_start < duration:
                chunk_end = min(chunk_start + CHUNK_SIZE_SECONDS, duration)

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
                chunk_start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
                chunk_index += 1

        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks

    except Exception as e:
        logger.error(f"Failed to create audio chunks: {e}")
        # Fallback to processing the whole file
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
    """Transcribe audio using OpenAI's Whisper API with enhanced features."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not available. Install with: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        logger.info(f"Transcribing file with OpenAI API: {file_path}")
        start_time = time.time()

        # Get file info
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
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
        logger.error(f"OpenAI API transcription failed: {str(e)}")
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


def transcribe_chunk(chunk_info: Tuple[str, float, float], method: str, model: str) -> TranscriptionResult:
    """Transcribe a single audio chunk."""
    chunk_path, start_offset, end_offset = chunk_info

    try:
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

        return result

    except Exception as e:
        logger.error(f"Failed to transcribe chunk {chunk_path}: {e}")
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
            needs_chunking = (
                duration > CHUNK_SIZE_SECONDS * 2 or  # File is significantly longer than chunk size
                file_size_mb > 100  # Large file that might cause memory issues
            )

            if needs_chunking and (AUDIO_PROCESSING_AVAILABLE or PYDUB_AVAILABLE):
                logger.info(f"File duration: {duration:.1f}s, using chunked processing")
                result = transcribe_with_chunking(file_path, chunks_folder, method_name, optimal_model)
            else:
                logger.info("Using direct transcription (no chunking needed)")
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

    # Create chunks
    chunks = create_audio_chunks(file_path, chunks_folder)
    logger.info(f"Processing {len(chunks)} chunks")

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
                    except Exception as e:
                        logger.error(f"Chunk {chunk_info[0]} failed: {e}")
                        # Continue with other chunks
            else:
                for future in as_completed(future_to_chunk):
                    chunk_info = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        all_segments.extend(chunk_result.segments)
                        all_text_parts.append(chunk_result.text)
                    except Exception as e:
                        logger.error(f"Chunk {chunk_info[0]} failed: {e}")
                        # Continue with other chunks
    else:
        # Sequential processing
        logger.info("Using sequential processing")

        chunk_iterator = tqdm(chunks, desc="Transcribing chunks") if PROGRESS_AVAILABLE else chunks

        for chunk_info in chunk_iterator:
            try:
                chunk_result = transcribe_chunk(chunk_info, method, model)
                all_segments.extend(chunk_result.segments)
                all_text_parts.append(chunk_result.text)
            except Exception as e:
                logger.error(f"Chunk {chunk_info[0]} failed: {e}")
                # Continue with other chunks

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
