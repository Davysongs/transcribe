import os
import uuid
import whisper
import logging
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg'}

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

def transcribe_audio(file_path, transcription_folder):
    """Transcribe audio file using OpenAI's Whisper model."""
    try:
        logger.info(f"Loading Whisper model...")
        model = whisper.load_model("base")  # Use "small" or "medium" for better accuracy but slower processing
        
        logger.info(f"Transcribing file: {file_path}")
        result = model.transcribe(file_path)
        
        # Create a unique filename for the transcription
        base_filename = os.path.basename(file_path)
        transcription_filename = f"{os.path.splitext(base_filename)[0]}.txt"
        transcription_path = os.path.join(transcription_folder, transcription_filename)
        
        # Save transcription to file
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        logger.info(f"Transcription saved to: {transcription_path}")
        return result['text'], transcription_path, transcription_filename
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise
