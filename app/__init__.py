import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['TRANSCRIPTION_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'transcriptions')

    # Set max upload size based on environment variable (default 200MB)
    max_file_size_mb = int(os.environ.get('MAX_FILE_SIZE_MB', '200'))
    app.config['MAX_CONTENT_LENGTH'] = max_file_size_mb * 1024 * 1024

    # Add cache folder for processed chunks
    app.config['CACHE_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')

    # Add chunks folder for temporary audio segments
    app.config['CHUNKS_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chunks')

    # Make sure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TRANSCRIPTION_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CHUNKS_FOLDER'], exist_ok=True)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
