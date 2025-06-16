import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, current_app, jsonify
from werkzeug.utils import secure_filename
from app.utils import (
    save_upload,
    transcribe_audio,
    allowed_file,
    validate_audio_file,
    get_transcription_methods_status
)
import logging

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

@main_bp.route('/')
def index():
    """Render the upload form."""
    # Get transcription methods status for display
    methods_status = get_transcription_methods_status()
    return render_template('index.html', methods_status=methods_status)

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and transcription."""
    # Check if file is in request
    if 'audio_file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['audio_file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('main.index'))
    
    # Check if file has allowed extension
    if not allowed_file(file.filename):
        from app.utils import ALLOWED_EXTENSIONS
        flash(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
        return redirect(url_for('main.index'))
    
    try:
        # Save uploaded file
        file_path, original_filename = save_upload(file, current_app.config['UPLOAD_FOLDER'])
        if file_path is None:
            flash('Error saving file', 'error')
            return redirect(url_for('main.index'))

        # Validate audio file
        try:
            validate_audio_file(file_path)
        except (ValueError, FileNotFoundError) as e:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            flash(str(e), 'error')
            return redirect(url_for('main.index'))

        # Transcribe audio
        text, transcription_path, transcription_filename = transcribe_audio(
            file_path,
            current_app.config['TRANSCRIPTION_FOLDER']
        )

        # Clean up uploaded file after successful transcription
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up uploaded file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up uploaded file: {cleanup_error}")

        # Redirect to results page
        return render_template(
            'result.html',
            transcription=text,
            filename=original_filename,
            transcription_filename=transcription_filename
        )

    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        flash('Transcription service not available. Please check server configuration.', 'error')
        return redirect(url_for('main.index'))

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        flash('Transcription service configuration error. Please contact administrator.', 'error')
        return redirect(url_for('main.index'))

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('main.index'))


@main_bp.route('/api/status')
def api_status():
    """Get transcription methods status."""
    try:
        status = get_transcription_methods_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main_bp.route('/download/<path:filename>')
def download_file(filename):
    """Allow downloading the transcription file."""
    try:
        return send_file(
            os.path.join(current_app.config['TRANSCRIPTION_FOLDER'], secure_filename(filename)),
            as_attachment=True
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('main.index'))
