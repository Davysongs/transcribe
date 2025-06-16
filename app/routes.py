import os
import json
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, current_app, jsonify
from werkzeug.utils import secure_filename
from app.utils import (
    save_upload,
    transcribe_audio,
    transcribe_audio_enhanced,
    allowed_file,
    validate_audio_file,
    get_transcription_methods_status,
    get_audio_duration,
    MAX_FILE_SIZE_MB
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


@main_bp.route('/api/file-info', methods=['POST'])
def get_file_info():
    """Get information about an uploaded file."""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save file temporarily to get info
        temp_path, _ = save_upload(file, current_app.config['UPLOAD_FOLDER'])
        if temp_path is None:
            return jsonify({'error': 'Failed to save file'}), 500

        try:
            # Get file information
            file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            duration = get_audio_duration(temp_path)

            # Estimate processing time (rough estimate)
            estimated_time = max(duration * 0.1, 5)  # At least 5 seconds, roughly 10% of audio duration

            # Determine if chunking will be used
            will_chunk = duration > 60 or file_size_mb > 100
            estimated_chunks = max(1, int(duration / 30)) if will_chunk else 1

            info = {
                'file_size_mb': round(file_size_mb, 2),
                'duration_seconds': round(duration, 1),
                'estimated_processing_time': round(estimated_time, 1),
                'will_use_chunking': will_chunk,
                'estimated_chunks': estimated_chunks,
                'max_file_size_mb': MAX_FILE_SIZE_MB
            }

            return jsonify(info)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@main_bp.route('/enhanced-upload', methods=['POST'])
def enhanced_upload():
    """Handle enhanced file upload with detailed transcription."""
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

        # Enhanced transcription
        try:
            result = transcribe_audio_enhanced(
                file_path,
                current_app.config['TRANSCRIPTION_FOLDER'],
                current_app.config['CHUNKS_FOLDER']
            )
            logger.info(f"Enhanced transcription completed successfully for {original_filename}")
        except Exception as transcription_error:
            logger.error(f"Enhanced transcription failed: {transcription_error}")
            # Fallback to basic transcription
            logger.info("Falling back to basic transcription")
            text, transcription_path, transcription_filename = transcribe_audio(
                file_path,
                current_app.config['TRANSCRIPTION_FOLDER']
            )

            # Create a basic result object for compatibility
            from app.utils import TranscriptionResult, TranscriptionSegment
            result = TranscriptionResult(
                text=text,
                segments=[TranscriptionSegment(text=text, start=0.0, end=0.0)],
                method_used="fallback",
                model_used="unknown",
                processing_time=0.0,
                file_size_mb=os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0.0
            )

        # Clean up uploaded file after successful transcription
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up uploaded file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up uploaded file: {cleanup_error}")

        # Render enhanced results page
        return render_template(
            'enhanced_result.html',
            result=result,
            filename=original_filename
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


@main_bp.route('/download-enhanced/<path:filename>/<format>')
def download_enhanced_file(filename, format):
    """Download enhanced transcription files in various formats."""
    try:
        # Security: ensure filename is safe
        safe_filename = secure_filename(filename)
        base_name = os.path.splitext(safe_filename)[0]

        logger.info(f"Download request for file: {safe_filename}, format: {format}")
        logger.info(f"Base name: {base_name}")
        logger.info(f"Transcription folder: {current_app.config['TRANSCRIPTION_FOLDER']}")

        # Determine file path based on format
        if format == 'txt':
            file_path = os.path.join(current_app.config['TRANSCRIPTION_FOLDER'], f"{base_name}.txt")
        elif format == 'json':
            file_path = os.path.join(current_app.config['TRANSCRIPTION_FOLDER'], f"{base_name}_detailed.json")
        elif format == 'srt':
            file_path = os.path.join(current_app.config['TRANSCRIPTION_FOLDER'], f"{base_name}.srt")
        elif format == 'vtt':
            file_path = os.path.join(current_app.config['TRANSCRIPTION_FOLDER'], f"{base_name}.vtt")
        else:
            logger.error(f"Invalid format requested: {format}")
            flash('Invalid format requested', 'error')
            return redirect(url_for('main.index'))

        logger.info(f"Looking for file at: {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            # List available files for debugging
            transcription_folder = current_app.config['TRANSCRIPTION_FOLDER']
            available_files = []
            if os.path.exists(transcription_folder):
                available_files = os.listdir(transcription_folder)

            logger.error(f"File not found: {file_path}")
            logger.error(f"Available files in transcription folder: {available_files}")

            flash(f'File not found: {os.path.basename(file_path)}. Available files: {", ".join(available_files) if available_files else "None"}', 'error')
            return redirect(url_for('main.index'))

        logger.info(f"Sending file: {file_path}")
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        logger.error(f"Error downloading enhanced file: {str(e)}")
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('main.index'))

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


@main_bp.route('/debug/files')
def debug_files():
    """Debug route to show available files."""
    try:
        transcription_folder = current_app.config['TRANSCRIPTION_FOLDER']

        if not os.path.exists(transcription_folder):
            return jsonify({
                'error': 'Transcription folder does not exist',
                'folder': transcription_folder
            })

        files = os.listdir(transcription_folder)
        file_details = []

        for file in files:
            file_path = os.path.join(transcription_folder, file)
            file_details.append({
                'name': file,
                'size': os.path.getsize(file_path),
                'modified': os.path.getmtime(file_path)
            })

        return jsonify({
            'folder': transcription_folder,
            'files': file_details,
            'count': len(files)
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@main_bp.route('/test-result')
def test_result():
    """Test route to verify the enhanced result template works."""
    try:
        from app.utils import TranscriptionResult, TranscriptionSegment, WordTimestamp

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

        return render_template(
            'enhanced_result.html',
            result=result,
            filename="test_file.wav"
        )

    except Exception as e:
        logger.error(f"Error in test result: {str(e)}")
        return f"Error: {str(e)}"


@main_bp.route('/files')
def list_files():
    """List all available transcription files for download."""
    try:
        transcription_folder = current_app.config['TRANSCRIPTION_FOLDER']

        if not os.path.exists(transcription_folder):
            flash('No transcription files found.', 'info')
            return redirect(url_for('main.index'))

        files = []
        for filename in os.listdir(transcription_folder):
            file_path = os.path.join(transcription_folder, filename)
            if os.path.isfile(file_path):
                file_info = {
                    'name': filename,
                    'size': os.path.getsize(file_path),
                    'modified': os.path.getmtime(file_path),
                    'extension': os.path.splitext(filename)[1].lower()
                }
                files.append(file_info)

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)

        return render_template('files_list.html', files=files)

    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        flash(f'Error listing files: {str(e)}', 'error')
        return redirect(url_for('main.index'))


@main_bp.route('/download-direct/<path:filename>')
def download_direct(filename):
    """Direct download of any file in the transcriptions folder."""
    try:
        safe_filename = secure_filename(filename)
        file_path = os.path.join(current_app.config['TRANSCRIPTION_FOLDER'], safe_filename)

        if not os.path.exists(file_path):
            flash(f'File not found: {safe_filename}', 'error')
            return redirect(url_for('main.list_files'))

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        logger.error(f"Error downloading file directly: {str(e)}")
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('main.list_files'))
