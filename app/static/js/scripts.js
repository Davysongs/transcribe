document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const fileInput = document.getElementById('audio_file');
    const fileNameDisplay = document.getElementById('file-name');
    const fileSizeDisplay = document.getElementById('file-size');
    const uploadForm = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn?.querySelector('.btn-text');
    const btnLoading = submitBtn?.querySelector('.btn-loading');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const progressDetail = document.getElementById('progress-detail');
    const fileInfo = document.getElementById('file-info');
    const basicTranscribeLink = document.getElementById('basic-transcribe');

    // Maximum file size in MB (will be updated from server)
    let MAX_FILE_SIZE_MB = 200;

    // Update file name and size display when a file is selected
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);

                fileNameDisplay.textContent = file.name;

                if (fileSizeDisplay) {
                    fileSizeDisplay.textContent = `Size: ${fileSizeMB} MB`;

                    // Check file size
                    if (file.size / (1024 * 1024) > MAX_FILE_SIZE_MB) {
                        fileSizeDisplay.style.color = '#c53030';
                        fileSizeDisplay.textContent += ` (Exceeds ${MAX_FILE_SIZE_MB}MB limit)`;
                        submitBtn.disabled = true;
                    } else {
                        fileSizeDisplay.style.color = '#718096';
                        submitBtn.disabled = false;
                    }
                }

                // Get enhanced file information
                getFileInfo(file);
            } else {
                fileNameDisplay.textContent = 'No file selected';
                if (fileSizeDisplay) {
                    fileSizeDisplay.textContent = '';
                }
                submitBtn.disabled = false;
            }
        });
    }

    // Handle form submission
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener('submit', function(e) {
            // Validate file selection
            if (!fileInput.files.length) {
                e.preventDefault();
                alert('Please select an audio file first.');
                return;
            }

            // Validate file size
            const file = fileInput.files[0];
            if (file.size / (1024 * 1024) > MAX_FILE_SIZE_MB) {
                e.preventDefault();
                alert(`File size exceeds the maximum limit of ${MAX_FILE_SIZE_MB}MB.`);
                return;
            }

            // Show loading state
            submitBtn.disabled = true;
            if (btnText) btnText.style.display = 'none';
            if (btnLoading) btnLoading.style.display = 'inline';

            // Show progress container
            if (progressContainer) {
                progressContainer.style.display = 'block';
                simulateProgress();
            }
        });
    }

    // Simulate progress for user feedback
    function simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90; // Don't complete until actual completion

            if (progressFill) {
                progressFill.style.width = progress + '%';
            }

            if (progress >= 90) {
                clearInterval(interval);
                if (progressText) {
                    progressText.textContent = 'Finalizing transcription...';
                }
            }
        }, 500);
    }

    // Drag and drop functionality
    const uploadContainer = document.querySelector('.upload-container');
    if (uploadContainer && fileInput) {
        uploadContainer.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadContainer.classList.add('drag-over');
        });

        uploadContainer.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadContainer.classList.remove('drag-over');
        });

        uploadContainer.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadContainer.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            }
        });
    }

    // Enhanced file information
    async function getFileInfo(file) {
        if (!fileInfo) return;

        try {
            const formData = new FormData();
            formData.append('audio_file', file);

            const response = await fetch('/api/file-info', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const info = await response.json();

                // Update max file size from server
                MAX_FILE_SIZE_MB = info.max_file_size_mb;

                // Display file information
                document.getElementById('duration').textContent = `${info.duration_seconds}s`;
                document.getElementById('est-time').textContent = `${info.estimated_processing_time}s`;
                document.getElementById('chunking').textContent = info.will_use_chunking ? 'Yes' : 'No';

                fileInfo.style.display = 'block';

                // Update progress text with estimated chunks
                if (info.will_use_chunking && progressDetail) {
                    progressDetail.textContent = `Will process in ${info.estimated_chunks} chunks`;
                }
            } else {
                console.warn('Failed to get file info');
                fileInfo.style.display = 'none';
            }
        } catch (error) {
            console.error('Error getting file info:', error);
            fileInfo.style.display = 'none';
        }
    }

    // Basic transcription fallback
    if (basicTranscribeLink) {
        basicTranscribeLink.addEventListener('click', (e) => {
            e.preventDefault();

            // Change form action to basic upload
            const form = document.getElementById('upload-form');
            if (form) {
                form.action = '/upload';

                // Hide enhanced options
                const options = document.querySelector('.transcription-options');
                if (options) options.style.display = 'none';

                // Update button text
                const btnText = document.querySelector('.btn-text');
                if (btnText) btnText.textContent = 'Basic Transcribe';

                // Hide the basic link
                basicTranscribeLink.style.display = 'none';
            }
        });
    }

    // Enhanced progress simulation
    function simulateEnhancedProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 90) progress = 90;

            if (progressFill) {
                progressFill.style.width = progress + '%';
            }

            // Update progress text based on progress
            if (progressText) {
                if (progress < 30) {
                    progressText.textContent = 'Analyzing audio file...';
                } else if (progress < 60) {
                    progressText.textContent = 'Processing audio chunks...';
                } else if (progress < 90) {
                    progressText.textContent = 'Generating timestamps...';
                } else {
                    progressText.textContent = 'Finalizing transcription...';
                }
            }

            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 800);
    }

    // Override the original progress simulation for enhanced upload
    const originalSimulateProgress = simulateProgress;
    simulateProgress = function() {
        const form = document.getElementById('upload-form');
        if (form && form.action.includes('enhanced-upload')) {
            simulateEnhancedProgress();
        } else {
            originalSimulateProgress();
        }
    };
});
