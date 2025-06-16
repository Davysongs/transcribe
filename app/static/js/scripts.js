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

    // Maximum file size in MB
    const MAX_FILE_SIZE_MB = 25;

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
});
