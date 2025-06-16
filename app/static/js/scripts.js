document.addEventListener('DOMContentLoaded', function() {
    // Update file name display when a file is selected
    const fileInput = document.getElementById('audio_file');
    const fileNameDisplay = document.getElementById('file-name');
    
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = fileInput.files[0].name;
            } else {
                fileNameDisplay.textContent = 'No file selected';
            }
        });
    }
});
