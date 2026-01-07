// Prediction page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');
    const resultsSection = document.getElementById('resultsSection');
    const predictedClass = document.getElementById('predictedClass');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const probList = document.getElementById('probList');

    let selectedFile = null;

    // File input change
    imageInput.addEventListener('change', function(e) {
        handleFile(e.target.files[0]);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });

    uploadArea.addEventListener('click', function() {
        imageInput.click();
    });

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            uploadArea.style.display = 'none';
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Predict button
    predictBtn.addEventListener('click', function() {
        if (!selectedFile) return;

        // Show loading state
        predictBtn.disabled = true;
        btnText.textContent = 'Predicting...';
        btnLoader.style.display = 'inline-block';
        resultsSection.style.display = 'none';

        // Prepare form data
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Send request
        fetch('/predict/upload/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                resetButton();
                return;
            }

            // Display results
            predictedClass.textContent = data.predicted_class;
            const confidence = (data.confidence * 100).toFixed(1);
            confidenceValue.textContent = confidence + '%';
            confidenceFill.style.width = confidence + '%';

            // Display probabilities
            probList.innerHTML = '';
            for (const [class_name, prob] of Object.entries(data.probabilities)) {
                const probItem = document.createElement('div');
                probItem.className = 'prob-item';
                probItem.innerHTML = `
                    <span class="prob-label">${class_name.replace('_', ' ')}</span>
                    <span class="prob-value">${(prob * 100).toFixed(2)}%</span>
                `;
                probList.appendChild(probItem);
            }

            resultsSection.style.display = 'block';
            resetButton();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
            resetButton();
        });
    });

    function resetButton() {
        predictBtn.disabled = false;
        btnText.textContent = 'Predict';
        btnLoader.style.display = 'none';
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});

