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

            // Display image preview in results
            const resultImagePreview = document.getElementById('resultImagePreview');
            if (resultImagePreview) {
                resultImagePreview.src = imagePreview.src;
            }

            // Display results with enhanced visuals
            displayEnhancedResults(data);
            
            // Show Grad-CAM visualization if available
            if (data.gradcam_path) {
                displayGradCAM(data.gradcam_path);
            }

            resultsSection.style.display = 'block';
            resetButton();
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
            resetButton();
        });
    });

    function displayEnhancedResults(data) {
        // Store data for potential redraw
        window.lastPredictionData = data;
        
        // Update prediction class
        const predictedClassElement = document.getElementById('predictedClass');
        const overlayLabel = document.getElementById('overlayLabel');
        const badgeIcon = document.getElementById('badgeIcon');
        
        if (predictedClassElement) {
            predictedClassElement.textContent = data.predicted_class.replace('_', ' ');
        }
        if (overlayLabel) {
            overlayLabel.textContent = data.predicted_class.replace('_', ' ');
        }
        
        // Update icon based on prediction
        if (badgeIcon) {
            badgeIcon.textContent = data.predicted_class === 'clear' ? '✅' : '⚠️';
        }

        // Update confidence
        const confidence = (data.confidence * 100).toFixed(1);
        const confidenceValueElement = document.getElementById('confidenceValue');
        if (confidenceValueElement) {
            confidenceValueElement.textContent = confidence + '%';
        }

        // Update probability chart (with delay to ensure DOM is ready)
        setTimeout(() => {
            drawProbabilityChart(data.probabilities);
        }, 100);
        
        // Update probability list
        updateProbabilityList(data.probabilities);
        
        // Update comparison bars
        updateComparisonBars(data.probabilities);
        
        // Mark canvas as having data
        const canvas = document.getElementById('probabilityCanvas');
        if (canvas) {
            canvas.dataset.hasData = 'true';
        }
    }

    function drawProbabilityChart(probabilities) {
        const canvas = document.getElementById('probabilityCanvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const container = canvas.parentElement;
        const width = container ? container.clientWidth - 40 : 600;
        const height = 300;
        canvas.width = width;
        canvas.height = height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        const entries = Object.entries(probabilities);
        const barWidth = (width - 60) / entries.length;
        const maxProb = Math.max(...Object.values(probabilities));
        const colors = [ '#10b981', '#ef4444' ]; // green for clear, red for pollution

        entries.forEach(([className, prob], index) => {
            const barHeight = (prob / maxProb) * (height - 80);
            const x = 40 + index * (barWidth + 20);
            const y = height - barHeight - 40;
            const color = className === 'clear' ? colors[0] : colors[1];

            // Draw bar
            ctx.fillStyle = color;
            ctx.fillRect(x, y, barWidth, barHeight);

            // Draw value on top
            ctx.fillStyle = '#1e293b';
            ctx.font = 'bold 16px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(
                (prob * 100).toFixed(1) + '%',
                x + barWidth / 2,
                y - 10
            );

            // Draw label
            ctx.fillStyle = '#64748b';
            ctx.font = '14px sans-serif';
            ctx.fillText(
                className.replace('_', ' ').toUpperCase(),
                x + barWidth / 2,
                height - 15
            );
        });

        // Draw axes
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(30, height - 40);
        ctx.lineTo(width - 20, height - 40);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(30, 20);
        ctx.lineTo(30, height - 40);
        ctx.stroke();
    }

    function updateProbabilityList(probabilities) {
        const probList = document.getElementById('probList');
        if (!probList) return;
        
        probList.innerHTML = '';
        for (const [class_name, prob] of Object.entries(probabilities)) {
            const probItem = document.createElement('div');
            probItem.className = 'prob-item';
            const percentage = (prob * 100).toFixed(2);
            const colorClass = class_name === 'clear' ? 'status-success' : 'status-error';
            
            probItem.innerHTML = `
                <span class="prob-label">${class_name.replace('_', ' ').toUpperCase()}</span>
                <div style="flex: 1; margin: 0 1rem;">
                    <div style="height: 8px; background: var(--border-color); border-radius: 4px; overflow: hidden;">
                        <div style="height: 100%; width: ${percentage}%; background: ${class_name === 'clear' ? 'var(--success-color)' : 'var(--error-color)'}; transition: width 0.5s;"></div>
                    </div>
                </div>
                <span class="prob-value ${colorClass}">${percentage}%</span>
            `;
            probList.appendChild(probItem);
        }
    }

    function updateComparisonBars(probabilities) {
        const comparisonBars = document.getElementById('comparisonBars');
        if (!comparisonBars) return;
        
        comparisonBars.innerHTML = '';
        
        // Sort by probability (highest first)
        const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
        
        sorted.forEach(([class_name, prob]) => {
            const barItem = document.createElement('div');
            barItem.className = 'comparison-bar-item';
            const percentage = (prob * 100).toFixed(2);
            const color = class_name === 'clear' ? 'var(--success-color)' : 'var(--error-color)';
            
            barItem.innerHTML = `
                <div class="comparison-bar-label">${class_name.replace('_', ' ')}</div>
                <div class="comparison-bar-wrapper">
                    <div class="comparison-bar-fill" style="width: ${percentage}%; background: ${color};">
                        ${percentage}%
                    </div>
                </div>
                <div class="comparison-bar-value">${percentage}%</div>
            `;
            comparisonBars.appendChild(barItem);
        });
    }

    function resetButton() {
        predictBtn.disabled = false;
        btnText.textContent = 'Predict';
        btnLoader.style.display = 'none';
    }

    function displayGradCAM(gradcamPath) {
        const visualizationContainer = document.getElementById('visualizationContainer');
        const visualizationGrid = document.getElementById('visualizationGrid');
        
        if (!visualizationContainer || !visualizationGrid) return;
        
        // Clear previous visualizations
        visualizationGrid.innerHTML = '';
        
        // Create image element
        const img = document.createElement('img');
        img.src = gradcamPath;
        img.alt = 'Grad-CAM Visualization';
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.borderRadius = '8px';
        
        const item = document.createElement('div');
        item.className = 'visualization-item';
        item.innerHTML = '<div class="visualization-item-title">Model Attention Analysis</div>';
        item.appendChild(img);
        
        visualizationGrid.appendChild(item);
        visualizationContainer.style.display = 'block';
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

