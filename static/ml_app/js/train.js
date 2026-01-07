// Training page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const trainingForm = document.getElementById('trainingForm');
    const startBtn = document.getElementById('startBtn');
    const progressContainer = document.getElementById('progressContainer');
    const statusMessage = document.getElementById('statusMessage');
    const progressFill = document.getElementById('progressFill');
    const currentEpoch = document.getElementById('currentEpoch');
    const totalEpochs = document.getElementById('totalEpochs');
    const lossValue = document.getElementById('lossValue');
    const accuracyValue = document.getElementById('accuracyValue');
    const valLossValue = document.getElementById('valLossValue');
    const valAccuracyValue = document.getElementById('valAccuracyValue');
    const epochHistory = document.getElementById('epochHistory');

    let pollInterval = null;

    // Auto-calculate test split
    const trainSplitInput = document.getElementById('train_split');
    const valSplitInput = document.getElementById('val_split');
    const testSplitInput = document.getElementById('test_split');
    
    function updateTestSplit() {
        const trainSplit = parseInt(trainSplitInput.value) || 70;
        const valSplit = parseInt(valSplitInput.value) || 15;
        const testSplit = 100 - trainSplit - valSplit;
        testSplitInput.value = Math.max(5, testSplit); // Ensure at least 5%
    }
    
    trainSplitInput.addEventListener('change', updateTestSplit);
    valSplitInput.addEventListener('change', updateTestSplit);

    // Form submission
    trainingForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Collect all form values
        const epochs = document.getElementById('epochs').value;
        const batchSize = document.getElementById('batch_size').value;
        const learningRate = document.getElementById('learning_rate').value;
        const optimizer = document.getElementById('optimizer').value;
        const imageSize = document.getElementById('image_size').value;
        const dropoutRate = document.getElementById('dropout_rate').value;
        const trainSplit = document.getElementById('train_split').value;
        const valSplit = document.getElementById('val_split').value;
        const earlyStoppingPatience = document.getElementById('early_stopping_patience').value;
        const pageLimit = document.getElementById('page_limit').value;

        // Validate splits
        const train = parseInt(trainSplit) || 70;
        const val = parseInt(valSplit) || 15;
        const test = 100 - train - val;
        
        if (test < 5) {
            alert('Train and Validation splits cannot exceed 95% total. Please adjust.');
            return;
        }

        // Disable form
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
        progressContainer.style.display = 'block';

        // Build form data
        const formData = new URLSearchParams();
        formData.append('epochs', epochs);
        formData.append('batch_size', batchSize);
        formData.append('learning_rate', learningRate);
        formData.append('optimizer', optimizer);
        formData.append('image_size', imageSize);
        formData.append('dropout_rate', dropoutRate);
        formData.append('train_split', trainSplit);
        formData.append('val_split', valSplit);
        formData.append('early_stopping_patience', earlyStoppingPatience);
        formData.append('page_limit', pageLimit);

        // Start training
        fetch('/train/start/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: formData.toString()
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                startBtn.disabled = false;
                startBtn.textContent = 'Start Training';
                return;
            }

            // Start polling for progress
            startPolling();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
            startBtn.disabled = false;
            startBtn.textContent = 'Start Training';
        });
    });

    function startPolling() {
        // Poll every 500ms
        pollInterval = setInterval(updateProgress, 500);
        updateProgress(); // Immediate first update
    }

    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }

    function updateProgress() {
        fetch('/train/progress/')
            .then(response => response.json())
            .then(data => {
                // Update status
                statusMessage.textContent = data.message || 'Processing...';
                statusMessage.className = 'status-message status-' + data.status;

                // Update progress bar
                const progress = data.progress_percent || 0;
                progressFill.style.width = progress + '%';
                progressFill.textContent = progress + '%';

                // Update metrics
                currentEpoch.textContent = data.current_epoch || 0;
                totalEpochs.textContent = data.total_epochs || 0;

                // Update latest values
                const history = data.history || {};
                const loss = history.loss || [];
                const accuracy = history.accuracy || [];
                const valLoss = history.val_loss || [];
                const valAccuracy = history.val_accuracy || [];

                if (loss.length > 0) {
                    lossValue.textContent = loss[loss.length - 1].toFixed(4);
                }
                if (accuracy.length > 0) {
                    accuracyValue.textContent = (accuracy[accuracy.length - 1] * 100).toFixed(2) + '%';
                }
                if (valLoss.length > 0) {
                    valLossValue.textContent = valLoss[valLoss.length - 1].toFixed(4);
                }
                if (valAccuracy.length > 0) {
                    valAccuracyValue.textContent = (valAccuracy[valAccuracy.length - 1] * 100).toFixed(2) + '%';
                }

                // Update epoch history (show last 10)
                updateEpochHistory(data);

                // Check if training is complete or errored
                if (data.status === 'completed' || data.status === 'error') {
                    stopPolling();
                    startBtn.disabled = false;
                    startBtn.textContent = 'Start Training';
                    
                    // Update final stats if available
                    if (data.status === 'completed' && data.final_stats) {
                        updateFinalStats(data.final_stats);
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching progress:', error);
            });
    }

    function updateFinalStats(finalStats) {
        const finalStatsSection = document.getElementById('finalStatsSection');
        if (!finalStatsSection) return;
        
        if (finalStats && Object.keys(finalStats).length > 0) {
            finalStatsSection.style.display = 'block';
            
            // Update all stat values
            if (finalStats.test_accuracy !== null && finalStats.test_accuracy !== undefined) {
                document.getElementById('finalTestAcc').textContent = (finalStats.test_accuracy * 100).toFixed(2) + '%';
            }
            if (finalStats.test_loss !== null && finalStats.test_loss !== undefined) {
                document.getElementById('finalTestLoss').textContent = finalStats.test_loss.toFixed(4);
            }
            if (finalStats.train_accuracy !== null && finalStats.train_accuracy !== undefined) {
                document.getElementById('finalTrainAcc').textContent = (finalStats.train_accuracy * 100).toFixed(2) + '%';
            }
            if (finalStats.train_loss !== null && finalStats.train_loss !== undefined) {
                document.getElementById('finalTrainLoss').textContent = finalStats.train_loss.toFixed(4);
            }
            if (finalStats.val_accuracy !== null && finalStats.val_accuracy !== undefined) {
                document.getElementById('finalValAcc').textContent = (finalStats.val_accuracy * 100).toFixed(2) + '%';
            }
            if (finalStats.val_loss !== null && finalStats.val_loss !== undefined) {
                document.getElementById('finalValLoss').textContent = finalStats.val_loss.toFixed(4);
            }
            if (finalStats.total_epochs_trained !== null && finalStats.total_epochs_trained !== undefined) {
                document.getElementById('finalEpochs').textContent = finalStats.total_epochs_trained;
            }
        }
    }

    function updateEpochHistory(data) {
        const history = data.history || {};
        const epochs = Math.max(
            (history.loss || []).length,
            (history.accuracy || []).length,
            (history.val_loss || []).length,
            (history.val_accuracy || []).length
        );

        if (epochs === 0) {
            epochHistory.innerHTML = '<div class="epoch-item">No epochs yet...</div>';
            return;
        }

        // Show last 10 epochs
        const startIdx = Math.max(0, epochs - 10);
        let html = '';
        
        for (let i = startIdx; i < epochs; i++) {
            const epochNum = i + 1;
            const loss = history.loss && history.loss[i] ? history.loss[i].toFixed(4) : '-';
            const acc = history.accuracy && history.accuracy[i] ? (history.accuracy[i] * 100).toFixed(2) : '-';
            const vLoss = history.val_loss && history.val_loss[i] ? history.val_loss[i].toFixed(4) : '-';
            const vAcc = history.val_accuracy && history.val_accuracy[i] ? (history.val_accuracy[i] * 100).toFixed(2) : '-';
            
            html += `<div class="epoch-item">
                Epoch ${epochNum}: Loss=${loss}, Acc=${acc}%, Val Loss=${vLoss}, Val Acc=${vAcc}%
            </div>`;
        }
        
        epochHistory.innerHTML = html;
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

    // Check initial status on page load
    updateProgress();
    setInterval(updateProgress, 2000); // Also poll every 2 seconds when not training
});

