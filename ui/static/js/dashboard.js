/**
 * NBA Prediction System Dashboard
 * 
 * This module handles the dashboard UI functionality, including:
 * - Fetching data from the API
 * - Rendering performance charts
 * - Updating model status indicators
 * - Handling training controls
 */

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI components
    initializePerformanceCharts();
    initializeModelStatusUpdates();
    initializeTrainingControls();
    initializeDriftDetection();

    // Set up periodic refreshes
    setInterval(refreshDashboardData, 60000); // Refresh every minute
});

// Chart objects for easy reference
const charts = {
    performanceChart: null,
    modelComparisonChart: null,
    trainingHistoryChart: null,
    driftDetectionChart: null
};

/**
 * Initialize performance charts
 */
async function initializePerformanceCharts() {
    try {
        // Fetch model performance data
        const performanceData = await apiClient.getModelPerformance(30);
        
        if (performanceData.status !== 'success') {
            showErrorMessage('Failed to load performance data');
            return;
        }

        // Render main performance chart
        renderPerformanceChart(performanceData.performance);
        
        // Render model comparison chart
        renderModelComparisonChart(performanceData.performance);

    } catch (error) {
        console.error('Error initializing performance charts:', error);
        showErrorMessage('Failed to load performance data');
    }
}

/**
 * Render the main performance chart showing accuracy over time
 * @param {Array} performanceData - Model performance data from API
 */
function renderPerformanceChart(performanceData) {
    // Prepare data for chart
    const chartData = preparePerformanceChartData(performanceData);
    
    // Get the canvas element
    const ctx = document.getElementById('performance-chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (charts.performanceChart) {
        charts.performanceChart.destroy();
    }
    
    // Create new chart
    charts.performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: chartData.datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Model Accuracy Trends (30-Day)',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Update last updated timestamp
    updateLastUpdatedTimestamp();
}

/**
 * Prepare data for the performance chart
 * @param {Array} performanceData - Raw performance data from API
 * @returns {Object} - Formatted data for Chart.js
 */
function preparePerformanceChartData(performanceData) {
    // Set up colors for each model type
    const modelColors = {
        'RandomForest': 'rgba(54, 162, 235, 1)',
        'XGBoost': 'rgba(255, 99, 132, 1)',
        'Bayesian': 'rgba(75, 192, 192, 1)',
        'AnomalyDetection': 'rgba(153, 102, 255, 1)',
        'ModelMixing': 'rgba(255, 159, 64, 1)',
        'EnsembleStacking': 'rgba(255, 205, 86, 1)',
        'HyperparameterTuning': 'rgba(201, 203, 207, 1)'
    };
    
    // Extract dates from the data (assuming data is sorted by date)
    const allDates = new Set();
    const modelPerformanceMap = {};
    
    // First pass: collect all dates and organize data by model
    performanceData.forEach(modelData => {
        const modelKey = `${modelData.model_name}_${modelData.prediction_target}`;
        
        if (!modelPerformanceMap[modelKey]) {
            modelPerformanceMap[modelKey] = {
                modelName: modelData.model_name,
                predictionTarget: modelData.prediction_target,
                dataByDate: {}
            };
        }
        
        // Process history entries
        modelData.metrics_history.forEach(entry => {
            const date = entry.created_at.split('T')[0]; // Extract date part
            allDates.add(date);
            
            // Store metrics by date
            modelPerformanceMap[modelKey].dataByDate[date] = {
                accuracy: entry.metrics.accuracy * 100, // Convert to percentage
                precision: entry.metrics.precision * 100,
                recall: entry.metrics.recall * 100,
                f1_score: entry.metrics.f1_score * 100
            };
        });
    });
    
    // Sort dates chronologically
    const sortedDates = Array.from(allDates).sort();
    
    // Create datasets for chart
    const datasets = [];
    
    // Focus on moneyline predictions first (most common)
    Object.values(modelPerformanceMap)
        .filter(model => model.predictionTarget === 'moneyline')
        .forEach(model => {
            const modelName = model.modelName;
            const color = modelColors[modelName] || 'rgba(0, 0, 0, 1)';
            
            const accuracyData = sortedDates.map(date => {
                return model.dataByDate[date] ? model.dataByDate[date].accuracy : null;
            });
            
            datasets.push({
                label: `${modelName} (Moneyline)`,
                data: accuracyData,
                borderColor: color,
                backgroundColor: color.replace('1)', '0.2)'),
                borderWidth: 2,
                tension: 0.1,
                fill: false
            });
        });
    
    return {
        labels: sortedDates,
        datasets: datasets
    };
}

/**
 * Render model comparison chart
 * @param {Array} performanceData - Model performance data from API
 */
function renderModelComparisonChart(performanceData) {
    // Get the canvas element
    const ctx = document.getElementById('model-comparison-chart').getContext('2d');
    
    // Prepare data for comparison chart (most recent accuracy for each model)
    const comparisonData = prepareModelComparisonData(performanceData);
    
    // Destroy existing chart if it exists
    if (charts.modelComparisonChart) {
        charts.modelComparisonChart.destroy();
    }
    
    // Create new chart
    charts.modelComparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: comparisonData.labels,
            datasets: comparisonData.datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Current Model Performance Comparison',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

/**
 * Prepare data for model comparison chart
 * @param {Array} performanceData - Raw performance data from API
 * @returns {Object} - Formatted data for Chart.js
 */
function prepareModelComparisonData(performanceData) {
    // Extract latest performance metrics for each model
    const latestMetrics = {};
    
    performanceData.forEach(modelData => {
        // Skip if no history
        if (!modelData.metrics_history || modelData.metrics_history.length === 0) {
            return;
        }
        
        const modelKey = `${modelData.model_name}_${modelData.prediction_target}`;
        const latestEntry = modelData.metrics_history[0]; // First entry is most recent
        
        latestMetrics[modelKey] = {
            modelName: modelData.model_name,
            predictionTarget: modelData.prediction_target,
            accuracy: latestEntry.metrics.accuracy * 100, // Convert to percentage
            precision: latestEntry.metrics.precision * 100,
            recall: latestEntry.metrics.recall * 100,
            f1_score: latestEntry.metrics.f1_score * 100
        };
    });
    
    // Organize by prediction target
    const predictionTargets = ['moneyline', 'spread', 'total', 'player_props'];
    const datasets = [];
    
    predictionTargets.forEach((target, index) => {
        // Filter models for this prediction target
        const targetModels = Object.values(latestMetrics)
            .filter(model => model.predictionTarget === target);
        
        if (targetModels.length === 0) {
            return; // Skip if no models for this target
        }
        
        // Get model names and accuracy values
        const modelNames = targetModels.map(model => model.modelName);
        const accuracyValues = targetModels.map(model => model.accuracy);
        
        // Define colors based on prediction target
        let backgroundColor;
        let borderColor;
        let label;
        
        switch(target) {
            case 'moneyline':
                backgroundColor = 'rgba(54, 162, 235, 0.5)';
                borderColor = 'rgba(54, 162, 235, 1)';
                label = 'Moneyline';
                break;
            case 'spread':
                backgroundColor = 'rgba(255, 99, 132, 0.5)';
                borderColor = 'rgba(255, 99, 132, 1)';
                label = 'Spread';
                break;
            case 'total':
                backgroundColor = 'rgba(75, 192, 192, 0.5)';
                borderColor = 'rgba(75, 192, 192, 1)';
                label = 'Total Points';
                break;
            case 'player_props':
                backgroundColor = 'rgba(153, 102, 255, 0.5)';
                borderColor = 'rgba(153, 102, 255, 1)';
                label = 'Player Props';
                break;
            default:
                backgroundColor = 'rgba(201, 203, 207, 0.5)';
                borderColor = 'rgba(201, 203, 207, 1)';
                label = target;
        }
        
        datasets.push({
            label: label,
            data: accuracyValues,
            backgroundColor: backgroundColor,
            borderColor: borderColor,
            borderWidth: 1
        });
    });
    
    // Get unique model names across all prediction targets
    const uniqueModelNames = Array.from(new Set(
        Object.values(latestMetrics).map(model => model.modelName)
    ));
    
    return {
        labels: uniqueModelNames,
        datasets: datasets
    };
}

/**
 * Initialize and handle model status updates
 */
async function initializeModelStatusUpdates() {
    try {
        // Fetch model list
        const modelListData = await apiClient.getModelsList();
        
        if (modelListData.status !== 'success') {
            showErrorMessage('Failed to load model list');
            return;
        }
        
        // Update model status indicators
        updateModelStatusIndicators(modelListData.models);
        
        // Fetch latest training status
        const trainingStatus = await apiClient.getTrainingStatus();
        updateTrainingStatusIndicator(trainingStatus);
        
    } catch (error) {
        console.error('Error initializing model status updates:', error);
        showErrorMessage('Failed to load model status');
    }
}

/**
 * Update model status indicators
 * @param {Array} models - List of models from API
 */
function updateModelStatusIndicators(models) {
    // Group models by name
    const modelGroups = {};
    
    models.forEach(model => {
        if (!modelGroups[model.model_name]) {
            modelGroups[model.model_name] = [];
        }
        modelGroups[model.model_name].push(model);
    });
    
    // Update status indicators
    Object.entries(modelGroups).forEach(([modelName, modelVersions]) => {
        // Find the status indicator element
        const statusElement = document.getElementById(`${modelName.toLowerCase()}-status`);
        if (!statusElement) return;
        
        // Get active models
        const activeModels = modelVersions.filter(model => model.active);
        const needsTraining = modelVersions.some(model => model.needs_training);
        
        // Update status indicator
        if (needsTraining) {
            statusElement.textContent = 'Needs Training';
            statusElement.classList.add('text-warning');
            statusElement.classList.remove('text-success', 'text-danger');
        } else if (activeModels.length > 0) {
            statusElement.textContent = 'Active';
            statusElement.classList.add('text-success');
            statusElement.classList.remove('text-warning', 'text-danger');
        } else {
            statusElement.textContent = 'Inactive';
            statusElement.classList.add('text-danger');
            statusElement.classList.remove('text-success', 'text-warning');
        }
        
        // Update last trained date
        const lastTrainedElement = document.getElementById(`${modelName.toLowerCase()}-last-trained`);
        if (lastTrainedElement && activeModels.length > 0) {
            // Find latest training date
            const trainedDates = activeModels
                .map(model => model.trained_at)
                .filter(date => date) // Filter out null dates
                .sort()
                .reverse(); // Most recent first
                
            if (trainedDates.length > 0) {
                const formattedDate = new Date(trainedDates[0]).toLocaleDateString();
                lastTrainedElement.textContent = formattedDate;
            } else {
                lastTrainedElement.textContent = 'Never';
            }
        }
    });
}

/**
 * Update training status indicator
 * @param {Object} status - Training status from API
 */
function updateTrainingStatusIndicator(status) {
    const trainingStatusElement = document.getElementById('training-status');
    if (!trainingStatusElement) return;
    
    const trainingProgressElement = document.getElementById('training-progress');
    
    if (status.is_training) {
        trainingStatusElement.textContent = 'Training in Progress';
        trainingStatusElement.classList.add('text-primary');
        trainingStatusElement.classList.remove('text-success', 'text-secondary');
        
        // Show progress indicator if available
        if (trainingProgressElement) {
            trainingProgressElement.style.display = 'block';
            
            // Update progress information if available
            if (status.model_status) {
                const currentModel = status.model_status.current_model || 'Unknown';
                const progressElement = document.getElementById('training-progress-text');
                if (progressElement) {
                    progressElement.textContent = `Training: ${currentModel}`;
                }
            }
        }
    } else {
        trainingStatusElement.textContent = 'Training Idle';
        trainingStatusElement.classList.add('text-secondary');
        trainingStatusElement.classList.remove('text-primary', 'text-success');
        
        // Hide progress indicator
        if (trainingProgressElement) {
            trainingProgressElement.style.display = 'none';
        }
    }
}

/**
 * Initialize training controls
 */
function initializeTrainingControls() {
    // Start training button
    const startTrainingBtn = document.getElementById('start-training-btn');
    if (startTrainingBtn) {
        startTrainingBtn.addEventListener('click', async () => {
            try {
                // Get auth token from input
                const tokenInput = document.getElementById('api-token-input');
                const token = tokenInput ? tokenInput.value : null;
                
                if (token) {
                    apiClient.setApiToken(token);
                }
                
                // Show loading state
                startTrainingBtn.disabled = true;
                startTrainingBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';
                
                // Start training
                const response = await apiClient.startTraining(true);
                
                // Show success message
                if (response.status === 'success') {
                    showSuccessMessage('Training started successfully');
                    // Refresh training status after a short delay
                    setTimeout(async () => {
                        const status = await apiClient.getTrainingStatus();
                        updateTrainingStatusIndicator(status);
                    }, 2000);
                } else {
                    showErrorMessage(response.message || 'Failed to start training');
                }
                
            } catch (error) {
                console.error('Error starting training:', error);
                showErrorMessage('Failed to start training');
            } finally {
                // Reset button
                startTrainingBtn.disabled = false;
                startTrainingBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
            }
        });
    }
    
    // Initialize individual model retrain buttons
    document.querySelectorAll('[id$="-retrain-btn"]').forEach(button => {
        button.addEventListener('click', async (event) => {
            // Extract model name from button ID
            const modelName = event.target.id.replace('-retrain-btn', '');
            
            try {
                // Get auth token from input
                const tokenInput = document.getElementById('api-token-input');
                const token = tokenInput ? tokenInput.value : null;
                
                if (token) {
                    apiClient.setApiToken(token);
                }
                
                // Show loading state
                button.disabled = true;
                button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
                
                // Determine prediction target (default to moneyline)
                const predictionTarget = 'moneyline';
                
                // Start retraining for this model
                const response = await apiClient.retrainModel(modelName, predictionTarget, true);
                
                // Show success message
                if (response.status === 'success') {
                    showSuccessMessage(`Retraining started for ${modelName}`);
                    // Refresh training status after a short delay
                    setTimeout(async () => {
                        const status = await apiClient.getTrainingStatus();
                        updateTrainingStatusIndicator(status);
                    }, 2000);
                } else {
                    showErrorMessage(response.message || `Failed to retrain ${modelName}`);
                }
                
            } catch (error) {
                console.error(`Error retraining ${modelName}:`, error);
                showErrorMessage(`Failed to retrain ${modelName}`);
            } finally {
                // Reset button
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-sync-alt"></i>';
            }
        });
    });
}

/**
 * Initialize drift detection monitoring
 */
async function initializeDriftDetection() {
    try {
        // Fetch drift detection data
        const driftData = await apiClient.checkModelDrift();
        
        if (driftData.status !== 'success') {
            console.error('Failed to load drift detection data');
            return;
        }
        
        // Update drift detection indicators
        updateDriftDetectionIndicators(driftData.drift_results);
        
    } catch (error) {
        console.error('Error initializing drift detection:', error);
    }
}

/**
 * Update drift detection indicators
 * @param {Array} driftResults - Drift detection results from API
 */
function updateDriftDetectionIndicators(driftResults) {
    // Count models with drift
    const modelsWithDrift = driftResults.filter(result => result.drift_detected).length;
    const totalModels = driftResults.length;
    
    // Update summary card
    const driftSummaryElement = document.getElementById('drift-summary');
    if (driftSummaryElement) {
        driftSummaryElement.textContent = `${modelsWithDrift} of ${totalModels} models need retraining`;
    }
    
    // Update status class
    const driftStatusElement = document.getElementById('drift-status');
    if (driftStatusElement) {
        if (modelsWithDrift > 0) {
            driftStatusElement.className = 'badge bg-warning';
            driftStatusElement.textContent = 'Drift Detected';
        } else {
            driftStatusElement.className = 'badge bg-success';
            driftStatusElement.textContent = 'No Drift';
        }
    }
    
    // Update individual model drift indicators
    driftResults.forEach(result => {
        const elementId = `${result.model_name.toLowerCase()}-drift`;
        const driftElement = document.getElementById(elementId);
        
        if (driftElement) {
            if (result.drift_detected) {
                driftElement.className = 'badge bg-warning';
                driftElement.textContent = 'Drift Detected';
            } else {
                driftElement.className = 'badge bg-success';
                driftElement.textContent = 'Stable';
            }
        }
    });
}

/**
 * Refresh all dashboard data
 */
async function refreshDashboardData() {
    try {
        // Fetch new data
        const [performanceData, trainingStatus, driftData] = await Promise.all([
            apiClient.getModelPerformance(30),
            apiClient.getTrainingStatus(),
            apiClient.checkModelDrift()
        ]);
        
        // Update UI with new data
        if (performanceData.status === 'success') {
            renderPerformanceChart(performanceData.performance);
            renderModelComparisonChart(performanceData.performance);
        }
        
        updateTrainingStatusIndicator(trainingStatus);
        
        if (driftData.status === 'success') {
            updateDriftDetectionIndicators(driftData.drift_results);
        }
        
        // Update last updated timestamp
        updateLastUpdatedTimestamp();
        
    } catch (error) {
        console.error('Error refreshing dashboard data:', error);
    }
}

/**
 * Update the last updated timestamp
 */
function updateLastUpdatedTimestamp() {
    const timestampElement = document.getElementById('last-updated');
    if (timestampElement) {
        const now = new Date();
        timestampElement.textContent = now.toLocaleString();
    }
}

/**
 * Show a success message
 * @param {string} message - Message to display
 */
function showSuccessMessage(message) {
    // Use the toast system if available, otherwise alert
    const toastContainer = document.getElementById('toast-container');
    
    if (toastContainer) {
        const toastId = 'success-toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-success border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-check-circle me-2"></i> ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: 5000 });
        toast.show();
    } else {
        console.log('Success:', message);
    }
}

/**
 * Show an error message
 * @param {string} message - Error message to display
 */
function showErrorMessage(message) {
    // Use the toast system if available, otherwise console error
    const toastContainer = document.getElementById('toast-container');
    
    if (toastContainer) {
        const toastId = 'error-toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-danger border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-exclamation-circle me-2"></i> ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: 5000 });
        toast.show();
    } else {
        console.error('Error:', message);
    }
}
