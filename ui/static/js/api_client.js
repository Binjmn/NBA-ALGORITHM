/**
 * NBA Prediction System API Client
 * 
 * This module provides a JavaScript client for interacting with the NBA Prediction
 * System API endpoints. It handles fetching model performance data, training status,
 * and prediction information.
 */

const API_BASE_URL = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';

class NbaPredictionApi {
    /**
     * Initialize the API client
     * @param {string} baseUrl - Base URL for API requests
     * @param {string} apiToken - Optional API token for authenticated requests
     */
    constructor(baseUrl = API_BASE_URL, apiToken = null) {
        this.baseUrl = baseUrl;
        this.apiToken = apiToken;
    }

    /**
     * Set the API token for authenticated requests
     * @param {string} token - API token
     */
    setApiToken(token) {
        this.apiToken = token;
    }

    /**
     * Create headers for API requests
     * @param {boolean} includeAuth - Whether to include authentication token
     * @returns {Object} - Headers object
     */
    _createHeaders(includeAuth = false) {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (includeAuth && this.apiToken) {
            headers['Authorization'] = `Bearer ${this.apiToken}`;
        }

        return headers;
    }

    /**
     * Make an API request
     * @param {string} endpoint - API endpoint
     * @param {string} method - HTTP method
     * @param {Object} data - Request data
     * @param {boolean} requiresAuth - Whether the endpoint requires authentication
     * @returns {Promise} - Promise resolving to API response
     */
    async _request(endpoint, method = 'GET', data = null, requiresAuth = false) {
        const url = `${this.baseUrl}${endpoint}`;
        const options = {
            method,
            headers: this._createHeaders(requiresAuth),
        };

        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * Check API health status
     * @returns {Promise} - Promise resolving to health status
     */
    async getHealthStatus() {
        return this._request('/api/health');
    }

    /**
     * Get list of available models
     * @returns {Promise} - Promise resolving to model list
     */
    async getModelsList() {
        return this._request('/api/models/list');
    }

    /**
     * Get details for a specific model
     * @param {string} modelName - Name of the model
     * @param {string} predictionTarget - Type of prediction (moneyline, spread, etc.)
     * @param {number} version - Optional model version
     * @returns {Promise} - Promise resolving to model details
     */
    async getModelDetails(modelName, predictionTarget, version = null) {
        let endpoint = `/api/models/${modelName}/details?prediction_target=${predictionTarget}`;
        if (version) {
            endpoint += `&version=${version}`;
        }
        return this._request(endpoint);
    }

    /**
     * Get current training status
     * @returns {Promise} - Promise resolving to training status
     */
    async getTrainingStatus() {
        return this._request('/api/training/status');
    }

    /**
     * Start model training
     * @param {boolean} force - Whether to force training regardless of last training time
     * @param {Array} models - Optional list of specific models to train
     * @returns {Promise} - Promise resolving to training start status
     */
    async startTraining(force = false, models = null) {
        const data = { force };
        if (models && models.length > 0) {
            data.models = models.join(',');
        }
        return this._request('/api/training/start', 'POST', data, true);
    }

    /**
     * Cancel ongoing training
     * @returns {Promise} - Promise resolving to cancel status
     */
    async cancelTraining() {
        return this._request('/api/training/cancel', 'POST', {}, true);
    }

    /**
     * Retrain a specific model
     * @param {string} modelName - Name of the model to retrain
     * @param {string} predictionTarget - Type of prediction
     * @param {boolean} force - Whether to force immediate retraining
     * @returns {Promise} - Promise resolving to retrain status
     */
    async retrainModel(modelName, predictionTarget, force = true) {
        return this._request(
            `/api/models/${modelName}/retrain`,
            'POST',
            { prediction_target: predictionTarget, force },
            true
        );
    }

    /**
     * Get performance metrics for all models
     * @param {number} days - Number of days of history to retrieve
     * @returns {Promise} - Promise resolving to performance metrics
     */
    async getModelPerformance(days = 7) {
        return this._request(`/api/models/performance?days=${days}`);
    }

    /**
     * Check for model drift
     * @returns {Promise} - Promise resolving to drift detection results
     */
    async checkModelDrift() {
        return this._request('/api/models/drift');
    }

    /**
     * Get predictions for today's games
     * @returns {Promise} - Promise resolving to today's predictions
     */
    async getTodaysPredictions() {
        return this._request('/api/predictions/today');
    }
}

// Create global API client instance
const apiClient = new NbaPredictionApi();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NbaPredictionApi, apiClient };
}
