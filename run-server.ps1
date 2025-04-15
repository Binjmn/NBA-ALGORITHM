# Production-ready NBA Prediction System Server Launcher
# =====================================================

# Clear terminal for better readability
Clear-Host

# Configure environment variables for database connection and API keys
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
$env:POSTGRES_DB = "postgres"
$env:POSTGRES_USER = "postgres"
$env:POSTGRES_PASSWORD = "ALGO123"

# Set API keys (your real API keys for production use)
$env:BALLDONTLIE_API_KEY = "d0e93357-b9b0-4a62-bed1-920eeab5db50"
$env:ODDS_API_KEY = "12186f096bb2e6a9f9b472391323893d"
$env:API_TOKEN = "nba_api_token_2025"

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -Path "logs" -ItemType Directory -Force | Out-Null
}

# Display configuration information
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " NBA PREDICTION SYSTEM - PRODUCTION SERVER STARTUP " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Database Configuration:" -ForegroundColor Yellow
Write-Host "  Host:     $($env:POSTGRES_HOST)" 
Write-Host "  Port:     $($env:POSTGRES_PORT)"
Write-Host "  Database: $($env:POSTGRES_DB)"
Write-Host "  User:     $($env:POSTGRES_USER)"
Write-Host ""
Write-Host "API Keys Status:" -ForegroundColor Yellow
Write-Host "  BallDontLie API: " -NoNewline
if ($env:BALLDONTLIE_API_KEY) {
    Write-Host "Configured" -ForegroundColor Green
} else {
    Write-Host "Missing" -ForegroundColor Red
}
Write-Host "  The Odds API:    " -NoNewline
if ($env:ODDS_API_KEY) {
    Write-Host "Configured" -ForegroundColor Green
} else {
    Write-Host "Missing" -ForegroundColor Red
}
Write-Host ""
Write-Host "Server Information:" -ForegroundColor Yellow
Write-Host "  Dashboard URL: http://localhost:5000/dashboard"
Write-Host "  API Base URL:  http://localhost:5000/api"
Write-Host "  Health Check:  http://localhost:5000/api/health"
Write-Host ""
Write-Host "Starting server with production settings..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server at any time"
Write-Host "======================================================" -ForegroundColor Cyan

# Run the server with proper path resolution
python -m src.api.server
