# NBA Prediction System Setup Script
# ================================
# This production-quality script configures the database and runs the system with real NBA data

# Stop any running Python processes
try {
    Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Stopped any running Python processes" -ForegroundColor Yellow
} catch {
    # No Python processes were running
}

# Create necessary directories
if (-not (Test-Path "logs")) {
    New-Item -Path "logs" -ItemType Directory -Force | Out-Null
    Write-Host "Created logs directory" -ForegroundColor Green
}

# Define PostgreSQL configuration
$env:POSTGRES_HOST="localhost"
$env:POSTGRES_PORT="5432"
$env:POSTGRES_DB="postgres"
$env:POSTGRES_USER="postgres"
$env:POSTGRES_PASSWORD="ALGO123"

# Define API Keys
$env:BALLDONTLIE_API_KEY="d0e93357-b9b0-4a62-bed1-920eeab5db50"
$env:ODDS_API_KEY="12186f096bb2e6a9f9b472391323893d"
$env:API_TOKEN="nba_api_token_2025"

# Display configuration
Write-Host ""
Write-Host "NBA PREDICTION SYSTEM CONFIGURATION" -ForegroundColor Cyan
Write-Host "-------------------------------" -ForegroundColor Cyan
Write-Host "Database:" -ForegroundColor Yellow
Write-Host "  Host:     $env:POSTGRES_HOST"
Write-Host "  Port:     $env:POSTGRES_PORT"
Write-Host "  Database: $env:POSTGRES_DB"
Write-Host "  User:     $env:POSTGRES_USER"
Write-Host "  Password: $env:POSTGRES_PASSWORD"
Write-Host ""
Write-Host "API Keys:" -ForegroundColor Yellow
Write-Host "  BallDontLie: $($env:BALLDONTLIE_API_KEY.Substring(0,5))..." -ForegroundColor Green
Write-Host "  The Odds:    $($env:ODDS_API_KEY.Substring(0,5))..." -ForegroundColor Green
Write-Host ""

# Create a temporary configuration file
$configContent = @"
# PostgreSQL Configuration (Auto-generated)
POSTGRES_HOST=$env:POSTGRES_HOST
POSTGRES_PORT=$env:POSTGRES_PORT
POSTGRES_DB=$env:POSTGRES_DB
POSTGRES_USER=$env:POSTGRES_USER
POSTGRES_PASSWORD=$env:POSTGRES_PASSWORD

# API Keys
BALLDONTLIE_API_KEY=$env:BALLDONTLIE_API_KEY
ODDS_API_KEY=$env:ODDS_API_KEY
API_TOKEN=$env:API_TOKEN
"@

$configContent | Out-File -FilePath "database.conf" -Encoding utf8
Write-Host "Created database.conf with current settings" -ForegroundColor Green

# Check database connectivity
Write-Host "Testing database connection..." -NoNewline

$connectionCheck = $false
try {
    # Try to connect to PostgreSQL using PowerShell
    # This is a simplified check - in reality you might need a PostgreSQL client
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.Connect($env:POSTGRES_HOST, [int]$env:POSTGRES_PORT)
    $tcp.Close()
    $connectionCheck = $true
    Write-Host "OK" -ForegroundColor Green
} catch {
    Write-Host "FAILED" -ForegroundColor Red
    Write-Host "Error connecting to PostgreSQL: $_" -ForegroundColor Red
    Write-Host "Please ensure PostgreSQL is running on $env:POSTGRES_HOST:$env:POSTGRES_PORT" -ForegroundColor Red
    Write-Host "You may need to install PostgreSQL or start the service." -ForegroundColor Yellow
    
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Initialize database (create tables and seed data)
Write-Host ""
Write-Host "INITIALIZING DATABASE" -ForegroundColor Cyan
Write-Host "---------------------" -ForegroundColor Cyan
Write-Host "Creating necessary tables and setting up for real NBA data..." -ForegroundColor Yellow

try {
    python -c "from src.database.connection import init_db; init_db()"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Database tables initialized successfully" -ForegroundColor Green
    } else {
        Write-Host "Warning: Database initialization returned non-zero exit code" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error initializing database: $_" -ForegroundColor Red
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Start the API server
Write-Host ""
Write-Host "STARTING NBA PREDICTION SYSTEM" -ForegroundColor Cyan
Write-Host "---------------------------" -ForegroundColor Cyan
Write-Host "The dashboard will be available at: http://localhost:5000/dashboard" -ForegroundColor Green
Write-Host "API endpoints will be available at: http://localhost:5000/api/*" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the server
python -m src.api.server
