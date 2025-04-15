<#
.SYNOPSIS
    Installs PostgreSQL for the NBA Prediction System
.DESCRIPTION
    This script downloads and installs PostgreSQL, creates the database,
    and sets up the initial schema all in one step.
#>

# Script parameters
param(
    [string]$PostgresPassword = "ALGO123",
    [string]$DatabaseName = "nba_prediction"
)

# Ensure we're running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script must be run as Administrator. Please restart with admin rights." -ForegroundColor Red
    exit 1
}

# Create log directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
Start-Transcript -Path "logs\postgres_install.log" -Force

Write-Host "\n===== NBA PREDICTION SYSTEM - POSTGRESQL INSTALLER =====" -ForegroundColor Cyan

# Download the PostgreSQL installer
$tempDir = "$env:TEMP\postgres_installer"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$postgresVersion = "15.4-1"
$installerUrl = "https://get.enterprisedb.com/postgresql/postgresql-$postgresVersion-windows-x64.exe"
$installerPath = "$tempDir\postgresql-installer.exe"

Write-Host "Downloading PostgreSQL installer..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "Download completed!" -ForegroundColor Green
}
catch {
    Write-Host "Failed to download PostgreSQL installer: $_" -ForegroundColor Red
    exit 1
}

# Install PostgreSQL silently
Write-Host "\nInstalling PostgreSQL..." -ForegroundColor Yellow
Write-Host "This will take a few minutes. Please be patient." -ForegroundColor Yellow

$installArgs = "--mode unattended --unattendedmodeui none --superpassword $PostgresPassword " + 
              "--serverport 5432 --disable-components pgAdmin,StackBuilder --enable-components server"

try {
    $proc = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Host "PostgreSQL installation failed with exit code $($proc.ExitCode)" -ForegroundColor Red
        exit 1
    }
    Write-Host "PostgreSQL installed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "Failed to install PostgreSQL: $_" -ForegroundColor Red
    exit 1
}

# Add PostgreSQL bin directory to PATH for this session
$pgBinPath = "C:\Program Files\PostgreSQL\15\bin"
$env:Path = "$pgBinPath;$env:Path"

# Wait a moment for services to start
Write-Host "\nWaiting for PostgreSQL service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Create the database
Write-Host "\nCreating database $DatabaseName..." -ForegroundColor Yellow

$env:PGPASSWORD = $PostgresPassword
try {
    $output = & "$pgBinPath\psql" -U postgres -c "CREATE DATABASE $DatabaseName;" 2>&1
    Write-Host "Database created successfully!" -ForegroundColor Green
}
catch {
    Write-Host "Failed to create database: $_" -ForegroundColor Red
    # Check if database already exists
    $checkDbResult = & "$pgBinPath\psql" -U postgres -c "SELECT 1 FROM pg_database WHERE datname = '$DatabaseName';" 2>&1
    if ($checkDbResult -like "*1 row*") {
        Write-Host "Database already exists. Continuing..." -ForegroundColor Yellow
    }
    else {
        exit 1
    }
}

# Create database schema
Write-Host "\nCreating database schema..." -ForegroundColor Yellow

$schemaFile = "$PSScriptRoot\db_schema.sql"

# Create the schema file
$schemaContent = @"
-- NBA Prediction System Database Schema

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    abbreviation VARCHAR(10) NOT NULL,
    city VARCHAR(100) NOT NULL,
    conference VARCHAR(50),
    division VARCHAR(50),
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Players table
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    team_id VARCHAR(100) REFERENCES teams(team_id),
    position VARCHAR(50),
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Games table
CREATE TABLE IF NOT EXISTS games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) UNIQUE NOT NULL,
    season_year INTEGER NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team_id VARCHAR(100) NOT NULL REFERENCES teams(team_id),
    away_team_id VARCHAR(100) NOT NULL REFERENCES teams(team_id),
    status VARCHAR(50) NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    odds JSONB NOT NULL DEFAULT '{}'::jsonb,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    predictions JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Game Statistics table
CREATE TABLE IF NOT EXISTS game_stats (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL REFERENCES games(game_id),
    player_id VARCHAR(100) NOT NULL REFERENCES players(player_id),
    team_id VARCHAR(100) NOT NULL REFERENCES teams(team_id),
    minutes INTEGER,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    free_throws_made INTEGER,
    free_throws_attempted INTEGER,
    plus_minus INTEGER,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, player_id)
);

-- Model Weights table
CREATE TABLE IF NOT EXISTS model_weights (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    weights BYTEA NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    trained_at TIMESTAMP WITH TIME ZONE,
    active BOOLEAN NOT NULL DEFAULT true,
    needs_training BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, (params->>'prediction_target'), version)
);

-- Model Performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    prediction_target VARCHAR(100) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_baseline BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System Logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_games_date ON games (date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games (status);
CREATE INDEX IF NOT EXISTS idx_games_season ON games (season_year);
CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams (conference);
CREATE INDEX IF NOT EXISTS idx_players_team ON players (team_id);
CREATE INDEX IF NOT EXISTS idx_game_stats_game ON game_stats (game_id);
CREATE INDEX IF NOT EXISTS idx_game_stats_player ON game_stats (player_id);
CREATE INDEX IF NOT EXISTS idx_model_weights_active ON model_weights (active);

-- Add model templates
INSERT INTO model_weights (model_name, model_type, params, weights, needs_training) VALUES
('RandomForest', 'classification', '{"n_estimators": 100, "max_depth": 10, "prediction_target": "moneyline"}'::jsonb, '\\x00'::bytea, true),
('GradientBoosting', 'regression', '{"n_estimators": 150, "learning_rate": 0.1, "prediction_target": "spread"}'::jsonb, '\\x00'::bytea, true)
ON CONFLICT DO NOTHING;
"@

Set-Content -Path $schemaFile -Value $schemaContent

try {
    $output = & "$pgBinPath\psql" -U postgres -d $DatabaseName -f $schemaFile 2>&1
    Write-Host "Database schema created successfully!" -ForegroundColor Green
}
catch {
    Write-Host "Failed to create database schema: $_" -ForegroundColor Red
    exit 1
}

# Create .env file with database configuration
Write-Host "\nCreating environment configuration..." -ForegroundColor Yellow

$envFile = "$PSScriptRoot\database.env"
$envContent = @"
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=$DatabaseName
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$PostgresPassword
BALLDONTLIE_API_KEY=d0e93357-b9b0-4a62-bed1-920eeab5db50
ODDS_API_KEY=12186f096bb2e6a9f9b472391323893d
"@

Set-Content -Path $envFile -Value $envContent

# Create a batch file to set environment variables
$batchFile = "$PSScriptRoot\set_env.bat"
$batchContent = @"
@echo off
echo Setting environment variables for NBA Prediction System...

set POSTGRES_HOST=localhost
set POSTGRES_PORT=5432
set POSTGRES_DB=$DatabaseName
set POSTGRES_USER=postgres
set POSTGRES_PASSWORD=$PostgresPassword
set BALLDONTLIE_API_KEY=d0e93357-b9b0-4a62-bed1-920eeab5db50
set ODDS_API_KEY=12186f096bb2e6a9f9b472391323893d

echo Environment variables set!
"@

Set-Content -Path $batchFile -Value $batchContent

Write-Host "\n===== INSTALLATION COMPLETED SUCCESSFULLY =====" -ForegroundColor Green
Write-Host "PostgreSQL has been installed and configured for your NBA Prediction System." -ForegroundColor Green
Write-Host "\nDatabase Configuration:" -ForegroundColor Yellow
Write-Host "  Host: localhost" -ForegroundColor White
Write-Host "  Port: 5432" -ForegroundColor White
Write-Host "  Database: $DatabaseName" -ForegroundColor White
Write-Host "  Username: postgres" -ForegroundColor White
Write-Host "  Password: $PostgresPassword" -ForegroundColor White

Write-Host "\nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'set_env.bat' to set environment variables" -ForegroundColor White
Write-Host "2. Import NBA data: python -m src.database.setup_production_db" -ForegroundColor White
Write-Host "3. Train models: python -m src.models.advanced_trainer" -ForegroundColor White
Write-Host "4. Start the API server: python -m src.api.server" -ForegroundColor White

Stop-Transcript
