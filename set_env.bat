@echo off
echo Setting up NBA Prediction System environment variables...

:: Database configuration
set POSTGRES_HOST=localhost
set POSTGRES_PORT=5432
set POSTGRES_DB=nba_prediction
set POSTGRES_USER=postgres
set POSTGRES_PASSWORD=ALGO123

:: API Keys
set BALLDONTLIE_API_KEY=d0e93357-b9b0-4a62-bed1-920eeab5db50
set ODDS_API_KEY=12186f096bb2e6a9f9b472391323893d

:: Training configuration
set AUTO_TRAIN_ENABLED=true
set AUTO_TRAIN_INTERVAL=86400

echo Environment variables set successfully!
echo.
echo Database: %POSTGRES_HOST%:%POSTGRES_PORT%/%POSTGRES_DB%
echo BallDontLie API: Configured
echo The Odds API: Configured
echo.
