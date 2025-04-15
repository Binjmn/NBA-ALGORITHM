@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo NBA PREDICTION SYSTEM - DATABASE SETUP
echo ===============================================
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script requires administrator privileges.
    echo Please right-click and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

REM Create directories if they don't exist
if not exist "db_setup" mkdir db_setup
if not exist "logs" mkdir logs

echo Setting up PostgreSQL database for NBA Prediction System...
echo.

REM Step 1: Run PostgreSQL installer
echo Step 1: Installing and configuring PostgreSQL...
echo This may take a few minutes. Please be patient.
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0db_setup\install_postgres.ps1"
if %errorLevel% neq 0 (
    echo PostgreSQL installation failed. Please check the logs.
    echo.
    pause
    exit /b 1
)

REM Step 2: Initialize schema
echo.
echo Step 2: Initializing database schema...

set PGPASSWORD=ALGO123
psql -U postgres -d nba_prediction -f "%~dp0db_setup\init_nba_schema.sql"
if %errorLevel% neq 0 (
    echo Schema initialization failed. Please check the logs.
    echo.
    pause
    exit /b 1
)

REM Step 3: Import initial data from API
echo.
echo Step 3: Importing real NBA data from APIs...

call set_env.bat
python -m src.database.setup_production_db
if %errorLevel% neq 0 (
    echo WARNING: Data import encountered issues, but the database is setup correctly.
    echo You can retry data import later.
    echo.
)

echo.
echo ===============================================
echo DATABASE SETUP COMPLETED SUCCESSFULLY!
echo ===============================================
echo.
echo Your NBA Prediction System database is now ready.
echo.
echo Database details:
echo   Host: localhost
echo   Port: 5432
echo   Database: nba_prediction
echo   Username: postgres
echo   Password: ALGO123
echo.
echo Next steps:
echo   1. Train models with real NBA data: python -m src.models.advanced_trainer
echo   2. Start the API server: python -m src.api.server
echo   3. Open the dashboard: http://localhost:5000/dashboard
echo.

pause
