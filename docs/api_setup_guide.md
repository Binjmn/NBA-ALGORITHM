# NBA Prediction System - API Setup Guide

## Overview

This guide explains how to properly set up and run the NBA Prediction System API server with the dashboard UI. The system requires a PostgreSQL database for storing model data, but includes a setup script to initialize the database with demo data.

## Prerequisites

- Python 3.8+ installed
- PostgreSQL 12+ database server installed and running
- NBA Prediction System codebase
- Required Python packages installed

## Step 1: Install Required Dependencies

Ensure you have the necessary Python packages installed:

```powershell
pip install flask flask-cors psycopg2-binary apscheduler pandas scikit-learn xgboost lightgbm
```

## Step 2: Configure Database Connection

The API server connects to a PostgreSQL database using environment variables:

### Using PowerShell:

```powershell
# Database connection
$env:POSTGRES_HOST="localhost"
$env:POSTGRES_PORT="5432"
$env:POSTGRES_DB="nba_prediction"
$env:POSTGRES_USER="postgres"
$env:POSTGRES_PASSWORD="postgres"

# API configuration
$env:API_TOKEN="your_secure_token_here"
$env:API_HOST="0.0.0.0"
$env:API_PORT="5000"
```

### Using Command Prompt:

```cmd
# Database connection
set POSTGRES_HOST=localhost
set POSTGRES_PORT=5432
set POSTGRES_DB=nba_prediction
set POSTGRES_USER=postgres
set POSTGRES_PASSWORD=postgres

# API configuration
set API_TOKEN=your_secure_token_here
set API_HOST=0.0.0.0
set API_PORT=5000
```

## Step 3: Set Up the Database

The system includes a script to set up your PostgreSQL database with the required tables and sample data. Run:

```powershell
python -m src.database.setup_demo_db
```

This will:
1. Create all required database tables
2. Insert sample model configurations
3. Generate performance metrics data for the dashboard

If you need to recreate the demo data, use the `--force` flag:

```powershell
python -m src.database.setup_demo_db --force
```

## Step 4: Start the API Server

Once the database is set up, start the API server:

```powershell
python -m src.api.server
```

You should see output similar to:

```
* Starting NBA Prediction System API Server *
* Dashboard available at: http://localhost:5000/dashboard
* API endpoints available at: http://localhost:5000/api/*
* Press Ctrl+C to stop the server *

 * Serving Flask app 'server'
 * Running on http://0.0.0.0:5000
```

## Step 5: Access the Dashboard

Open your web browser and navigate to:

```
http://localhost:5000/dashboard
```

You should see the performance dashboard with charts showing model accuracy trends and performance metrics.

## Troubleshooting

### Database Connection Issues

If you see database connection errors:

1. Check that PostgreSQL is running on your system
2. Verify your database connection environment variables
3. Ensure the database exists (create it manually if needed):
   ```sql
   CREATE DATABASE nba_prediction;
   ```
4. If needed, run the setup script with the `--force` flag

### API Server Won't Start

1. Check for port conflicts - change the `API_PORT` environment variable if port 5000 is in use
2. Ensure the `logs` directory exists in the project root
3. Verify all required Python packages are installed

### Dashboard Not Loading Properly

1. Check the browser console for JavaScript errors
2. Clear your browser cache and reload
3. Ensure the API server started without errors
4. Check that static files are being served correctly

## Testing the API

To test if the API is working correctly:

```powershell
python -m tests.test_api --host localhost --port 5000
```

For dashboard integration testing:

```powershell
python -m tests.test_dashboard_integration --host localhost --port 5000
```

## Production Deployment

For production deployments, consider:

1. Using a proper WSGI server like Gunicorn
2. Setting up a reverse proxy with Nginx
3. Configuring HTTPS for secure connections
4. Using a production-grade PostgreSQL setup with proper backups
5. Setting a strong API token for authentication
