@echo off
REM NBA Prediction System Control Script
REM This script provides a user-friendly way to control the NBA Prediction System

REM Check for docker-compose existence
WHERE docker-compose >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Docker Compose is not installed or not in your PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    exit /b 1
)

REM Parse command
IF "%1" == "" (
    GOTO showHelp
) ELSE IF "%1" == "start" (
    GOTO startSystem
) ELSE IF "%1" == "stop" (
    GOTO stopSystem
) ELSE IF "%1" == "restart" (
    GOTO restartSystem
) ELSE IF "%1" == "status" (
    GOTO showStatus
) ELSE IF "%1" == "logs" (
    GOTO showLogs
) ELSE (
    GOTO showHelp
)

:startSystem
echo Starting NBA Prediction System...
docker-compose up -d
echo.
echo System is now starting! 
echo You can view the dashboard at http://localhost:8080
echo (It may take a minute to fully initialize)
GOTO end

:stopSystem
echo Stopping NBA Prediction System...
docker-compose down
echo System has been stopped.
GOTO end

:restartSystem
echo Restarting NBA Prediction System...
docker-compose restart
echo System has been restarted.
echo You can view the dashboard at http://localhost:8080
GOTO end

:showStatus
echo NBA Prediction System Status:
echo ----------------------------
docker-compose ps
echo.
echo For more detailed status, visit http://localhost:8080
GOTO end

:showLogs
IF "%2" == "" (
    echo Showing logs for all services:
    docker-compose logs --tail=100
) ELSE (
    echo Showing logs for %2:
    docker-compose logs --tail=100 %2
)
GOTO end

:showHelp
echo NBA Prediction System - User-friendly Control Script
echo ---------------------------------------------------
echo Usage: nba-control COMMAND
echo.
echo Commands:
echo   start       Start the NBA Prediction System
echo   stop        Stop the NBA Prediction System
echo   restart     Restart the NBA Prediction System
echo   status      Show the status of all services
echo   logs        Show last 100 lines of logs (all services)
echo   logs SERVICE Show logs for a specific service
echo.
echo Examples:
echo   nba-control start
echo   nba-control logs nba-prediction
echo.
echo After starting, you can access the web dashboard at:
echo http://localhost:8080

:end
