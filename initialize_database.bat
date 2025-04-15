@echo off
echo Initializing NBA Prediction database...
echo.
psql -U postgres -p 5433 -d nba_prediction -f scripts/init_nba_schema.sql
echo.
echo Database initialization complete.
echo.
pause
