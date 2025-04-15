@echo off
echo Starting NBA Prediction Training Pipeline with Virtual Environment...
echo.

:: Activate the virtual environment first
call .\venv\Scripts\activate.bat

:: Install required packages if needed
pip install lightgbm xgboost --quiet

:: Run the training pipeline
python -m src.training_pipeline

:: Deactivate virtual environment
call deactivate

echo.
echo Training process complete.
echo.
pause
