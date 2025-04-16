@echo off
echo Starting NBA Prediction Training Pipeline with Virtual Environment...
echo.
echo [Info] Using automatic season detection - the pipeline will use the current NBA season
echo.

:: Activate the virtual environment first
call .\venv\Scripts\activate.bat

:: Install required packages if needed
pip install lightgbm xgboost --quiet

:: Run the training pipeline with auto-detection enabled and pass all additional arguments
python -m src.training_pipeline --auto-detect-season %*

:: Deactivate virtual environment
call deactivate

echo.
echo Training process complete.
echo.
pause
