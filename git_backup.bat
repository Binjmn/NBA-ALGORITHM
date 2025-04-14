@echo off
REM NBA Betting Prediction System Git Backup Script
REM Created on: 2025-04-14

echo.
echo ===== NBA ALGORITHM GIT BACKUP UTILITY =====
echo.

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/downloads
    echo.
    pause
    exit /b 1
)

REM Check if we're in a git repository
if not exist ".git" (
    echo Error: Not in a git repository.
    echo Please run this script from the root of your project.
    echo.
    pause
    exit /b 1
)

REM Stage all changes
git add .

REM Show status of changes
echo.
echo Current changes to be committed:
echo -----------------------------------
git status --short
echo -----------------------------------
echo.

REM Get commit message from user
set /p COMMIT_MSG="Enter commit message (leave blank to cancel): "

REM Check if commit message is empty
if "x%COMMIT_MSG%" == "x" (
    echo.
    echo Commit cancelled. No changes were committed.
    echo.
    pause
    exit /b 0
)

REM Commit changes with the provided message
echo.
echo Committing changes with message: "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%"

REM Ask if the user wants to push changes
echo.
set /p PUSH_CHOICE="Push changes to remote repository? (Y/N): "

if /i "%PUSH_CHOICE%" == "Y" (
    echo.
    echo Pushing changes to remote repository...
    git push
    
    REM Check if push was successful
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Changes successfully pushed to remote repository.
    ) else (
        echo.
        echo Error: Failed to push changes to remote repository.
        echo Please check your internet connection and repository configuration.
    )
) else (
    echo.
    echo Changes committed locally but not pushed.
    echo Run 'git push' manually when you're ready to push changes.
)

echo.
echo Backup process completed.
echo.

pause
