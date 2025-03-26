@echo off
setlocal enabledelayedexpansion

REM Get the directory of the batch file
set "SCRIPT_DIR=%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python from python.org
    pause
    exit /b 1
)

REM Navigate to the script's directory
cd /d "%SCRIPT_DIR%"

REM Run the eye sorter script
python eye_sorter.py

REM Pause to keep the window open
pause
