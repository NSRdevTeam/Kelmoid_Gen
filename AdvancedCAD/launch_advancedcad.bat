@echo off
title AdvancedCAD Launcher

echo ====================================
echo      AdvancedCAD Launcher
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Install dependencies if needed
echo Checking dependencies...
python -c "import PySide6, OpenGL.GL, numpy" >nul 2>&1
if errorlevel 1 (
    echo Some dependencies missing. Installing...
    python -m pip install PySide6 PyOpenGL PyOpenGL-accelerate numpy
    echo.
)

REM Set Python path to include src directory
set PYTHONPATH=%~dp0src;%PYTHONPATH%

echo Starting AdvancedCAD...
echo.

REM Try to launch the main application
if exist "%~dp0src\main.py" (
    python "%~dp0src\main.py"
) else if exist "%~dp0main.py" (
    python "%~dp0main.py"
) else (
    echo ERROR: Cannot find main.py
    echo Please ensure you're running this from the AdvancedCAD directory.
)

echo.
echo AdvancedCAD has closed.
pause
