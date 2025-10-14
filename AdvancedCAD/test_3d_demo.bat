@echo off
title AdvancedCAD 3D Viewport Demo

echo ====================================
echo   AdvancedCAD 3D Viewport Demo
echo ====================================
echo.
echo This demo shows the 3D rendering capabilities
echo of AdvancedCAD with interactive camera controls.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
python -c "import PySide6, OpenGL.GL, numpy" >nul 2>&1
if errorlevel 1 (
    echo Some dependencies missing. Installing...
    python -m pip install PySide6 PyOpenGL PyOpenGL-accelerate numpy
    echo.
)

REM Set Python path
set PYTHONPATH=%~dp0src;%PYTHONPATH%

echo Starting 3D Viewport Demo...
echo.

REM Launch the demo
python "%~dp0test_viewport_demo.py"

echo.
echo Demo has closed.
pause
