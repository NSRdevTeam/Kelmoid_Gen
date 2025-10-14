#!/usr/bin/env python3
"""
Setup script for AdvancedCAD
"""

from pathlib import Path
import subprocess
import sys


def install_dependencies():
    """Install required dependencies"""
    print("Installing AdvancedCAD dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file)
        ])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_launcher():
    """Create launcher script"""
    launcher_content = '''@echo off
cd /d "%~dp0"
python main.py
pause'''
    
    launcher_path = Path(__file__).parent / "run_advancedcad.bat"
    
    try:
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        print(f"✓ Launcher created: {launcher_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create launcher: {e}")
        return False


def main():
    """Main setup function"""
    print("AdvancedCAD Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create launcher
    if not create_launcher():
        return False
    
    print("\\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("\\nTo run AdvancedCAD:")
    print("1. Double-click 'run_advancedcad.bat'")
    print("2. Or run: python main.py")
    print("\\nFor testing core functionality:")
    print("python test_app.py")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
