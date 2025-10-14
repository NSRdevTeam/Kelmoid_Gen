#!/usr/bin/env python3
"""
AdvancedCAD - Next-Generation 3D Modeling Software
Main application entry point
"""

import sys
import os
from pathlib import Path

# Add the source directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QIcon, QFont, QTextCharFormat


from ui.main_window import MainWindow
from core.config import AppConfig


def setup_application():
    """Setup application properties and settings"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AdvancedCAD")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AdvancedCAD")
    app.setOrganizationDomain("advancedcad.org")
    
    # Enable high DPI scaling
    #app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    #app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Set application icon
    icon_path = Path(__file__).parent / "resources" / "icons" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    return app


def main():
    """Main application function"""
    try:
        # Setup application
        app = setup_application()
        
        # Initialize configuration
        config = AppConfig()
        
        # Create and show main window
        main_window = MainWindow()
        main_window.show()
        
        # Start the application event loop
        return app.exec()
        
    except Exception as e:
        print(f"Error starting AdvancedCAD: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
