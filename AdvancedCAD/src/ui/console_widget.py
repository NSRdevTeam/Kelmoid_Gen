"""
Console Widget
Logging and message display for AdvancedCAD
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat
from datetime import datetime


class ConsoleWidget(QWidget):
    """Console widget for displaying log messages"""
    
    message_logged = Signal(str, str)  # message, level
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the console UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title label
        title = QLabel("Console")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Text area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(200)
        
        # Console font
        font = QFont("Consolas", 9)
        font.setFixedPitch(True)
        self.text_display.setFont(font)
        
        # Console dark theme
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                selection-background-color: #264f78;
            }
        """)
        
        layout.addWidget(self.text_display)
        
        # Log startup message
        self.log_message("AdvancedCAD Console ready", "info")
    
    def log_message(self, message: str, level: str = "info"):
        """Log a message to the console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color map
        colors = {
            "info": "#d4d4d4",
            "warning": "#ffcc02",
            "error": "#f44747",
            "success": "#4ec9b0"
        }
        
        color = colors.get(level, "#d4d4d4")
        
        # Create color format (safe QTextCharFormat)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Normal)
        if level in ("warning", "error"):
            fmt.setFontWeight(QFont.Bold)
        
        # Build formatted message
        formatted_message = f"[{timestamp}] {message}\n"
        
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(formatted_message, fmt)
        self.text_display.setTextCursor(cursor)
        
        # Auto-scroll to bottom
        scrollbar = self.text_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Emit log signal
        self.message_logged.emit(message, level)
    
    def clear_console(self):
        """Clear the console contents"""
        self.text_display.clear()
        self.log_message("Console cleared", "info")
