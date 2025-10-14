"""
Project Explorer Widget
File browser and project management
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class ProjectExplorer(QWidget):
    """Project explorer widget for managing files"""
    
    file_selected = Signal(str)  # Emitted when a file is selected
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the project explorer UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Project Explorer")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Placeholder content
        placeholder = QLabel("Project management\\nnot yet implemented\\n\\nThis will show:\\n• File browser\\n• Recent projects\\n• Templates")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            color: #888;
            padding: 20px;
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px;
        """)
        layout.addWidget(placeholder)
        
        layout.addStretch()
