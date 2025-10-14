#!/usr/bin/env python3
"""
Test script for AdvancedCAD 3D Viewport
Demonstrates the OpenGL-based 3D rendering capabilities
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QComboBox
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    PYSIDE6_AVAILABLE = True
except ImportError:
    print("PySide6 not available. Please install with: pip install PySide6")
    PYSIDE6_AVAILABLE = False
    sys.exit(1)

try:
    from ui.viewport_3d import Viewport3D
    from core.primitives import PrimitiveGenerator
    VIEWPORT_AVAILABLE = True
except ImportError as e:
    print(f"Could not import viewport or primitives: {e}")
    VIEWPORT_AVAILABLE = False


class ViewportTestWindow(QMainWindow):
    """Test window for demonstrating 3D viewport functionality"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_mesh = None
        
    def setup_ui(self):
        """Setup the test window UI"""
        self.setWindowTitle("AdvancedCAD 3D Viewport Test")
        self.setMinimumSize(1000, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Control panel
        control_panel = QWidget()
        control_panel.setMaximumWidth(200)
        control_panel.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        control_layout = QVBoxLayout(control_panel)
        
        # Title
        title_label = QLabel("3D Viewport Test")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title_label)
        
        control_layout.addWidget(QLabel(""))  # Spacer
        
        # Shape selection
        control_layout.addWidget(QLabel("Select Shape:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems([
            "Cube", "Sphere", "Cylinder", "Cone", "Torus"
        ])
        self.shape_combo.currentTextChanged.connect(self.on_shape_changed)
        control_layout.addWidget(self.shape_combo)
        
        control_layout.addWidget(QLabel(""))  # Spacer
        
        # View mode selection\n        control_layout.addWidget(QLabel("Display Mode:"))\n        self.mode_combo = QComboBox()\n        self.mode_combo.addItems([\n            "Solid", "Wireframe", "Transparent"\n        ])\n        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)\n        control_layout.addWidget(self.mode_combo)
        
        control_layout.addWidget(QLabel(""))  # Spacer
        
        # Action buttons
        frame_button = QPushButton("Frame All (F)")
        frame_button.clicked.connect(self.frame_all)
        control_layout.addWidget(frame_button)
        
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_view)
        control_layout.addWidget(reset_button)
        
        control_layout.addStretch()  # Push everything to top
        
        # Instructions
        instructions = QLabel(
            "Mouse Controls:\\n"
            "• Left drag: Orbit\\n"
            "• Right drag: Pan\\n"
            "• Wheel: Zoom\\n\\n"
            "Keyboard:\\n"
            "• F: Frame all\\n"
            "• 1: Solid mode\\n"
            "• 2: Wireframe\\n"
            "• 3: Transparent"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 10px; color: #666;")
        control_layout.addWidget(instructions)
        
        # 3D Viewport
        if VIEWPORT_AVAILABLE:
            self.viewport = Viewport3D()
            
            # Connect viewport signals
            if hasattr(self.viewport, 'camera_changed'):\n                try:\n                    self.viewport.camera_changed.connect(self.on_camera_changed)\n                except Exception as e:\n                    print(f\"Could not connect camera signal: {e}\")\n        else:\n            # Fallback widget\n            self.viewport = QWidget()\n            self.viewport.setStyleSheet(\"background-color: #333; color: white;\")\n            layout = QVBoxLayout(self.viewport)\n            error_label = QLabel(\"3D Viewport not available\\nCheck dependencies\")\n            error_label.setAlignment(Qt.AlignCenter)\n            layout.addWidget(error_label)\n        \n        # Add widgets to main layout\n        main_layout.addWidget(control_panel)\n        main_layout.addWidget(self.viewport, stretch=1)\n        \n        # Status bar\n        self.status_label = QLabel(\"Ready\")\n        self.statusBar().addWidget(self.status_label)\n        \n        # Load initial shape\n        self.create_shape(\"Cube\")\n        \n    def create_shape(self, shape_name: str):\n        \"\"\"Create and display a 3D shape\"\"\"\n        if not VIEWPORT_AVAILABLE:\n            self.status_label.setText(f\"Shape: {shape_name} (viewport not available)\")\n            return\n            \n        try:\n            if shape_name == \"Cube\":\n                mesh = PrimitiveGenerator.create_cube([10, 10, 10])\n            elif shape_name == \"Sphere\":\n                mesh = PrimitiveGenerator.create_sphere(radius=8, resolution=32)\n            elif shape_name == \"Cylinder\":\n                mesh = PrimitiveGenerator.create_cylinder(height=12, radius=6, resolution=24)\n            elif shape_name == \"Cone\":\n                mesh = PrimitiveGenerator.create_cylinder(height=12, radius=6, radius_top=0, resolution=24)\n            elif shape_name == \"Torus\":\n                mesh = PrimitiveGenerator.create_torus(major_radius=8, minor_radius=2, major_segments=24, minor_segments=12)\n            else:\n                mesh = PrimitiveGenerator.create_cube([10, 10, 10])\n                \n            self.current_mesh = mesh\n            self.viewport.set_mesh(mesh)\n            \n            # Update status\n            vertex_count = len(mesh.vertices) if mesh.vertices else 0\n            face_count = len(mesh.faces) if mesh.faces else 0\n            self.status_label.setText(f\"{shape_name}: {vertex_count} vertices, {face_count} faces\")\n            \n        except Exception as e:\n            self.status_label.setText(f\"Error creating {shape_name}: {str(e)}\")\n            print(f\"Error creating shape {shape_name}: {e}\")\n    \n    def on_shape_changed(self, shape_name: str):\n        \"\"\"Handle shape selection change\"\"\"\n        self.create_shape(shape_name)\n    \n    def on_mode_changed(self, mode_name: str):\n        \"\"\"Handle display mode change\"\"\"\n        if hasattr(self.viewport, 'set_display_mode'):\n            mode_map = {\n                \"Solid\": \"solid\",\n                \"Wireframe\": \"wireframe\",\n                \"Transparent\": \"transparent\"\n            }\n            self.viewport.set_display_mode(mode_map.get(mode_name, \"solid\"))\n    \n    def on_camera_changed(self):\n        \"\"\"Handle camera changes\"\"\"\n        # Could update status or perform other actions\n        pass\n    \n    def frame_all(self):\n        \"\"\"Frame the current object in the viewport\"\"\"\n        if hasattr(self.viewport, 'keyPressEvent'):\n            from PySide6.QtGui import QKeyEvent\n            from PySide6.QtCore import Qt\n            # Simulate F key press\n            key_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_F, Qt.NoModifier)\n            self.viewport.keyPressEvent(key_event)\n    \n    def reset_view(self):\n        \"\"\"Reset the camera to default view\"\"\"\n        if hasattr(self.viewport, 'camera'):\n            self.viewport.camera.distance = 50.0\n            self.viewport.camera.rotation_x = -20.0\n            self.viewport.camera.rotation_y = 45.0\n            if hasattr(self.viewport.camera, 'center'):\n                if hasattr(self.viewport.camera.center, 'setX'):\n                    self.viewport.camera.center.setX(0)\n                    self.viewport.camera.center.setY(0)\n                    self.viewport.camera.center.setZ(0)\n                else:\n                    self.viewport.camera.center = [0, 0, 0]\n            \n            # Frame the current mesh if available\n            if self.current_mesh:\n                self.frame_all()


def main():
    \"\"\"Main function to run the test\"\"\"\n    print(\"AdvancedCAD 3D Viewport Test\")\n    print(\"=============================\")\n    \n    # Check dependencies\n    missing_deps = []\n    \n    if not PYSIDE6_AVAILABLE:\n        missing_deps.append(\"PySide6\")\n    \n    try:\n        import OpenGL.GL\n    except ImportError:\n        missing_deps.append(\"PyOpenGL\")\n    \n    try:\n        import numpy\n    except ImportError:\n        missing_deps.append(\"numpy\")\n    \n    if missing_deps:\n        print(f\"Missing dependencies: {', '.join(missing_deps)}\")\n        print(\"Install with: pip install PySide6 PyOpenGL numpy\")\n        print(\"\\nContinuing with limited functionality...\")\n    else:\n        print(\"All dependencies available!\")\n    \n    # Create application\n    app = QApplication(sys.argv)\n    app.setApplicationName(\"AdvancedCAD 3D Viewport Test\")\n    \n    # Create and show window\n    window = ViewportTestWindow()\n    window.show()\n    \n    print(\"\\n3D Viewport Test Window opened.\")\n    print(\"Try the following:\")\n    print(\"1. Select different shapes from the dropdown\")\n    print(\"2. Change display modes (Solid/Wireframe/Transparent)\")\n    print(\"3. Use mouse to orbit (left drag), pan (right drag), zoom (wheel)\")\n    print(\"4. Press 'F' or click 'Frame All' to fit the object in view\")\n    print(\"5. Use keyboard shortcuts: 1=Solid, 2=Wireframe, 3=Transparent\")\n    \n    # Run application\n    sys.exit(app.exec())


if __name__ == \"__main__\":\n    main()
