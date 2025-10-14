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
    from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                                 QHBoxLayout, QWidget, QPushButton, QLabel, QComboBox)
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
        
        # View mode selection
        control_layout.addWidget(QLabel("Display Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Solid", "Wireframe", "Transparent"
        ])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        control_layout.addWidget(self.mode_combo)
        
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
            if hasattr(self.viewport, 'camera_changed'):
                try:
                    self.viewport.camera_changed.connect(self.on_camera_changed)
                except Exception as e:
                    print(f"Could not connect camera signal: {e}")
        else:
            # Fallback widget
            self.viewport = QWidget()
            self.viewport.setStyleSheet("background-color: #333; color: white;")
            layout = QVBoxLayout(self.viewport)
            error_label = QLabel("3D Viewport not available\\nCheck dependencies")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)
        
        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.viewport, stretch=1)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        
        # Load initial shape
        self.create_shape("Cube")
        
    def create_shape(self, shape_name: str):
        """Create and display a 3D shape"""
        if not VIEWPORT_AVAILABLE:
            self.status_label.setText(f"Shape: {shape_name} (viewport not available)")
            return
            
        try:
            if shape_name == "Cube":
                mesh = PrimitiveGenerator.create_cube([10, 10, 10])
            elif shape_name == "Sphere":
                mesh = PrimitiveGenerator.create_sphere(radius=8, resolution=32)
            elif shape_name == "Cylinder":
                mesh = PrimitiveGenerator.create_cylinder(height=12, radius=6, resolution=24)
            elif shape_name == "Cone":
                mesh = PrimitiveGenerator.create_cylinder(height=12, radius=6, radius_top=0, resolution=24)
            elif shape_name == "Torus":
                mesh = PrimitiveGenerator.create_torus(major_radius=8, minor_radius=2, major_segments=24, minor_segments=12)
            else:
                mesh = PrimitiveGenerator.create_cube([10, 10, 10])
                
            self.current_mesh = mesh
            self.viewport.set_mesh(mesh)
            
            # Update status
            vertex_count = len(mesh.vertices) if mesh.vertices else 0
            face_count = len(mesh.faces) if mesh.faces else 0
            self.status_label.setText(f"{shape_name}: {vertex_count} vertices, {face_count} faces")
            
        except Exception as e:
            self.status_label.setText(f"Error creating {shape_name}: {str(e)}")
            print(f"Error creating shape {shape_name}: {e}")
    
    def on_shape_changed(self, shape_name: str):
        """Handle shape selection change"""
        self.create_shape(shape_name)
    
    def on_mode_changed(self, mode_name: str):
        """Handle display mode change"""
        if hasattr(self.viewport, 'set_display_mode'):
            mode_map = {
                "Solid": "solid",
                "Wireframe": "wireframe",
                "Transparent": "transparent"
            }
            self.viewport.set_display_mode(mode_map.get(mode_name, "solid"))
    
    def on_camera_changed(self):
        """Handle camera changes"""
        # Could update status or perform other actions
        pass
    
    def frame_all(self):
        """Frame the current object in the viewport"""
        if hasattr(self.viewport, 'keyPressEvent'):
            from PySide6.QtGui import QKeyEvent
            from PySide6.QtCore import Qt
            # Simulate F key press
            key_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_F, Qt.NoModifier)
            self.viewport.keyPressEvent(key_event)
    
    def reset_view(self):
        """Reset the camera to default view"""
        if hasattr(self.viewport, 'camera'):
            self.viewport.camera.distance = 50.0
            self.viewport.camera.rotation_x = -20.0
            self.viewport.camera.rotation_y = 45.0
            if hasattr(self.viewport.camera, 'center'):
                if hasattr(self.viewport.camera.center, 'setX'):
                    self.viewport.camera.center.setX(0)
                    self.viewport.camera.center.setY(0)
                    self.viewport.camera.center.setZ(0)
                else:
                    self.viewport.camera.center = [0, 0, 0]
            
            # Frame the current mesh if available
            if self.current_mesh:
                self.frame_all()


def main():
    """Main function to run the test"""
    print("AdvancedCAD 3D Viewport Test")
    print("=============================")
    
    # Check dependencies
    missing_deps = []
    
    if not PYSIDE6_AVAILABLE:
        missing_deps.append("PySide6")
    
    try:
        import OpenGL.GL
    except ImportError:
        missing_deps.append("PyOpenGL")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install PySide6 PyOpenGL numpy")
        print("\\nContinuing with limited functionality...")
    else:
        print("All dependencies available!")
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AdvancedCAD 3D Viewport Test")
    
    # Create and show window
    window = ViewportTestWindow()
    window.show()
    
    print("\\n3D Viewport Test Window opened.")
    print("Try the following:")
    print("1. Select different shapes from the dropdown")
    print("2. Change display modes (Solid/Wireframe/Transparent)")
    print("3. Use mouse to orbit (left drag), pan (right drag), zoom (wheel)")
    print("4. Press 'F' or click 'Frame All' to fit the object in view")
    print("5. Use keyboard shortcuts: 1=Solid, 2=Wireframe, 3=Transparent")
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
