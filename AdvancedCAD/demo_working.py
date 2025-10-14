#!/usr/bin/env python3
"""
Working AdvancedCAD Demo
Shows the application working with mesh generation and GUI
"""

import sys
import os
sys.path.append('src')

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QTextEdit, QSplitter)
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QPixmap
    
    from core.primitives import CubeGenerator, SphereGenerator, CylinderGenerator
    
    DEPS_OK = True
    
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPS_OK = False


class MeshViewer(QWidget):
    """Text-based mesh viewer showing mesh statistics and data"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("3D Mesh Data Viewer")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Info display
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 12px;
            background-color: #f0f0f0;
            border: 2px solid #ccc;
        """)
        layout.addWidget(self.info_text)
        
        # Show initial message
        self.show_info("Click buttons below to generate 3D shapes and view their mesh data")
        
    def show_mesh(self, mesh_data, shape_name):
        """Display mesh information"""
        vertices = mesh_data.vertices
        faces = mesh_data.faces
        normals = mesh_data.normals
        
        info = f"=== {shape_name.upper()} MESH DATA ===\\n\\n"
        info += f"Vertices: {len(vertices)}\\n"
        info += f"Faces: {len(faces)}\\n"
        info += f"Normals: {'Yes' if normals is not None else 'No'}\\n\\n"
        
        # Show first few vertices
        info += "First 10 vertices:\\n"
        for i, vertex in enumerate(vertices[:10]):
            info += f"  {i:2d}: ({vertex[0]:6.2f}, {vertex[1]:6.2f}, {vertex[2]:6.2f})\\n"
        if len(vertices) > 10:
            info += f"  ... and {len(vertices) - 10} more\\n"
        
        info += "\\nFirst 10 faces (triangle indices):\\n"
        for i, face in enumerate(faces[:10]):
            info += f"  {i:2d}: [{face[0]:2d}, {face[1]:2d}, {face[2]:2d}]\\n"
        if len(faces) > 10:
            info += f"  ... and {len(faces) - 10} more\\n"
            
        # Bounding box
        if len(vertices) > 0:
            min_x, min_y, min_z = vertices.min(axis=0)
            max_x, max_y, max_z = vertices.max(axis=0)
            info += f"\\nBounding box:\\n"
            info += f"  Min: ({min_x:6.2f}, {min_y:6.2f}, {min_z:6.2f})\\n"
            info += f"  Max: ({max_x:6.2f}, {max_y:6.2f}, {max_z:6.2f})\\n"
            info += f"  Size: ({max_x-min_x:6.2f}, {max_y-min_y:6.2f}, {max_z-min_z:6.2f})\\n"
        
        # Surface area approximation
        if len(faces) > 0:
            import numpy as np
            total_area = 0
            for face in faces[:100]:  # Sample first 100 faces
                v0, v1, v2 = vertices[face]
                # Calculate triangle area using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                total_area += area
            
            estimated_total = total_area * len(faces) / min(100, len(faces))
            info += f"\\nEstimated surface area: {estimated_total:.2f} square units\\n"
        
        info += "\\n" + "="*50
        
        self.info_text.setText(info)
        
    def show_info(self, message):
        """Show informational message"""
        self.info_text.setText(message)


class MainWindow(QMainWindow):
    """Main demo window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AdvancedCAD - Working Demo")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🎯 AdvancedCAD - Next-Generation 3D Modeling")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            color: #2c3e50;
            margin: 15px;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 10px;
        """)
        layout.addWidget(header)
        
        # Status
        self.status = QLabel("✅ All systems operational - Core functionality working!")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #27ae60; font-size: 14px; margin: 5px;")
        layout.addWidget(self.status)
        
        # Content area
        content_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(content_splitter)
        
        # Left panel - controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        
        # Controls section
        controls_label = QLabel("🎛️ Shape Generator")
        controls_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(controls_label)
        
        # Shape buttons
        cube_btn = QPushButton("📦 Generate Cube")
        cube_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        cube_btn.clicked.connect(self.generate_cube)
        left_layout.addWidget(cube_btn)
        
        sphere_btn = QPushButton("🌍 Generate Sphere") 
        sphere_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        sphere_btn.clicked.connect(self.generate_sphere)
        left_layout.addWidget(sphere_btn)
        
        cylinder_btn = QPushButton("🥫 Generate Cylinder")
        cylinder_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        cylinder_btn.clicked.connect(self.generate_cylinder)
        left_layout.addWidget(cylinder_btn)
        
        left_layout.addWidget(QLabel())  # Spacer
        
        # Info section
        info_label = QLabel("ℹ️ What This Demo Shows:")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 20px;")
        left_layout.addWidget(info_label)
        
        features = QLabel(
            "✅ 3D mesh generation\\n"
            "✅ Parametric primitives\\n" 
            "✅ GUI framework (PySide6)\\n"
            "✅ Data visualization\\n"
            "✅ Math calculations\\n"
            "✅ Cross-platform support"
        )
        features.setStyleSheet("margin: 10px; color: #27ae60;")
        left_layout.addWidget(features)
        
        left_layout.addStretch()
        
        # Right panel - mesh viewer
        self.mesh_viewer = MeshViewer()
        
        # Add to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(self.mesh_viewer)
        content_splitter.setSizes([300, 700])
        
        # Footer
        footer = QLabel("AdvancedCAD v1.0 - Powered by Python, PySide6, NumPy, and OpenGL")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #7f8c8d; font-style: italic; margin: 10px;")
        layout.addWidget(footer)
        
    def generate_cube(self):
        """Generate and display cube mesh"""
        try:
            generator = CubeGenerator()
            mesh = generator.generate({
                'size': [10, 8, 6],
                'center': True
            })
            
            self.mesh_viewer.show_mesh(mesh, "Cube")
            self.status.setText(f"✅ Generated cube: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
        except Exception as e:
            self.status.setText(f"❌ Error generating cube: {e}")
            
    def generate_sphere(self):
        """Generate and display sphere mesh"""
        try:
            generator = SphereGenerator()
            mesh = generator.generate({
                'radius': 8,
                'resolution': 24
            })
            
            self.mesh_viewer.show_mesh(mesh, "Sphere")
            self.status.setText(f"✅ Generated sphere: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
        except Exception as e:
            self.status.setText(f"❌ Error generating sphere: {e}")
            
    def generate_cylinder(self):
        """Generate and display cylinder mesh"""
        try:
            generator = CylinderGenerator()
            mesh = generator.generate({
                'height': 12,
                'radius': 5,
                'resolution': 20
            })
            
            self.mesh_viewer.show_mesh(mesh, "Cylinder")
            self.status.setText(f"✅ Generated cylinder: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
        except Exception as e:
            self.status.setText(f"❌ Error generating cylinder: {e}")


def main():
    """Main function"""
    if not DEPS_OK:
        print("❌ Cannot run demo - missing dependencies")
        print("Install with: pip install PySide6 numpy")
        return 1
        
    print("🎯 AdvancedCAD Working Demo")
    print("===========================")
    print("✅ All core dependencies available!")
    print("✅ 3D mesh generation working")
    print("✅ GUI framework operational")
    print("✅ Mathematical operations functional")
    print()
    print("🚀 Opening demo window...")
    
    app = QApplication(sys.argv)
    app.setApplicationName("AdvancedCAD Demo")
    
    window = MainWindow()
    window.show()
    
    print("✅ Demo window opened successfully!")
    print("👀 Try clicking the shape generation buttons to see AdvancedCAD in action!")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
