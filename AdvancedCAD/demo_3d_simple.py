#!/usr/bin/env python3
"""
Simple 3D Demo for AdvancedCAD
Shows basic OpenGL rendering with our generated meshes
"""

import sys
import os
import math
sys.path.append('src')

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtGui import QSurfaceFormat
    
    import OpenGL.GL as gl
    import numpy as np
    
    from core.primitives import CubeGenerator, SphereGenerator
    
    DEPS_OK = True
    
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPS_OK = False


class Simple3DViewer(QOpenGLWidget):
    """Simple 3D viewer using OpenGL"""
    
    def __init__(self):
        super().__init__()
        
        # Set OpenGL format
        format = QSurfaceFormat()
        format.setDepthBufferSize(24)
        format.setSamples(4)
        self.setFormat(format)
        
        # Initialize variables
        self.rotation_x = 0
        self.rotation_y = 0
        self.mesh_data = None
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)  # ~60 FPS
        
    def set_mesh(self, vertices, faces):
        """Set the mesh to display"""
        self.mesh_data = (vertices, faces)
        self.update()
        
    def animate(self):
        """Animate the rotation"""
        self.rotation_y += 1
        self.update()
        
    def initializeGL(self):
        """Initialize OpenGL"""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        
        # Set up light
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        
        # Set material
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, [0.7, 0.7, 0.9, 1.0])
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 50.0)
        
        # Set clear color
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        
    def resizeGL(self, width, height):
        """Handle resize"""
        gl.glViewport(0, 0, width, height)
        
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        
        aspect = width / height if height > 0 else 1
        fov = 45.0
        near = 0.1
        far = 100.0
        
        # Perspective projection
        f = 1.0 / math.tan(fov * math.pi / 360.0)
        gl.glLoadMatrixf([
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far+near)/(near-far), (2*far*near)/(near-far),
            0, 0, -1, 0
        ])
        
    def paintGL(self):
        """Render the scene"""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        # Position camera
        gl.glTranslatef(0, 0, -15)
        gl.glRotatef(self.rotation_x, 1, 0, 0)
        gl.glRotatef(self.rotation_y, 0, 1, 0)
        
        # Draw coordinate axes
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1, 0, 0)  # X-axis red
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(5, 0, 0)
        gl.glColor3f(0, 1, 0)  # Y-axis green
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 5, 0)
        gl.glColor3f(0, 0, 1)  # Z-axis blue
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, 5)
        gl.glEnd()
        
        # Draw mesh if available
        if self.mesh_data:
            vertices, faces = self.mesh_data
            
            gl.glColor3f(0.7, 0.7, 0.9)
            gl.glBegin(gl.GL_TRIANGLES)
            
            for face in faces:
                for vertex_idx in face:
                    vertex = vertices[vertex_idx]
                    gl.glVertex3f(vertex[0], vertex[1], vertex[2])
                    
            gl.glEnd()


class MainWindow(QMainWindow):
    """Main demo window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AdvancedCAD - Simple 3D Demo")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout(central_widget)
        
        # Info label
        info = QLabel("AdvancedCAD 3D Rendering Demo\\nShowing OpenGL rendering of generated meshes")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(info)
        
        # 3D viewer
        self.viewer = Simple3DViewer()
        layout.addWidget(self.viewer)
        
        # Controls
        controls = QHBoxLayout()
        
        cube_btn = QPushButton("Show Cube")
        cube_btn.clicked.connect(self.show_cube)
        controls.addWidget(cube_btn)
        
        sphere_btn = QPushButton("Show Sphere")
        sphere_btn.clicked.connect(self.show_sphere)
        controls.addWidget(sphere_btn)
        
        layout.addLayout(controls)
        
        # Status
        self.status = QLabel("Click buttons to generate and display 3D shapes")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)
        
        # Show initial cube
        self.show_cube()
        
    def show_cube(self):
        """Generate and display a cube"""
        try:
            cube_gen = CubeGenerator()
            mesh = cube_gen.generate({'size': [8, 8, 8]})
            
            self.viewer.set_mesh(mesh.vertices, mesh.faces)
            self.status.setText(f"Cube: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
        except Exception as e:
            self.status.setText(f"Error generating cube: {e}")
            
    def show_sphere(self):
        """Generate and display a sphere"""
        try:
            sphere_gen = SphereGenerator()
            mesh = sphere_gen.generate({'radius': 6, 'resolution': 32})
            
            self.viewer.set_mesh(mesh.vertices, mesh.faces)
            self.status.setText(f"Sphere: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
        except Exception as e:
            self.status.setText(f"Error generating sphere: {e}")


def main():
    """Main function"""
    if not DEPS_OK:
        print("Cannot run demo - missing dependencies")
        print("Install with: pip install PySide6 PyOpenGL numpy")
        return 1
        
    print("AdvancedCAD Simple 3D Demo")
    print("==========================")
    print("This demo shows:")
    print("- 3D mesh generation from our primitives")
    print("- OpenGL rendering with lighting")
    print("- Real-time animation")
    print("- Interactive shape switching")
    print()
    
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    print("Demo window opened. Try clicking the buttons to switch shapes!")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
