"""
AdvancedCAD 3D Viewport Widget
Modern OpenGL-based 3D rendering with interactive camera controls
"""

import sys
import math
import numpy as np
from typing import List, Optional, Tuple

# Try to import PySide6 core/gui/widgets independently of OpenGL widgets
try:
    from PySide6.QtCore import Qt, Signal, QTimer, QPoint
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction, QLabel
    from PySide6.QtGui import QMatrix4x4, QVector3D, QQuaternion, QSurfaceFormat
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    QWidget = object
    Signal = lambda *args: None

# Try to import QOpenGLWidget separately (may not be available on all builds)
HAVE_QOPENGLWIDGET = False
if PYSIDE6_AVAILABLE:
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
        HAVE_QOPENGLWIDGET = True
    except ImportError:
        QOpenGLWidget = None
        HAVE_QOPENGLWIDGET = False

try:
    import OpenGL.GL as gl
    import OpenGL.arrays.vbo as glvbo
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    gl = None
    glvbo = None

# Import mesh data structure
try:
    from core.primitives import Mesh
except ImportError:
    # Fallback if core module not available
    Mesh = None


class Camera3D:
    """3D Camera with orbit, pan, and zoom controls"""
    
    def __init__(self):
        self.distance = 50.0
        self.rotation_x = -20.0  # Pitch
        self.rotation_y = 45.0   # Yaw
        self.center = QVector3D(0, 0, 0) if PYSIDE6_AVAILABLE else [0, 0, 0]
        self.fov = 45.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
        
    def get_view_matrix(self):
        """Get the view matrix for the camera"""
        if not PYSIDE6_AVAILABLE:
            return None
        view = QMatrix4x4()
        view.translate(0, 0, -self.distance)
        view.rotate(self.rotation_x, 1, 0, 0)
        view.rotate(self.rotation_y, 0, 1, 0)
        if hasattr(self.center, 'x'):
            view.translate(-self.center)
        else:
            view.translate(-self.center[0], -self.center[1], -self.center[2])
        return view
        
    def get_projection_matrix(self, aspect_ratio: float):
        """Get the projection matrix for the camera"""
        if not PYSIDE6_AVAILABLE:
            return None
        projection = QMatrix4x4()
        projection.perspective(self.fov, aspect_ratio, self.near_plane, self.far_plane)
        return projection
        
    def orbit(self, delta_x: float, delta_y: float):
        """Orbit the camera around the center point"""
        self.rotation_y += delta_x * 0.5
        self.rotation_x += delta_y * 0.5
        
        # Clamp pitch to prevent flipping
        self.rotation_x = max(-89.0, min(89.0, self.rotation_x))
        
    def pan(self, delta_x: float, delta_y: float):
        """Pan the camera (move center point)"""
        if not PYSIDE6_AVAILABLE:
            return
        # Convert screen space movement to world space
        sensitivity = self.distance * 0.001
        right = QVector3D(1, 0, 0)
        up = QVector3D(0, 1, 0)
        
        # Apply camera rotation to movement vectors
        rotation = QQuaternion.fromEulerAngles(self.rotation_x, self.rotation_y, 0)
        right = rotation.rotatedVector(right)
        up = rotation.rotatedVector(up)
        
        self.center += right * delta_x * sensitivity
        self.center += up * delta_y * sensitivity
        
    def zoom(self, delta: float):
        """Zoom the camera in/out"""
        zoom_speed = self.distance * 0.1
        self.distance = max(1.0, self.distance + delta * zoom_speed)
        
    def frame_bounds(self, min_point, max_point):
        """Frame the camera to show the given bounds"""
        if not PYSIDE6_AVAILABLE:
            return
        # Calculate center and size of bounds
        center = (min_point + max_point) * 0.5
        size = max_point - min_point
        max_dimension = max(size.x(), size.y(), size.z())
        
        # Set camera to show the entire object
        self.center = center
        self.distance = max_dimension * 2.0


class MeshRenderer:
    """OpenGL mesh renderer with multiple display modes"""
    
    def __init__(self):
        self.vertex_buffer = None
        self.index_buffer = None
        self.normal_buffer = None
        self.vertex_count = 0
        self.index_count = 0
        self.display_mode = 'solid'  # 'solid', 'wireframe', 'transparent'
        
    def upload_mesh(self, mesh):
        """Upload mesh data to GPU buffers"""
        if not OPENGL_AVAILABLE or mesh is None or not hasattr(mesh, 'vertices'):
            return
            
        # Convert mesh data to OpenGL format
        vertices = []
        normals = []
        indices = []
        
        # Flatten vertex data
        for vertex in mesh.vertices:
            vertices.extend(vertex)
            
        # Ensure we have normals
        if hasattr(mesh, 'normals') and mesh.normals and len(mesh.normals) == len(mesh.vertices):
            for normal in mesh.normals:
                normals.extend(normal)
        else:
            # Calculate simple normals if not provided
            if hasattr(mesh, 'calculate_normals'):
                mesh.calculate_normals()
                for normal in mesh.normals:
                    normals.extend(normal)
            else:
                # Default normals pointing up
                for _ in mesh.vertices:
                    normals.extend([0, 0, 1])
                
        # Flatten face indices
        for face in mesh.faces:
            if len(face) == 3:  # Triangle
                indices.extend(face)
            elif len(face) == 4:  # Quad - split into two triangles
                indices.extend([face[0], face[1], face[2]])
                indices.extend([face[0], face[2], face[3]])
                
        # Create vertex buffer objects
        try:
            vertices_array = np.array(vertices, dtype=np.float32)
            normals_array = np.array(normals, dtype=np.float32)
            indices_array = np.array(indices, dtype=np.uint32)
            
            # Clean up existing buffers
            self.cleanup()
            
            # Upload to GPU
            self.vertex_buffer = glvbo.VBO(vertices_array)
            self.normal_buffer = glvbo.VBO(normals_array)
            self.index_buffer = glvbo.VBO(indices_array, target=gl.GL_ELEMENT_ARRAY_BUFFER)
            
            self.vertex_count = len(vertices) // 3
            self.index_count = len(indices)
        except Exception as e:
            print(f"Error uploading mesh: {e}")
            
    def render(self, view_matrix, projection_matrix):
        """Render the mesh with the given matrices"""
        if not OPENGL_AVAILABLE or not self.vertex_buffer:
            return
            
        try:
            # Set up OpenGL state
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_CULL_FACE)
            
            if self.display_mode == 'transparent':
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                
            # Simple fixed-function pipeline rendering
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadMatrixf(self._matrix_to_array(projection_matrix))
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadMatrixf(self._matrix_to_array(view_matrix))
            
            # Set up lighting
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_LIGHT0)
            
            light_pos = [10.0, 10.0, 10.0, 1.0]
            light_diffuse = [0.8, 0.8, 0.8, 1.0]
            light_ambient = [0.2, 0.2, 0.2, 1.0]
            
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_pos)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient)
            
            # Set material properties
            material_color = [0.7, 0.7, 0.9, 0.8 if self.display_mode == 'transparent' else 1.0]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, material_color)
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 32.0)
            
            # Bind and render mesh
            self.vertex_buffer.bind()
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertex_buffer)
            
            if self.normal_buffer:
                self.normal_buffer.bind()
                gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                gl.glNormalPointer(gl.GL_FLOAT, 0, self.normal_buffer)
                
            # Choose rendering mode
            if self.display_mode == 'wireframe':
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glDisable(gl.GL_LIGHTING)
                gl.glColor3f(0.2, 0.2, 0.2)
            else:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                
            # Draw elements
            if self.index_buffer:
                self.index_buffer.bind()
                gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)
            else:
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertex_count)
                
            # Clean up
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisable(gl.GL_BLEND)
            
        except Exception as e:
            print(f"Render error: {e}")
            
    def _matrix_to_array(self, matrix):
        """Convert QMatrix4x4 to OpenGL-compatible array"""
        if not matrix:
            return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        data = []
        for i in range(4):
            for j in range(4):
                data.append(matrix(i, j))
        return data
        
    def cleanup(self):
        """Clean up GPU resources"""
        if self.vertex_buffer:
            self.vertex_buffer.delete()
            self.vertex_buffer = None
        if self.normal_buffer:
            self.normal_buffer.delete()
            self.normal_buffer = None
        if self.index_buffer:
            self.index_buffer.delete()
            self.index_buffer = None


# Define base class based on available libraries
if PYSIDE6_AVAILABLE and HAVE_QOPENGLWIDGET and OPENGL_AVAILABLE:
    ViewportBase = QOpenGLWidget
elif PYSIDE6_AVAILABLE:
    ViewportBase = QWidget
else:
    ViewportBase = object

class Viewport3D(ViewportBase):
    """3D Viewport widget with interactive camera controls"""
    
    # Signals
    camera_changed = Signal() if PYSIDE6_AVAILABLE else lambda: None
    mesh_selected = Signal(object) if PYSIDE6_AVAILABLE else lambda: None
    
    def __init__(self, parent=None):
        if ViewportBase != object:
            super().__init__(parent)
        else:
            super().__init__()
        
        # If OpenGL widget or PyOpenGL is unavailable, set up a fallback UI
        if not (PYSIDE6_AVAILABLE and HAVE_QOPENGLWIDGET and OPENGL_AVAILABLE):
            if PYSIDE6_AVAILABLE:  # We have PySide6 but not the GL stack
                self._setup_fallback_ui()
            return
            
        # Set up OpenGL format
        format = QSurfaceFormat()
        format.setVersion(3, 3)
        format.setProfile(QSurfaceFormat.CoreProfile)
        format.setDepthBufferSize(24)
        format.setStencilBufferSize(8)
        format.setSamples(4)  # Anti-aliasing
        self.setFormat(format)
        
        # Initialize components
        self.camera = Camera3D()
        self.renderer = MeshRenderer()
        self.current_mesh = None
        
        # Mouse interaction state
        self.last_mouse_pos = QPoint()
        self.mouse_buttons = Qt.NoButton
        
        # Animation timer for smooth updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(16)  # ~60 FPS
        
        # Set up mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
    def _setup_fallback_ui(self):
        """Setup fallback UI when OpenGL is not available"""
        layout = QVBoxLayout(self)
        
        error_label = QLabel("3D Viewport Unavailable")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ff6666;")
        
        missing_deps = []
        if not PYSIDE6_AVAILABLE:
            missing_deps.append("• PySide6")
        if not OPENGL_AVAILABLE:
            missing_deps.append("• PyOpenGL")
            
        info_text = "Missing dependencies:\n" + "\n".join(missing_deps)
        info_text += "\n\nInstall with: pip install PySide6 PyOpenGL numpy"
        
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #cccccc; background-color: #404040; padding: 20px; border-radius: 10px;")
        
        layout.addWidget(error_label)
        layout.addWidget(info_label)
        
        self.setStyleSheet("background-color: #2a2a2a;")
        
    def initializeGL(self):
        """Initialize OpenGL context"""
        if not OPENGL_AVAILABLE:
            return
            
        # Set clear color and enable depth testing
        gl.glClearColor(0.25, 0.25, 0.25, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        
        # Print OpenGL info
        try:
            print(f"OpenGL Version: {gl.glGetString(gl.GL_VERSION).decode()}")
            print(f"OpenGL Vendor: {gl.glGetString(gl.GL_VENDOR).decode()}")
        except Exception as e:
            print(f"Could not get OpenGL info: {e}")
        
    def resizeGL(self, width: int, height: int):
        """Handle viewport resize"""
        if not OPENGL_AVAILABLE:
            return
            
        gl.glViewport(0, 0, width, height)
        
    def paintGL(self):
        """Render the 3D scene"""
        if not OPENGL_AVAILABLE:
            return
            
        # Clear the screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Set up camera matrices
        aspect_ratio = self.width() / max(1, self.height())
        view_matrix = self.camera.get_view_matrix()
        projection_matrix = self.camera.get_projection_matrix(aspect_ratio)
        
        # Render grid
        self._render_grid(view_matrix, projection_matrix)
        
        # Render the current mesh
        if self.current_mesh and self.renderer:
            self.renderer.render(view_matrix, projection_matrix)
            
    def _render_grid(self, view_matrix, projection_matrix):
        """Render a reference grid"""
        if not OPENGL_AVAILABLE:
            return
            
        try:
            # Simple grid rendering using immediate mode
            gl.glDisable(gl.GL_LIGHTING)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadMatrixf(self.renderer._matrix_to_array(projection_matrix))
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadMatrixf(self.renderer._matrix_to_array(view_matrix))
            
            # Draw grid lines
            gl.glColor3f(0.3, 0.3, 0.3)
            gl.glBegin(gl.GL_LINES)
            
            grid_size = 10
            grid_spacing = 5
            
            for i in range(-grid_size, grid_size + 1):
                pos = i * grid_spacing
                # X-axis lines
                gl.glVertex3f(pos, -grid_size * grid_spacing, 0)
                gl.glVertex3f(pos, grid_size * grid_spacing, 0)
                # Y-axis lines
                gl.glVertex3f(-grid_size * grid_spacing, pos, 0)
                gl.glVertex3f(grid_size * grid_spacing, pos, 0)
                
            gl.glEnd()
            
            # Draw coordinate axes
            gl.glLineWidth(2.0)
            gl.glBegin(gl.GL_LINES)
            
            # X-axis (red)
            gl.glColor3f(1.0, 0.0, 0.0)
            gl.glVertex3f(0, 0, 0)
            gl.glVertex3f(10, 0, 0)
            
            # Y-axis (green)
            gl.glColor3f(0.0, 1.0, 0.0)
            gl.glVertex3f(0, 0, 0)
            gl.glVertex3f(0, 10, 0)
            
            # Z-axis (blue)
            gl.glColor3f(0.0, 0.0, 1.0)
            gl.glVertex3f(0, 0, 0)
            gl.glVertex3f(0, 0, 10)
            
            gl.glEnd()
            gl.glLineWidth(1.0)
        except Exception as e:
            print(f"Grid render error: {e}")
        
    def set_mesh(self, mesh):
        """Set the mesh to display"""
        self.current_mesh = mesh
        if mesh and self.renderer and hasattr(mesh, 'vertices'):
            self.renderer.upload_mesh(mesh)
            
            # Auto-frame the mesh
            if hasattr(mesh, 'vertices') and mesh.vertices is not None and PYSIDE6_AVAILABLE:
                try:
                    min_point = QVector3D(*[min(v[i] for v in mesh.vertices) for i in range(3)])
                    max_point = QVector3D(*[max(v[i] for v in mesh.vertices) for i in range(3)])
                    self.camera.frame_bounds(min_point, max_point)
                except Exception as e:
                    print(f"Auto-frame error: {e}")
                
        if hasattr(self, 'update'):
            self.update()
        
    def set_display_mode(self, mode: str):
        """Set the display mode ('solid', 'wireframe', 'transparent')"""
        if self.renderer:
            self.renderer.display_mode = mode
        if hasattr(self, 'update'):
            self.update()
        
    def set_wireframe_mode(self, enabled: bool):
        """Set wireframe rendering mode"""
        mode = 'wireframe' if enabled else 'solid'
        self.set_display_mode(mode)
        
    def set_view_mode(self, mode: str):
        """Set view mode (solid, wireframe, transparent)"""
        self.set_display_mode(mode)
        
    def update_model(self, model_data):
        """Update the displayed 3D model"""
        # This method maintains compatibility with the existing interface
        if hasattr(model_data, 'vertices'):
            self.set_mesh(model_data)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if not PYSIDE6_AVAILABLE:
            return
        self.last_mouse_pos = event.pos()
        self.mouse_buttons = event.buttons()
        
    def mouseMoveEvent(self, event):
        """Handle mouse move events for camera control"""
        if not PYSIDE6_AVAILABLE or not self.mouse_buttons:
            return
            
        delta_x = event.pos().x() - self.last_mouse_pos.x()
        delta_y = event.pos().y() - self.last_mouse_pos.y()
        
        if self.mouse_buttons & Qt.LeftButton:
            # Orbit camera
            self.camera.orbit(delta_x, -delta_y)
            
        elif self.mouse_buttons & Qt.RightButton:
            # Pan camera
            self.camera.pan(-delta_x, delta_y)
            
        self.last_mouse_pos = event.pos()
        if hasattr(self.camera_changed, 'emit'):
            self.camera_changed.emit()
        
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if not PYSIDE6_AVAILABLE:
            return
        delta = event.angleDelta().y() / 120.0  # Standard wheel step
        self.camera.zoom(-delta)
        if hasattr(self.camera_changed, 'emit'):
            self.camera_changed.emit()
        
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if not PYSIDE6_AVAILABLE:
            return
            
        if event.key() == Qt.Key_F:
            # Frame selection
            if self.current_mesh and hasattr(self.current_mesh, 'vertices') and self.current_mesh.vertices:
                try:
                    min_point = QVector3D(*[min(v[i] for v in self.current_mesh.vertices) for i in range(3)])
                    max_point = QVector3D(*[max(v[i] for v in self.current_mesh.vertices) for i in range(3)])
                    self.camera.frame_bounds(min_point, max_point)
                    if hasattr(self.camera_changed, 'emit'):
                        self.camera_changed.emit()
                except Exception as e:
                    print(f"Frame error: {e}")
                
        elif event.key() == Qt.Key_1:
            self.set_display_mode('solid')
        elif event.key() == Qt.Key_2:
            self.set_display_mode('wireframe')
        elif event.key() == Qt.Key_3:
            self.set_display_mode('transparent')
            
        if hasattr(super(), 'keyPressEvent'):
            super().keyPressEvent(event)
