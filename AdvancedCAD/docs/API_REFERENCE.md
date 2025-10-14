# AdvancedCAD API Reference

This document provides detailed technical information about the AdvancedCAD modules and their APIs for developers and advanced users.

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Script Parser API](#script-parser-api)
3. [3D Primitives API](#3d-primitives-api)
4. [CSG Operations API](#csg-operations-api)
5. [Extrusions API](#extrusions-api)
6. [File I/O API](#file-io-api)
7. [Transformations API](#transformations-api)
8. [Configuration API](#configuration-api)
9. [UI Components API](#ui-components-api)
10. [Extension Development](#extension-development)

## Core Architecture

AdvancedCAD is built using a modular architecture with the following key components:

```
AdvancedCAD/
├── core/               # Core functionality modules
│   ├── __init__.py
│   ├── parser.py       # Script parsing and lexical analysis
│   ├── primitives.py   # 3D primitive shape generation
│   ├── csg_ops.py      # CSG boolean operations
│   ├── extrusions.py   # 2D shapes and extrusions
│   ├── transforms.py   # Transformations and animations
│   └── file_io.py      # File import/export functionality
├── ui/                 # User interface components
│   ├── __init__.py
│   ├── main_window.py  # Main application window
│   ├── code_editor.py  # Code editor with syntax highlighting
│   ├── parameter_panel.py  # Interactive parameter controls
│   ├── project_explorer.py  # File browser
│   └── console_widget.py  # Console output
├── config.py           # Configuration management
├── main.py            # Application entry point
└── examples/          # Example scripts and models
```

## Script Parser API

### Module: `core.parser`

The parser module handles lexical analysis and parsing of AdvancedCAD scripts.

#### Classes

##### `Token`

Represents a lexical token in the script.

```python
class Token:
    def __init__(self, type: str, value: any, line: int, column: int)
    
    # Properties
    type: str       # Token type (IDENTIFIER, NUMBER, STRING, etc.)
    value: any      # Token value
    line: int       # Line number in source
    column: int     # Column number in source
```

**Token Types:**
- `IDENTIFIER` - Variable names, function names
- `NUMBER` - Numeric literals (int/float)
- `STRING` - String literals
- `BOOLEAN` - Boolean literals (true/false)
- `OPERATOR` - Mathematical operators (+, -, *, /, etc.)
- `DELIMITER` - Punctuation (;, ,, [, ], etc.)
- `KEYWORD` - Language keywords (if, for, module, etc.)

##### `Lexer`

Performs lexical analysis of script text.

```python
class Lexer:
    def __init__(self, text: str)
    
    def tokenize(self) -> List[Token]:
        """Convert script text into tokens."""
        
    def get_next_token(self) -> Optional[Token]:
        """Get the next token from the input stream."""
        
    def peek_token(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at upcoming tokens without consuming them."""
```

**Example Usage:**
```python
from core.parser import Lexer

script = "cube([10, 10, 10]);"
lexer = Lexer(script)
tokens = lexer.tokenize()

for token in tokens:
    print(f"{token.type}: {token.value}")
```

##### `ASTNode`

Base class for Abstract Syntax Tree nodes.

```python
class ASTNode:
    def __init__(self, node_type: str)
    
    # Properties
    node_type: str
    children: List[ASTNode]
    attributes: Dict[str, any]
    
    def add_child(self, child: ASTNode) -> None
    def get_attribute(self, key: str, default=None) -> any
    def set_attribute(self, key: str, value: any) -> None
```

**Node Types:**
- `PROGRAM` - Root node
- `FUNCTION_CALL` - Function invocations
- `VARIABLE_ASSIGNMENT` - Variable declarations
- `BLOCK` - Code blocks ({ ... })
- `EXPRESSION` - Mathematical expressions
- `LITERAL` - Constant values

##### `Parser`

Parses tokens into an Abstract Syntax Tree.

```python
class Parser:
    def __init__(self, tokens: List[Token])
    
    def parse(self) -> ASTNode:
        """Parse tokens into AST."""
        
    def parse_statement(self) -> ASTNode:
        """Parse a single statement."""
        
    def parse_expression(self) -> ASTNode:
        """Parse mathematical expressions."""
        
    def parse_function_call(self) -> ASTNode:
        """Parse function calls with parameters."""
```

**Example Usage:**
```python
from core.parser import Lexer, Parser

script = """
width = 20;
height = 10;
cube([width, height, 5]);
"""

lexer = Lexer(script)
tokens = lexer.tokenize()
parser = Parser(tokens)
ast = parser.parse()
```

#### Error Handling

The parser includes comprehensive error handling:

```python
class ParseError(Exception):
    def __init__(self, message: str, token: Token = None)
    
    @property
    def line(self) -> int:
        """Line number where error occurred."""
        
    @property
    def column(self) -> int:
        """Column number where error occurred."""
```

## 3D Primitives API

### Module: `core.primitives`

Generates 3D primitive shapes with parametric controls.

#### Classes

##### `Mesh`

Represents a 3D mesh with vertices, faces, and normals.

```python
class Mesh:
    def __init__(self, vertices: List[List[float]], faces: List[List[int]], 
                 normals: List[List[float]] = None)
    
    # Properties
    vertices: List[List[float]]  # 3D vertex coordinates
    faces: List[List[int]]       # Face indices (triangles)
    normals: List[List[float]]   # Vertex normals
    
    def calculate_normals(self) -> None:
        """Calculate vertex normals from face geometry."""
        
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Get bounding box min/max coordinates."""
        
    def transform(self, matrix: List[List[float]]) -> None:
        """Apply transformation matrix to all vertices."""
```

##### `PrimitiveGenerator`

Factory class for creating 3D primitive shapes.

```python
class PrimitiveGenerator:
    @staticmethod
    def create_cube(size: Union[float, List[float]], 
                   center: bool = False) -> Mesh:
        """Create cube/box mesh."""
        
    @staticmethod
    def create_sphere(radius: float, resolution: int = 16) -> Mesh:
        """Create sphere mesh using UV sphere algorithm."""
        
    @staticmethod
    def create_cylinder(height: float, radius: float, 
                       radius_top: float = None, 
                       resolution: int = 16,
                       center: bool = False) -> Mesh:
        """Create cylinder/cone mesh."""
        
    @staticmethod
    def create_torus(major_radius: float, minor_radius: float,
                    major_segments: int = 16, 
                    minor_segments: int = 8) -> Mesh:
        """Create torus mesh."""
        
    @staticmethod
    def create_cone(height: float, radius: float, 
                   resolution: int = 16,
                   center: bool = False) -> Mesh:
        """Create cone mesh (convenience method)."""
```

**Example Usage:**
```python
from core.primitives import PrimitiveGenerator

# Create basic shapes
cube = PrimitiveGenerator.create_cube([10, 5, 2])
sphere = PrimitiveGenerator.create_sphere(radius=5, resolution=32)
cylinder = PrimitiveGenerator.create_cylinder(height=10, radius=3)

# Access mesh data
print(f"Cube has {len(cube.vertices)} vertices")
print(f"Cube has {len(cube.faces)} faces")
```

#### Advanced Features

##### Custom Primitive Creation

```python
def create_custom_primitive(points: List[List[float]], 
                          faces: List[List[int]]) -> Mesh:
    """Create mesh from custom point/face data."""
    mesh = Mesh(points, faces)
    mesh.calculate_normals()
    return mesh

# Example: Create tetrahedron
points = [
    [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]
]
faces = [
    [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
]
tetrahedron = create_custom_primitive(points, faces)
```

## CSG Operations API

### Module: `core.csg_ops`

Performs Constructive Solid Geometry boolean operations.

#### Classes

##### `CSGOperation`

Base class for CSG operations.

```python
class CSGOperation:
    def __init__(self, operation_type: str)
    
    def execute(self, meshes: List[Mesh]) -> Mesh:
        """Execute the CSG operation on input meshes."""
        
    def validate_inputs(self, meshes: List[Mesh]) -> bool:
        """Validate that input meshes are suitable for operation."""
```

##### `CSGEngine`

Main engine for performing CSG operations with caching.

```python
class CSGEngine:
    def __init__(self, cache_size: int = 100, use_threading: bool = True)
    
    def union(self, *meshes: Mesh) -> Mesh:
        """Combine meshes (A ∪ B)."""
        
    def difference(self, base_mesh: Mesh, *subtract_meshes: Mesh) -> Mesh:
        """Subtract meshes from base (A - B - C ...)."""
        
    def intersection(self, *meshes: Mesh) -> Mesh:
        """Find intersection of meshes (A ∩ B ∩ C ...)."""
        
    def hull(self, *meshes: Mesh) -> Mesh:
        """Create convex hull around meshes."""
        
    def minkowski_sum(self, mesh_a: Mesh, mesh_b: Mesh) -> Mesh:
        """Compute Minkowski sum of two meshes."""
        
    def clear_cache(self) -> None:
        """Clear operation cache."""
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics."""
```

**Example Usage:**
```python
from core.primitives import PrimitiveGenerator
from core.csg_ops import CSGEngine

# Create shapes
cube = PrimitiveGenerator.create_cube([10, 10, 10])
sphere = PrimitiveGenerator.create_sphere(radius=6)
cylinder = PrimitiveGenerator.create_cylinder(height=12, radius=2)

# Perform operations
csg = CSGEngine()
result = csg.difference(cube, sphere, cylinder)

# Check cache statistics
stats = csg.get_cache_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
```

#### Performance Options

```python
class CSGEngine:
    def set_precision(self, precision: float) -> None:
        """Set geometric precision for operations."""
        
    def set_threading(self, enabled: bool) -> None:
        """Enable/disable multi-threading."""
        
    def set_cache_size(self, size: int) -> None:
        """Set maximum cache size."""
```

## Extrusions API

### Module: `core.extrusions`

Handles 2D shapes and extrusion operations.

#### Classes

##### `Shape2D`

Represents a 2D shape with points and curves.

```python
class Shape2D:
    def __init__(self, points: List[List[float]], closed: bool = True)
    
    # Properties
    points: List[List[float]]  # 2D point coordinates
    closed: bool              # Whether shape is closed
    curves: List[Dict]        # Curve segments (for advanced shapes)
    
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Get 2D bounding box."""
        
    def is_clockwise(self) -> bool:
        """Check if points are in clockwise order."""
        
    def reverse_winding(self) -> None:
        """Reverse point order."""
        
    def simplify(self, tolerance: float) -> None:
        """Simplify shape by removing redundant points."""
```

##### `Shape2DGenerator`

Factory for creating 2D shapes.

```python
class Shape2DGenerator:
    @staticmethod
    def create_circle(radius: float, segments: int = 16) -> Shape2D:
        """Create circular shape."""
        
    @staticmethod
    def create_square(size: Union[float, List[float]], 
                     center: bool = False) -> Shape2D:
        """Create square/rectangle shape."""
        
    @staticmethod
    def create_polygon(points: List[List[float]]) -> Shape2D:
        """Create custom polygon shape."""
        
    @staticmethod
    def create_regular_polygon(sides: int, radius: float) -> Shape2D:
        """Create regular polygon (triangle, pentagon, etc.)."""
```

##### `ExtrusionEngine`

Performs extrusion operations on 2D shapes.

```python
class ExtrusionEngine:
    def linear_extrude(self, shape: Shape2D, height: float,
                      twist: float = 0, scale: float = 1,
                      slices: int = 1, center: bool = False) -> Mesh:
        """Linear extrusion with optional twist and scaling."""
        
    def rotate_extrude(self, shape: Shape2D, angle: float = 360,
                      segments: int = 16) -> Mesh:
        """Rotational extrusion around Z-axis."""
        
    def path_extrude(self, shape: Shape2D, path: List[List[float]],
                    follow_path: bool = True, twist_per_unit: float = 0) -> Mesh:
        """Extrude shape along 3D path."""
        
    def sweep_extrude(self, shape: Shape2D, profile: List[List[float]],
                     up_vector: List[float] = [0, 0, 1]) -> Mesh:
        """Sweep shape along profile curve."""
```

**Example Usage:**
```python
from core.extrusions import Shape2DGenerator, ExtrusionEngine

# Create 2D shapes
circle = Shape2DGenerator.create_circle(radius=5, segments=32)
square = Shape2DGenerator.create_square([10, 10], center=True)
hexagon = Shape2DGenerator.create_regular_polygon(sides=6, radius=8)

# Perform extrusions
extruder = ExtrusionEngine()

# Linear extrusion with twist
twisted_hex = extruder.linear_extrude(
    hexagon, height=20, twist=90, scale=0.5, slices=20
)

# Rotational extrusion
bowl = extruder.rotate_extrude(
    Shape2DGenerator.create_polygon([
        [0, 0], [10, 0], [10, 2], [8, 4], [0, 4]
    ]), 
    angle=360, segments=32
)
```

#### Advanced Extrusion Features

##### Path Following

```python
def create_helix_path(radius: float, height: float, 
                     turns: float, segments: int) -> List[List[float]]:
    """Create helical path for path extrusion."""
    import math
    path = []
    for i in range(segments + 1):
        t = i / segments
        angle = turns * 2 * math.pi * t
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height * t
        path.append([x, y, z])
    return path

# Use helix path
helix_path = create_helix_path(radius=10, height=50, turns=3, segments=60)
spring = extruder.path_extrude(circle, helix_path, follow_path=True)
```

## File I/O API

### Module: `core.file_io`

Handles importing and exporting 3D models in various formats.

#### Classes

##### `FileIOEngine`

Main class for file operations.

```python
class FileIOEngine:
    def __init__(self)
    
    # Export methods
    def export_stl(self, mesh: Mesh, filepath: str, 
                   binary: bool = True, precision: int = 6) -> bool:
        """Export mesh to STL format."""
        
    def export_obj(self, mesh: Mesh, filepath: str, 
                   include_normals: bool = True, 
                   include_textures: bool = False) -> bool:
        """Export mesh to OBJ format."""
        
    def export_ply(self, mesh: Mesh, filepath: str,
                   binary: bool = True, 
                   include_colors: bool = False) -> bool:
        """Export mesh to PLY format."""
        
    def export_advancedcad(self, data: Dict, filepath: str) -> bool:
        """Export to native AdvancedCAD format."""
        
    # Import methods
    def import_stl(self, filepath: str) -> Mesh:
        """Import mesh from STL file."""
        
    def import_obj(self, filepath: str) -> Mesh:
        """Import mesh from OBJ file."""
        
    def import_ply(self, filepath: str) -> Mesh:
        """Import mesh from PLY file."""
        
    def import_advancedcad(self, filepath: str) -> Dict:
        """Import from native AdvancedCAD format."""
        
    # Utility methods
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported import/export formats."""
        
    def optimize_mesh(self, mesh: Mesh, 
                     merge_vertices: bool = True,
                     remove_duplicates: bool = True,
                     tolerance: float = 1e-6) -> Mesh:
        """Optimize mesh for export."""
```

**Example Usage:**
```python
from core.file_io import FileIOEngine
from core.primitives import PrimitiveGenerator

# Create a mesh
mesh = PrimitiveGenerator.create_sphere(radius=10, resolution=32)

# Export to various formats
io_engine = FileIOEngine()

# STL for 3D printing
io_engine.export_stl(mesh, "sphere.stl", binary=True)

# OBJ for graphics applications
io_engine.export_obj(mesh, "sphere.obj", include_normals=True)

# Optimized mesh
optimized = io_engine.optimize_mesh(mesh, merge_vertices=True)
io_engine.export_stl(optimized, "sphere_optimized.stl")

# Import mesh
imported_mesh = io_engine.import_stl("sphere.stl")
print(f"Imported {len(imported_mesh.vertices)} vertices")
```

#### Export Options

```python
class ExportOptions:
    def __init__(self):
        self.scale_factor = 1.0
        self.units = "mm"  # mm, cm, m, in, ft
        self.precision = 6
        self.binary_format = True
        self.include_normals = True
        self.optimize_mesh = True
        self.merge_threshold = 1e-6

# Use with export
options = ExportOptions()
options.scale_factor = 25.4  # Convert inches to mm
options.units = "mm"
io_engine.export_stl(mesh, "part.stl", options=options)
```

## Transformations API

### Module: `core.transforms`

Handles 3D transformations and animations.

#### Classes

##### `TransformationEngine`

Applies transformations to meshes.

```python
class TransformationEngine:
    @staticmethod
    def translate(mesh: Mesh, offset: List[float]) -> Mesh:
        """Translate mesh by offset vector."""
        
    @staticmethod
    def rotate(mesh: Mesh, angles: List[float], 
              center: List[float] = None) -> Mesh:
        """Rotate mesh by Euler angles (degrees)."""
        
    @staticmethod
    def scale(mesh: Mesh, factors: Union[float, List[float]],
             center: List[float] = None) -> Mesh:
        """Scale mesh uniformly or non-uniformly."""
        
    @staticmethod
    def mirror(mesh: Mesh, plane_normal: List[float],
              plane_point: List[float] = None) -> Mesh:
        """Mirror mesh across plane."""
        
    @staticmethod
    def apply_matrix(mesh: Mesh, matrix: List[List[float]]) -> Mesh:
        """Apply arbitrary 4x4 transformation matrix."""
        
    # Matrix utilities
    @staticmethod
    def create_translation_matrix(offset: List[float]) -> List[List[float]]:
        """Create translation matrix."""
        
    @staticmethod
    def create_rotation_matrix(angles: List[float]) -> List[List[float]]:
        """Create rotation matrix from Euler angles."""
        
    @staticmethod
    def create_scale_matrix(factors: List[float]) -> List[List[float]]:
        """Create scaling matrix."""
        
    @staticmethod
    def multiply_matrices(a: List[List[float]], 
                         b: List[List[float]]) -> List[List[float]]:
        """Multiply two 4x4 matrices."""
```

##### `AnimationEngine`

Handles keyframe-based animations.

```python
class AnimationEngine:
    def __init__(self)
    
    def add_keyframe(self, time: float, transform: Dict[str, any]) -> None:
        """Add transformation keyframe at time."""
        
    def set_interpolation(self, method: str) -> None:
        """Set interpolation method (linear, cubic, etc.)."""
        
    def get_transform_at_time(self, time: float) -> List[List[float]]:
        """Get transformation matrix at specific time."""
        
    def animate_mesh(self, mesh: Mesh, 
                    start_time: float, end_time: float,
                    steps: int) -> List[Mesh]:
        """Generate animated mesh sequence."""
        
    def export_animation(self, meshes: List[Mesh], 
                        filepath: str, format: str = "obj_sequence") -> bool:
        """Export animation as mesh sequence."""
```

**Example Usage:**
```python
from core.transforms import TransformationEngine, AnimationEngine
from core.primitives import PrimitiveGenerator

# Create base mesh
cube = PrimitiveGenerator.create_cube([5, 5, 5], center=True)

# Apply transformations
transformer = TransformationEngine()

# Rotate 45 degrees around Z-axis
rotated = transformer.rotate(cube, [0, 0, 45])

# Scale and translate
scaled = transformer.scale(rotated, [2, 1, 1])
final = transformer.translate(scaled, [10, 0, 5])

# Create animation
animator = AnimationEngine()
animator.set_interpolation("cubic")

# Add keyframes
animator.add_keyframe(0.0, {"translate": [0, 0, 0], "rotate": [0, 0, 0]})
animator.add_keyframe(1.0, {"translate": [10, 0, 0], "rotate": [0, 0, 90]})
animator.add_keyframe(2.0, {"translate": [10, 10, 0], "rotate": [0, 0, 180]})

# Generate animation frames
frames = animator.animate_mesh(cube, start_time=0, end_time=2, steps=60)
```

## Configuration API

### Module: `config`

Manages application configuration and settings.

#### Classes

##### `ConfigManager`

Handles application configuration with persistent storage.

```python
class ConfigManager:
    def __init__(self, config_file: str = "config.json", 
                 use_qt_settings: bool = True)
    
    def get(self, key: str, default_value: any = None) -> any:
        """Get configuration value."""
        
    def set(self, key: str, value: any) -> None:
        """Set configuration value."""
        
    def get_dict(self, prefix: str = "") -> Dict[str, any]:
        """Get all settings with optional prefix filter."""
        
    def save(self) -> bool:
        """Save configuration to file."""
        
    def load(self) -> bool:
        """Load configuration from file."""
        
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        
    # Convenience methods for common settings
    def get_recent_files(self) -> List[str]:
        """Get list of recent files."""
        
    def add_recent_file(self, filepath: str) -> None:
        """Add file to recent files list."""
        
    def get_ui_settings(self) -> Dict[str, any]:
        """Get UI-related settings."""
        
    def get_render_settings(self) -> Dict[str, any]:
        """Get rendering settings."""
```

**Example Usage:**
```python
from config import ConfigManager

config = ConfigManager()

# Set values
config.set("ui/window_width", 1200)
config.set("ui/window_height", 800)
config.set("render/resolution", 32)
config.set("render/enable_caching", True)

# Get values with defaults
width = config.get("ui/window_width", 1024)
cache_enabled = config.get("render/enable_caching", False)

# Recent files
config.add_recent_file("/path/to/model.acad")
recent = config.get_recent_files()

# Save configuration
config.save()
```

#### Default Settings

```python
DEFAULT_SETTINGS = {
    "ui/window_width": 1200,
    "ui/window_height": 800,
    "ui/code_editor_font": "Consolas",
    "ui/code_editor_font_size": 12,
    "ui/theme": "default",
    
    "render/default_resolution": 16,
    "render/max_resolution": 64,
    "render/enable_caching": True,
    "render/cache_size": 100,
    
    "export/default_format": "stl",
    "export/stl_binary": True,
    "export/precision": 6,
    
    "performance/use_threading": True,
    "performance/max_threads": 4,
}
```

## UI Components API

### Module: `ui.main_window`

Main application window and UI management.

#### Classes

##### `MainWindow`

Main application window class (inherits from QMainWindow).

```python
class MainWindow(QMainWindow):
    def __init__(self)
    
    # Signals
    script_changed = pyqtSignal(str)
    parameter_changed = pyqtSignal(str, object)
    render_requested = pyqtSignal()
    
    # Methods
    def set_script_content(self, content: str) -> None:
        """Set code editor content."""
        
    def get_script_content(self) -> str:
        """Get current script content."""
        
    def add_parameter_control(self, name: str, control_type: str,
                            value: any, **kwargs) -> None:
        """Add interactive parameter control."""
        
    def update_3d_view(self, mesh: Mesh) -> None:
        """Update 3D viewport with new mesh."""
        
    def show_console_message(self, message: str, 
                           level: str = "info") -> None:
        """Display message in console."""
        
    def set_status_message(self, message: str) -> None:
        """Set status bar message."""
```

### Module: `ui.code_editor`

Advanced code editor with syntax highlighting.

```python
class CodeEditor(QTextEdit):
    def __init__(self, parent=None)
    
    # Signals
    content_changed = pyqtSignal(str)
    syntax_error = pyqtSignal(int, str)  # line, message
    
    # Methods
    def set_syntax_highlighter(self, language: str) -> None:
        """Set syntax highlighting for language."""
        
    def set_theme(self, theme: str) -> None:
        """Set editor color theme."""
        
    def goto_line(self, line: int) -> None:
        """Jump to specific line."""
        
    def find_replace(self, find_text: str, replace_text: str = "",
                    replace_all: bool = False) -> int:
        """Find/replace text."""
        
    def get_current_word(self) -> str:
        """Get word under cursor."""
        
    def auto_complete(self, suggestions: List[str]) -> None:
        """Show auto-completion popup."""
```

## Extension Development

AdvancedCAD supports plugins and extensions through a well-defined API.

### Plugin Architecture

```python
class AdvancedCADPlugin:
    def __init__(self):
        self.name = "Plugin Name"
        self.version = "1.0.0"
        self.description = "Plugin description"
        
    def initialize(self, app_context) -> bool:
        """Initialize plugin with application context."""
        
    def get_menu_actions(self) -> List[QAction]:
        """Return menu actions to add to application."""
        
    def get_toolbar_actions(self) -> List[QAction]:
        """Return toolbar actions."""
        
    def process_mesh(self, mesh: Mesh) -> Mesh:
        """Process mesh (for filter plugins)."""
        
    def get_primitives(self) -> Dict[str, callable]:
        """Return custom primitive generators."""
        
    def cleanup(self) -> None:
        """Cleanup when plugin is unloaded."""
```

### Example Plugin

```python
class ExamplePlugin(AdvancedCADPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Custom Shapes"
        self.version = "1.0.0"
        
    def initialize(self, app_context):
        self.app = app_context
        return True
        
    def get_primitives(self):
        return {
            "star": self.create_star,
            "gear": self.create_gear,
        }
        
    def create_star(self, outer_radius=10, inner_radius=5, 
                   points=5, height=2):
        """Create star-shaped primitive."""
        # Implementation here
        pass
        
    def create_gear(self, teeth=20, module=2, height=5):
        """Create gear primitive."""
        # Implementation here
        pass
```

### Loading Plugins

```python
# In main application
class PluginManager:
    def __init__(self):
        self.plugins = []
        
    def load_plugin(self, plugin_path: str) -> bool:
        """Load plugin from file."""
        
    def load_plugins_from_directory(self, directory: str) -> int:
        """Load all plugins from directory."""
        
    def get_plugin(self, name: str) -> AdvancedCADPlugin:
        """Get plugin by name."""
        
    def unload_plugin(self, name: str) -> bool:
        """Unload plugin."""
```

This completes the comprehensive API Reference for AdvancedCAD. The reference provides detailed technical information for developers working with or extending the application.
