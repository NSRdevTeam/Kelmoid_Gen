# AdvancedCAD - Next-Generation 3D Modeling Software

AdvancedCAD is a modern, script-based 3D CAD modeling application inspired by OpenSCAD but with significant improvements in user experience, performance, and functionality.

## Key Improvements Over OpenSCAD

### 🚀 **Performance & Rendering**
- Real-time 3D rendering with modern OpenGL
- Multi-threaded CSG operations
- Smart caching system for faster iterations
- Progressive mesh generation for large models

### 💡 **Enhanced User Experience**
- Modern, customizable UI with docking panels
- Interactive parameter controls with real-time updates
- Advanced code editor with IDE features
- Multi-viewport support (wireframe, solid, transparent)

### 🛠️ **Advanced Features**
- Extended file format support (STL, OBJ, PLY, 3MF, AMF)
- Animation and path-following capabilities
- Better error handling and debugging tools
- Parametric animations for design verification

### 📝 **Improved Language**
- More intuitive scripting syntax
- Better error messages with line highlighting
- Auto-completion and syntax highlighting
- Integrated help system

## Features

- **Constructive Solid Geometry (CSG)**: Union, difference, intersection operations
- **2D/3D Primitives**: Comprehensive set of basic shapes
- **Extrusion Tools**: Linear and rotational extrusion with advanced options
- **Transformations**: Translate, rotate, scale, mirror operations
- **File I/O**: Import/export multiple 3D formats
- **Real-time Preview**: Instant visual feedback
- **Parameter Controls**: GUI sliders for script variables

## Requirements

- Python 3.8+
- PySide6/Qt6
- OpenGL 3.3+
- NumPy, SciPy
- Open3D for mesh processing

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AdvancedCAD

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Quick Start

1. Launch AdvancedCAD
2. Write your 3D model script in the left editor panel
3. Press F5 or click "Render" to see the 3D preview
4. Use the parameter panel to adjust values in real-time
5. Export your model in various formats

## Example

```python
# Create a simple tree model
union() {
    # Tree trunk
    cylinder(height=30, radius=8, center=True)
    
    # Tree crown
    translate([0, 0, 40])
        sphere(radius=20)
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.
