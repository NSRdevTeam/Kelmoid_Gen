# AdvancedCAD User Guide

Welcome to AdvancedCAD, a next-generation 3D modeling software that combines the power of script-based CAD with modern user interface design and enhanced features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Scripting Language](#scripting-language)
4. [3D Primitives](#3d-primitives)
5. [2D Shapes and Extrusions](#2d-shapes-and-extrusions)
6. [Transformations](#transformations)
7. [CSG Operations](#csg-operations)
8. [File Import/Export](#file-importexport)
9. [Interactive Parameters](#interactive-parameters)
10. [Advanced Features](#advanced-features)
11. [Tips and Best Practices](#tips-and-best-practices)
12. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. **Prerequisites**: Ensure you have Python 3.8+ installed on your system.

2. **Quick Setup**:
   ```bash
   python setup.py
   ```
   This will install all dependencies and create a launcher.

3. **Manual Installation**:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

### First Steps

1. **Launch AdvancedCAD**: Run `python main.py` or double-click `run_advancedcad.bat`
2. **Explore the Interface**: Familiarize yourself with the layout
3. **Try the Sample Script**: The editor starts with a sample script - press F5 to render it
4. **Experiment**: Modify parameters and see real-time changes

## Interface Overview

### Main Components

**Code Editor (Left Panel)**
- Syntax highlighting for AdvancedCAD language
- Line numbers and bracket matching
- Auto-indentation and code completion
- Error highlighting with tooltips

**3D Viewport (Right Panel)**
- Real-time 3D preview
- Multiple view modes: Solid, Wireframe, Transparent
- Mouse controls: Rotate (drag), Pan (Shift+drag), Zoom (wheel)

**Parameter Panel (Right Dock)**
- Interactive controls for script variables
- Real-time parameter adjustment with sliders
- Automatic detection of script parameters

**Project Explorer (Left Dock)**
- File browser and project management
- Recent files and templates
- Quick access to examples

**Console (Bottom Dock)**
- Error messages and warnings
- Render progress and statistics
- System information

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Render Model | F5 |
| Quick Render | F6 |
| New File | Ctrl+N |
| Open File | Ctrl+O |
| Save File | Ctrl+S |
| Save As | Ctrl+Shift+S |
| Toggle Console | Ctrl+` |

## Scripting Language

AdvancedCAD uses an enhanced version of the OpenSCAD language with improved syntax and additional features.

### Basic Syntax

```javascript
// Comments use double slashes
/* Multi-line comments
   are also supported */

// Variable assignments
width = 20;
height = 10;
depth = 5;

// Function calls with parameters
cube([width, height, depth]);

// Boolean operations
difference() {
    cube([10, 10, 10]);
    sphere(r=5);
}
```

### Variables and Parameters

```javascript
// Simple variables
box_size = 10;
hole_radius = 2;

// Arrays/vectors
dimensions = [20, 15, 8];
position = [0, 0, 5];

// Boolean values
center_object = true;
add_holes = false;
```

### Control Structures

```javascript
// Conditional statements
if (add_holes) {
    cylinder(h=10, r=hole_radius);
}

// Loops (for advanced users)
for (i = [0:3]) {
    translate([i*10, 0, 0])
        cube([5, 5, 5]);
}
```

## 3D Primitives

### Basic Shapes

#### Cube/Box
```javascript
// Basic cube
cube([10, 10, 10]);

// Centered cube
cube([10, 15, 5], center=true);

// Single dimension (creates cube)
cube(8);
```

#### Sphere
```javascript
// Basic sphere
sphere(r=5);

// High resolution sphere
sphere(radius=3, fn=32);

// Sphere using diameter
sphere(d=10);
```

#### Cylinder
```javascript
// Basic cylinder
cylinder(h=10, r=3);

// Tapered cylinder (cone)
cylinder(h=8, r1=5, r2=2);

// High resolution cylinder
cylinder(h=15, r=4, fn=24);
```

#### Cone
```javascript
// Cone (special case of cylinder)
cone(h=10, r=5);

// Equivalent to:
cylinder(h=10, r1=5, r2=0);
```

#### Torus
```javascript
// Basic torus
torus(R=5, r=1);

// High resolution torus
torus(major_radius=4, minor_radius=0.8, major_fn=32, minor_fn=16);
```

### Advanced Primitives

#### Polyhedron
```javascript
// Custom polyhedron from points and faces
points = [
    [0, 0, 0], [10, 0, 0], [5, 10, 0], [5, 5, 10]
];
faces = [
    [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
];
polyhedron(points=points, faces=faces);
```

## 2D Shapes and Extrusions

### 2D Primitives

#### Circle
```javascript
circle(r=5);
circle(d=10, fn=32);  // High resolution
```

#### Square/Rectangle
```javascript
square([10, 10]);          // Square
square([15, 8]);           // Rectangle
square([12, 12], center=true); // Centered
```

#### Polygon
```javascript
// Custom polygon
polygon([
    [0, 0], [10, 0], [15, 8], [5, 12], [-2, 8]
]);

// Regular polygon
polygon(sides=6, r=8);  // Hexagon
```

### Extrusion Operations

#### Linear Extrude
```javascript
// Basic extrusion
linear_extrude(height=10) {
    circle(r=5);
}

// With twist and scale
linear_extrude(height=20, twist=90, scale=0.5) {
    square([10, 10], center=true);
}

// Multiple slices for smooth twisting
linear_extrude(height=15, twist=180, slices=20) {
    polygon(sides=6, r=4);
}
```

#### Rotate Extrude
```javascript
// Full revolution
rotate_extrude() {
    translate([5, 0, 0])
        circle(r=2);
}

// Partial revolution
rotate_extrude(angle=270) {
    polygon([[0, 0], [8, 0], [8, 3], [6, 5], [0, 5]]);
}
```

## Transformations

### Translation
```javascript
// Move object
translate([10, 5, 0]) {
    cube([5, 5, 5]);
}

// Using variables
offset = [15, 0, 8];
translate(offset) {
    sphere(r=3);
}
```

### Rotation
```javascript
// Rotate around axes (degrees)
rotate([90, 0, 0]) {
    cylinder(h=10, r=2);
}

// Rotate around arbitrary point
translate([5, 5, 0]) {
    rotate([0, 0, 45]) {
        translate([-5, -5, 0]) {
            cube([10, 10, 2]);
        }
    }
}
```

### Scaling
```javascript
// Uniform scaling
scale(2) {
    cube([5, 5, 5]);
}

// Non-uniform scaling
scale([2, 1, 0.5]) {
    sphere(r=4);
}
```

### Mirroring
```javascript
// Mirror across YZ plane (x=0)
mirror([1, 0, 0]) {
    translate([2, 0, 0]) {
        cube([3, 3, 3]);
    }
}

// Mirror across custom plane
mirror([1, 1, 0]) {  // 45-degree diagonal
    cylinder(h=8, r=2);
}
```

## CSG Operations

### Union
```javascript
// Combine multiple objects
union() {
    cube([10, 10, 2]);
    translate([0, 0, 2]) {
        cylinder(h=5, r=4);
    }
    translate([0, 0, 7]) {
        sphere(r=3);
    }
}
```

### Difference
```javascript
// Subtract objects from the first
difference() {
    // Base object
    cube([20, 20, 8]);
    
    // Objects to subtract
    translate([5, 5, -1]) {
        cylinder(h=10, r=2);
    }
    translate([15, 15, -1]) {
        cylinder(h=10, r=2);
    }
}
```

### Intersection
```javascript
// Keep only overlapping parts
intersection() {
    cube([15, 15, 15], center=true);
    rotate([0, 45, 0]) {
        cube([15, 15, 15], center=true);
    }
}
```

### Hull
```javascript
// Create convex hull around objects
hull() {
    translate([0, 0, 0]) sphere(r=2);
    translate([10, 0, 0]) sphere(r=2);
    translate([5, 8, 5]) sphere(r=2);
}
```

### Minkowski Sum
```javascript
// Minkowski sum (rounded edges)
minkowski() {
    cube([10, 10, 2]);
    sphere(r=1);  // Rounds all edges with 1mm radius
}
```

## File Import/Export

### Supported Formats

**Export Formats:**
- STL (Binary/ASCII) - For 3D printing
- OBJ - For 3D graphics applications
- PLY - For mesh processing
- AdvancedCAD (.acad) - Native format with full feature support

**Import Formats:**
- STL (Binary/ASCII)
- OBJ
- AdvancedCAD (.acad)

### Export Options

```javascript
// Export with specific settings
File -> Export -> Export STL...

// Options available:
// - Binary vs ASCII format
// - Precision (decimal places)
// - Mesh optimization
// - Scale factor
// - Units (mm, cm, m, in, ft)
```

### Batch Export
Create multiple formats simultaneously from the export dialog.

## Interactive Parameters

### Automatic Parameter Detection

AdvancedCAD automatically detects variables in your script and creates interactive controls:

```javascript
// These variables will appear in the Parameter Panel
width = 20;        // Slider control
height = 10;       // Slider control  
add_holes = true;  // Checkbox control
hole_count = 4;    // Integer slider

// Main model using parameters
difference() {
    cube([width, height, 5]);
    
    if (add_holes) {
        for (i = [1:hole_count]) {
            translate([i * width/(hole_count+1), height/2, -1]) {
                cylinder(h=7, r=1);
            }
        }
    }
}
```

### Parameter Ranges

You can specify parameter ranges using comments:

```javascript
width = 20;    // range: [10, 50]
height = 10;   // range: [5, 25]
angle = 45;    // range: [0, 360]
```

### Parameter Groups

Organize parameters with comments:

```javascript
// === Dimensions ===
width = 20;
height = 10;
depth = 5;

// === Features ===
add_holes = true;
hole_diameter = 3;

// === Advanced ===
twist_angle = 0;
taper_factor = 1.0;
```

## Advanced Features

### Modules (Functions)
```javascript
// Define reusable modules
module mounting_hole(diameter, depth) {
    cylinder(h=depth, d=diameter);
}

// Use the module
difference() {
    cube([20, 20, 3]);
    
    translate([5, 5, -1]) {
        mounting_hole(diameter=3, depth=5);
    }
    translate([15, 15, -1]) {
        mounting_hole(diameter=3, depth=5);
    }
}
```

### Variables and Calculations
```javascript
// Mathematical operations
outer_diameter = 20;
wall_thickness = 2;
inner_diameter = outer_diameter - 2*wall_thickness;

// Use in geometry
difference() {
    cylinder(h=10, d=outer_diameter);
    translate([0, 0, -1]) {
        cylinder(h=12, d=inner_diameter);
    }
}
```

### Conditional Geometry
```javascript
include_lid = true;
include_handle = false;

// Main body
cylinder(h=20, r=8);

// Optional lid
if (include_lid) {
    translate([0, 0, 20]) {
        cylinder(h=2, r=9);
    }
}

// Optional handle
if (include_handle) {
    translate([0, 8, 10]) {
        rotate([90, 0, 0]) {
            torus(R=3, r=0.5);
        }
    }
}
```

## Tips and Best Practices

### Modeling Best Practices

1. **Use Meaningful Variable Names**
   ```javascript
   // Good
   base_width = 50;
   mounting_hole_diameter = 6;
   
   // Avoid
   w = 50;
   d = 6;
   ```

2. **Comment Your Code**
   ```javascript
   // Create base plate
   cube([base_width, base_depth, base_thickness]);
   
   // Add mounting posts
   for (i = [0:3]) {
       // Calculate post position
       x = (i % 2) * (base_width - post_spacing);
       y = floor(i / 2) * (base_depth - post_spacing);
       
       translate([x + post_spacing/2, y + post_spacing/2, base_thickness]) {
           cylinder(h=post_height, d=post_diameter);
       }
   }
   ```

3. **Use Parametric Design**
   ```javascript
   // Define all dimensions at the top
   box_width = 100;
   box_height = 60;
   box_depth = 40;
   wall_thickness = 3;
   
   // Calculate derived dimensions
   inner_width = box_width - 2*wall_thickness;
   inner_height = box_height - 2*wall_thickness;
   inner_depth = box_depth - wall_thickness;
   ```

4. **Break Complex Models into Modules**
   ```javascript
   module electronics_enclosure() {
       // Main enclosure logic here
   }
   
   module mounting_feet() {
       // Mounting feet logic here
   }
   
   // Combine modules
   electronics_enclosure();
   mounting_feet();
   ```

### Performance Tips

1. **Use Appropriate Resolution**
   ```javascript
   // For preview (fast)
   cylinder(h=10, r=5, fn=12);
   
   // For final render (smooth)
   cylinder(h=10, r=5, fn=32);
   ```

2. **Minimize Complex Operations**
   - Use hull() sparingly on complex objects
   - Prefer simple CSG operations when possible
   - Cache results of expensive calculations

3. **Use Preview Mode**
   - F6 for quick preview during design
   - F5 for full render when ready

### Code Organization

1. **File Structure**
   ```
   my_project/
   ├── main.acad           # Main model file
   ├── modules/
   │   ├── hardware.acad   # Hardware components
   │   └── brackets.acad   # Mounting brackets
   └── examples/
       └── variations.acad  # Design variations
   ```

2. **Version Control**
   - Use Git for version control
   - Commit frequently with meaningful messages
   - Tag release versions

## Troubleshooting

### Common Issues

**Rendering Errors**
- Check syntax errors in the Console panel
- Verify all variables are defined
- Ensure proper bracket matching

**Performance Issues**
- Reduce resolution for preview (fn parameter)
- Simplify complex CSG operations
- Use caching for repeated calculations

**File Import/Export Problems**
- Check file format compatibility
- Verify file permissions
- Ensure sufficient disk space

**Display Issues**
- Update graphics drivers
- Try different view modes
- Restart the application

### Getting Help

1. **Check Console Messages**: The console provides detailed error information
2. **Example Gallery**: Browse included examples for reference
3. **Documentation**: This user guide and API documentation
4. **Community**: Join the AdvancedCAD community forums

### Reporting Bugs

When reporting issues:
1. Include your script code
2. Describe expected vs actual behavior
3. Include console error messages
4. Specify your operating system and version

---

## Quick Reference

### Essential Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `cube([x,y,z])` | Create box | `cube([10,5,2])` |
| `sphere(r=n)` | Create sphere | `sphere(r=5)` |
| `cylinder(h=n,r=n)` | Create cylinder | `cylinder(h=10,r=3)` |
| `translate([x,y,z])` | Move object | `translate([5,0,0])` |
| `rotate([x,y,z])` | Rotate object | `rotate([90,0,0])` |
| `union()` | Combine objects | `union() { ... }` |
| `difference()` | Subtract objects | `difference() { ... }` |
| `intersection()` | Intersect objects | `intersection() { ... }` |

This completes the comprehensive User Guide for AdvancedCAD. The guide covers everything from basic usage to advanced techniques, providing users with the knowledge they need to effectively use the software.
