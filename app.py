import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import trimesh
from shapely.geometry import box, Point
from shapely.affinity import scale
import re
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# TEXT TO CAD PARSER AND GENERATOR
# =====================================================

class TextToCADGenerator:
    """Enhanced Text-to-CAD generator with all shapes"""
    def __init__(self):
        self.shapes_library = {
            'cube': self._create_cube,
            'box': self._create_cube,
            'sphere': self._create_sphere,
            'ball': self._create_sphere,
            'cylinder': self._create_cylinder,
            'tube': self._create_cylinder,
            'cone': self._create_cone,
            'pyramid': self._create_pyramid,
            'torus': self._create_torus,
            'ring': self._create_torus,
            'gear': self._create_gear,
            'bracket': self._create_bracket,
            'plate': self._create_plate,
            'rod': self._create_rod,
            'washer': self._create_washer,
            'screw': self._create_screw,
            'bolt': self._create_screw,
            'nut': self._create_nut,
            'bearing': self._create_bearing,
            'flange': self._create_flange,
            'pipe': self._create_pipe
        }

    def parse_prompt(self, prompt: str):
        """Parse text prompt to extract shape and parameters"""
        prompt = prompt.lower().strip()
        dimensions = self._extract_dimensions(prompt)
        
        # Identify shape
        shape_type = None
        for shape in self.shapes_library.keys():
            if shape in prompt:
                shape_type = shape
                break
        
        if not shape_type:
            shape_type = 'cube'
        
        color = self._extract_color(prompt)
        return {
            'shape': shape_type,
            'dimensions': dimensions,
            'color': color,
            'prompt': prompt
        }

    def _extract_dimensions(self, prompt: str):
        """Extract dimensions from prompt"""
        dimensions = {'length': 10, 'width': 10, 'height': 10, 'radius': 5, 'diameter': 10}
        
        patterns = {
            'length': r'length\s*[:=]?\s*(\d+\.?\d*)',
            'width': r'width\s*[:=]?\s*(\d+\.?\d*)',
            'height': r'height\s*[:=]?\s*(\d+\.?\d*)',
            'radius': r'radius\s*[:=]?\s*(\d+\.?\d*)',
            'diameter': r'diameter\s*[:=]?\s*(\d+\.?\d*)',
            'size': r'size\s*[:=]?\s*(\d+\.?\d*)',
            'thick': r'thick\w*\s*[:=]?\s*(\d+\.?\d*)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if key == 'size':
                    dimensions['length'] = dimensions['width'] = dimensions['height'] = value
                elif key == 'thick':
                    dimensions['height'] = value
                else:
                    dimensions[key] = value
        
        # Handle patterns like "10x20x30"
        dimension_match = re.search(r'(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)', prompt)
        if dimension_match:
            dimensions['length'] = float(dimension_match.group(1))
            dimensions['width'] = float(dimension_match.group(2))
            dimensions['height'] = float(dimension_match.group(3))
        
        return dimensions

    def _extract_color(self, prompt: str):
        """Extract color from prompt"""
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'grey']
        for color in colors:
            if color in prompt:
                return color
        return 'lightblue'

    # Shape creation methods
    def _create_cube(self, dims):
        return trimesh.creation.box(extents=[dims['length'], dims['width'], dims['height']])

    def _create_sphere(self, dims):
        radius = dims.get('radius', dims.get('diameter', 10) / 2)
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)

    def _create_cylinder(self, dims):
        radius = dims.get('radius', dims.get('diameter', 10) / 2)
        height = dims.get('height', 10)
        return trimesh.creation.cylinder(radius=radius, height=height)

    def _create_cone(self, dims):
        radius = dims.get('radius', dims.get('diameter', 10) / 2)
        height = dims.get('height', 10)
        return trimesh.creation.cone(radius=radius, height=height)

    def _create_pyramid(self, dims):
        height = dims.get('height', 10)
        base_size = dims.get('width', 10)
        vertices = np.array([
            [0, 0, height],
            [-base_size/2, -base_size/2, 0], [base_size/2, -base_size/2, 0],
            [base_size/2, base_size/2, 0], [-base_size/2, base_size/2, 0]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            [1, 4, 3], [1, 3, 2]
        ])
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def _create_torus(self, dims):
        major_radius = dims.get('radius', 10)
        minor_radius = major_radius * 0.3
        return trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius)

    def _create_gear(self, dims):
        radius = dims.get('radius', 10)
        height = dims.get('height', 5)
        return trimesh.creation.cylinder(radius=radius, height=height)

    def _create_bracket(self, dims):
        length = dims.get('length', 20)
        width = dims.get('width', 10)
        height = dims.get('height', 15)
        thickness = min(3, width * 0.3, height * 0.3)
        
        try:
            base = trimesh.creation.box(extents=[length, width, thickness])
            base = base.apply_translation([0, 0, -height/2 + thickness/2])
            
            vertical = trimesh.creation.box(extents=[thickness, width, height])
            vertical = vertical.apply_translation([length/2 - thickness/2, 0, 0])
            
            return base.union(vertical)
        except:
            return trimesh.creation.box(extents=[length, width, height])

    def _create_plate(self, dims):
        return trimesh.creation.box(extents=[dims.get('length', 20), dims.get('width', 15), dims.get('height', 2)])

    def _create_rod(self, dims):
        radius = dims.get('radius', 2)
        length = dims.get('length', 20)
        rod = trimesh.creation.cylinder(radius=radius, height=length)
        return rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))

    def _create_washer(self, dims):
        outer_radius = dims.get('radius', 10)
        inner_radius = outer_radius * 0.4
        height = dims.get('height', 2)
        outer = trimesh.creation.cylinder(radius=outer_radius, height=height)
        inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
        return outer.difference(inner)

    def _create_screw(self, dims):
        radius = dims.get('radius', 3)
        length = dims.get('length', 20)
        head_radius = radius * 1.5
        head_height = radius
        
        body = trimesh.creation.cylinder(radius=radius, height=length)
        head = trimesh.creation.cylinder(radius=head_radius, height=head_height)
        head = head.apply_translation([0, 0, length/2 + head_height/2])
        return body.union(head)

    def _create_nut(self, dims):
        radius = dims.get('radius', 5)
        height = dims.get('height', 4)
        inner_radius = radius * 0.4
        
        outer = trimesh.creation.cylinder(radius=radius, height=height, sections=6)
        inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
        return outer.difference(inner)

    def _create_bearing(self, dims):
        outer_radius = dims.get('radius', 10)
        inner_radius = outer_radius * 0.6
        height = dims.get('height', 5)
        outer = trimesh.creation.cylinder(radius=outer_radius, height=height)
        inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
        return outer.difference(inner)

    def _create_flange(self, dims):
        outer_radius = dims.get('radius', 15)
        inner_radius = outer_radius * 0.4
        height = dims.get('height', 5)
        outer = trimesh.creation.cylinder(radius=outer_radius, height=height)
        inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
        return outer.difference(inner)

    def _create_pipe(self, dims):
        outer_radius = dims.get('radius', 10)
        inner_radius = outer_radius * 0.8
        length = dims.get('length', 30)
        outer = trimesh.creation.cylinder(radius=outer_radius, height=length)
        inner = trimesh.creation.cylinder(radius=inner_radius, height=length * 1.1)
        return outer.difference(inner)

    def generate_3d_model(self, params):
        """Generate 3D model based on parameters"""
        shape_func = self.shapes_library.get(params['shape'], self._create_cube)
        return shape_func(params['dimensions'])

    def generate_3d_visualization(self, mesh, color='lightblue'):
        """Generate interactive 3D visualization using Plotly"""
        vertices = mesh.vertices
        faces = mesh.faces
        
        color_map = {
            'red': '#FF0000', 'blue': '#0000FF', 'green': '#00FF00',
            'yellow': '#FFFF00', 'orange': '#FFA500', 'purple': '#800080',
            'pink': '#FFC0CB', 'brown': '#A52A2A', 'black': '#000000',
            'white': '#FFFFFF', 'gray': '#808080', 'grey': '#808080',
            'lightblue': '#ADD8E6'
        }
        mesh_color = color_map.get(color, '#ADD8E6')
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=mesh_color,
                opacity=0.8,
                lighting=dict(ambient=0.18, diffuse=1, fresnel=0.1, specular=1, roughness=0.05),
                lightposition=dict(x=100, y=200, z=0)
            )
        ])
        
        fig.update_layout(
            title="3D CAD Model",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        return fig

# =====================================================
# 2D PLATE DESIGN AND G-CODE GENERATOR
# =====================================================

def parse_plate_description(description):
    """Parse textual description to extract geometric parameters for a plate"""
    width, height = 100, 100
    holes = []
    slots = []
    ovals = []
    
    # Find width and height
    match = re.search(r'(\d+)\s*mm\s*[xX×]\s*(\d+)\s*mm', description)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
    
    # Find circular holes
    for center_match in re.finditer(r'(hole|circle|circular cutout)[^\d]*(\d+)\s*mm', description):
        diameter = int(center_match.group(2))
        holes.append({
            'x': width / 2,
            'y': height / 2,
            'diameter': diameter
        })
    
    # Find slots
    for slot_match in re.finditer(r'(\d+)\s*mm\s+long\s+and\s+(\d+)\s*mm\s+wide\s+slot', description):
        slots.append({
            'x': width / 2,
            'y': height / 2,
            'length': int(slot_match.group(1)),
            'width': int(slot_match.group(2))
        })
    
    # Find ovals
    for oval_match in re.finditer(r'oval\s+hole\s+(\d+)\s*mm\s+long\s+and\s+(\d+)\s*mm\s+wide', description):
        ovals.append({
            'x': width / 2,
            'y': height / 2,
            'length': int(oval_match.group(1)),
            'width': int(oval_match.group(2))
        })
    
    return {
        "width": width,
        "height": height,
        "holes": holes,
        "slots": slots,
        "ovals": ovals
    }

def generate_3_view_drawings(description):
    """Generate a 3-view engineering drawing from the description"""
    parsed = parse_plate_description(description)
    width = parsed["width"]
    height = parsed["height"]
    depth = 5
    holes = parsed["holes"]
    slots = parsed["slots"]
    ovals = parsed["ovals"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = ['Top View', 'Front View', 'Side View']
    
    for ax, view in zip(axes, views):
        ax.set_title(view)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        if view == "Top View":
            # Draw the main plate
            shape = box(0, 0, width, height)
            x, y = shape.exterior.xy
            ax.plot(x, y, color='black')
            
            # Draw features
            for hole in holes:
                r = hole['diameter'] / 2
                circle = Point(hole['x'], hole['y']).buffer(r)
                hx, hy = circle.exterior.xy
                ax.plot(hx, hy, color='black')
            
            for slot in slots:
                slot_shape = Point(slot['x'], slot['y']).buffer(1)
                slot_shape = scale(slot_shape, slot['length']/2, slot['width']/2)
                sx, sy = slot_shape.exterior.xy
                ax.plot(sx, sy, color='black')
            
            for oval in ovals:
                ellipse = Point(oval['x'], oval['y']).buffer(1)
                ellipse = scale(ellipse, oval['length'] / 2, oval['width'] / 2)
                ox, oy = ellipse.exterior.xy
                ax.plot(ox, oy, color='black')
            
            ax.set_xlim(-10, width + 10)
            ax.set_ylim(-10, height + 10)
            
        elif view == "Front View":
            shape = box(0, 0, width, depth)
            x, y = shape.exterior.xy
            ax.plot(x, y, color='black')
            ax.set_xlim(-10, width + 10)
            ax.set_ylim(-10, depth + 10)
            
        elif view == "Side View":
            shape = box(0, 0, height, depth)
            x, y = shape.exterior.xy
            ax.plot(x, y, color='black')
            ax.set_xlim(-10, height + 10)
            ax.set_ylim(-10, depth + 10)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def generate_gcode(description):
    """Generate G-code for milling the part based on the description"""
    parsed = parse_plate_description(description)
    gcode = [
        "G21 ; Set units to mm",
        "G90 ; Use absolute positioning",
        "G17 ; Select XY plane",
        "M3 S1000 ; Start spindle",
        "G0 Z5 ; Lift Z to a safe height"
    ]
    
    # Mill the outer rectangle
    w, h = parsed['width'], parsed['height']
    gcode.extend([
        "\n; --- Mill Outer Profile ---",
        "G0 X0 Y0 ; Move to starting corner",
        "G1 Z-1 F100 ; Plunge down",
        f"G1 X{w} F300 ; Mill along X",
        f"G1 Y{h} ; Mill along Y",
        f"G1 X0 ; Mill back along X",
        "G1 Y0 ; Mill back to start",
        "G0 Z5 ; Retract Z"
    ])
    
    # Mill circular holes
    for hole in parsed['holes']:
        x, y, d = hole['x'], hole['y'], hole['diameter']
        r = d / 2
        gcode.extend([
            f"\n; --- Mill Hole at X{x}, Y{y}, D{d} ---",
            f"G0 X{x - r} Y{y} ; Move to start of circle",
            "G1 Z-1 F100 ; Plunge down",
            f"G2 I{r} J0 F200 ; Mill full circle (CW)",
            "G0 Z5 ; Retract Z"
        ])
    
    # Mill slots
    for slot in parsed['slots']:
        x, y, l, w_slot = slot['x'], slot['y'], slot['length'], slot['width']
        r = w_slot / 2
        x_start = x - (l - w_slot) / 2
        x_end = x + (l - w_slot) / 2
        gcode.extend([
            f"\n; --- Mill Slot at center X{x}, Y{y} ---",
            f"G0 X{x_start} Y{y - r}",
            "G1 Z-1 F100",
            f"G1 X{x_end} F200",
            f"G2 I0 J{r}",
            f"G1 X{x_start}",
            f"G2 I0 J{r}",
            "G0 Z5"
        ])
    
    # Ovals
    for oval in parsed['ovals']:
        x, y, l, w_oval = oval['x'], oval['y'], oval['length'], oval['width']
        gcode.append(f"\n; --- Oval hole at X{x}, Y{y} (manual operation needed) ---")
        gcode.append(f"; Oval of length {l} and width {w_oval} cannot be interpolated with simple G2/G3")
    
    gcode.append("\nM5 ; Stop spindle")
    gcode.append("G0 X0 Y0 ; Return to home")
    gcode.append("M30 ; End of program")
    
    return "\n".join(gcode)

# =====================================================
# CFD SOLVER
# =====================================================

def run_cfd_simulation(Lx=2.0, Ly=1.0, Nx=41, Ny=21, inlet_velocity=0.005, 
                      density=1.0, viscosity=0.05, obstacle_params=None, 
                      max_iterations=1000):
    """Run CFD simulation with given parameters"""
    try:
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
        
        # Initialize fields
        u = np.zeros((Ny, Nx))
        v = np.zeros((Ny, Nx))
        p = np.ones((Ny, Nx))
        
        # Create obstacle mask
        obstacle_mask = np.zeros((Ny, Nx), dtype=bool)
        if obstacle_params:
            ox1, oy1, ox2, oy2 = obstacle_params
            if ox1 != 0 or oy1 != 0 or ox2 != 0 or oy2 != 0:
                i_ox1 = max(0, int(round(ox1 / dx)))
                j_oy1 = max(0, int(round(oy1 / dy)))
                i_ox2 = min(Nx - 1, int(round(ox2 / dx)))
                j_oy2 = min(Ny - 1, int(round(oy2 / dy)))
                obstacle_mask[j_oy1:j_oy2, i_ox1:i_ox2] = True
        
        dt = 0.01
        nu = viscosity / density
        
        # Simple simulation loop
        for iteration in range(max_iterations):
            un = u.copy()
            vn = v.copy()
            
            # Apply boundary conditions
            u[:, 0] = inlet_velocity  # Inlet
            v[:, 0] = 0.0
            u[:, -1] = u[:, -2]  # Outlet
            v[:, -1] = v[:, -2]
            u[0, :] = 0.0  # Walls
            u[-1, :] = 0.0
            v[0, :] = 0.0
            v[-1, :] = 0.0
            u[obstacle_mask] = 0.0
            v[obstacle_mask] = 0.0
            
            # Simple explicit update (simplified)
            for j in range(1, Ny-1):
                for i in range(1, Nx-1):
                    if obstacle_mask[j, i]:
                        continue
                    
                    # Diffusion terms
                    diff_u = nu * ((un[j, i+1] - 2*un[j, i] + un[j, i-1])/dx**2 + 
                                  (un[j+1, i] - 2*un[j, i] + un[j-1, i])/dy**2)
                    diff_v = nu * ((vn[j, i+1] - 2*vn[j, i] + vn[j, i-1])/dx**2 + 
                                  (vn[j+1, i] - 2*vn[j, i] + vn[j-1, i])/dy**2)
                    
                    u[j, i] = un[j, i] + dt * diff_u
                    v[j, i] = vn[j, i] + dt * diff_v
            
            # Check convergence
            if iteration % 100 == 0:
                diff = np.max(np.abs(u - un))
                if diff < 1e-6:
                    break
        
        # Generate visualization
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)
        
        # Mask obstacle points
        u_plot = u.copy()
        v_plot = v.copy()
        u_plot[obstacle_mask] = np.nan
        v_plot[obstacle_mask] = np.nan
        
        velocity_magnitude = np.sqrt(u_plot**2 + v_plot**2)
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Velocity magnitude
        im1 = axes[0].contourf(X, Y, velocity_magnitude, levels=20, cmap='viridis')
        axes[0].set_title('Velocity Magnitude')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        plt.colorbar(im1, ax=axes[0])
        
        # Streamlines
        axes[1].streamplot(x, y, u_plot, v_plot, density=2, color='blue', linewidth=0.8)
        axes[1].set_title('Flow Streamlines')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        
        if np.any(obstacle_mask):
            for ax in axes:
                ax.contour(X, Y, obstacle_mask, levels=[0.5], colors='red', linewidths=2)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img, f"CFD simulation completed after {iteration+1} iterations"
        
    except Exception as e:
        # Return error image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"CFD Error: {str(e)}", ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.title("CFD Simulation Error")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img, f"Error: {str(e)}"

# =====================================================
# ORTHOGRAPHIC VIEWS GENERATOR
# =====================================================

def generate_orthographic_views(mesh):
    """Generate orthographic views from 3D mesh"""
    try:
        bounds = mesh.bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        x_dim = x_max - x_min
        y_dim = y_max - y_min
        z_dim = z_max - z_min
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        view_titles = ["Front View (XY)", "Top View (XZ)", "Side View (YZ)"]
        dims = [(x_dim, y_dim), (x_dim, z_dim), (y_dim, z_dim)]
        
        for i, (ax, title, (dim1, dim2)) in enumerate(zip(axes, view_titles, dims)):
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Draw a simple rectangular outline (simplified orthographic projection)
            if i == 0:  # Front view
                vertices_2d = mesh.vertices[:, [0, 1]]  # XY projection
            elif i == 1:  # Top view
                vertices_2d = mesh.vertices[:, [0, 2]]  # XZ projection
            else:  # Side view
                vertices_2d = mesh.vertices[:, [1, 2]]  # YZ projection
            
            # Simple boundary outline
            from scipy.spatial import ConvexHull
            if len(vertices_2d) > 3:
                hull = ConvexHull(vertices_2d)
                hull_points = vertices_2d[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                ax.plot(hull_points[:, 0], hull_points[:, 1], 'b-', linewidth=2)
            
            # Add dimensions
            ax.text(0.5, 0.95, f"Dim1: {dim1:.1f} mm", transform=ax.transAxes, ha='center')
            ax.text(0.05, 0.5, f"Dim2: {dim2:.1f} mm", transform=ax.transAxes, va='center', rotation=90)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
        
    except Exception as e:
        # Return error image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Orthographic Views Error: {str(e)}", ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.title("Orthographic Views Error")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img

# =====================================================
# MAIN APPLICATION FUNCTIONS
# =====================================================

# Initialize CAD generator
cad_generator = TextToCADGenerator()

def process_text_to_cad(prompt):
    """Process text prompt and generate CAD outputs"""
    try:
        params = cad_generator.parse_prompt(prompt)
        mesh_3d = cad_generator.generate_3d_model(params)
        
        # Generate 3D visualization
        fig_3d = cad_generator.generate_3d_visualization(mesh_3d, params['color'])
        
        # Generate orthographic views
        ortho_views = generate_orthographic_views(mesh_3d)
        
        summary = f"""
**Generated CAD Model Summary:**
- **Shape:** {params['shape'].title()}
- **Dimensions:** Length: {params['dimensions']['length']}mm, Width: {params['dimensions']['width']}mm, Height: {params['dimensions']['height']}mm
- **Color:** {params['color'].title()}
- **Original Prompt:** "{params['prompt']}"

The kelmoid has been successfully generated with both 3D visualization and orthographic views.
"""
        
        return fig_3d, ortho_views, summary
        
    except Exception as e:
        error_msg = f"Error generating CAD model: {str(e)}"
        placeholder_fig = go.Figure()
        placeholder_fig.add_annotation(text=error_msg, x=0.5, y=0.5, showarrow=False)
        
        # Create error image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        error_img = Image.open(buf)
        plt.close()
        
        return placeholder_fig, error_img, error_msg

def process_plate_design(description):
    """Process plate description and generate outputs"""
    try:
        if not description.strip():
            return None, "Please enter a description.", None
        
        drawing = generate_3_view_drawings(description)
        gcode = generate_gcode(description)
        
        return drawing, gcode, "Plate design generated successfully!"
        
    except Exception as e:
        error_msg = f"Error generating plate design: {str(e)}"
        return None, error_msg, None

def process_cfd_simulation(length, height, grid_x, grid_y, inlet_vel, density, viscosity, 
                          obs_x1, obs_y1, obs_x2, obs_y2, max_iter):
    """Process CFD simulation with given parameters"""
    try:
        obstacle_params = None
        if obs_x1 != 0 or obs_y1 != 0 or obs_x2 != 0 or obs_y2 != 0:
            obstacle_params = (obs_x1, obs_y1, obs_x2, obs_y2)
        
        result_img, message = run_cfd_simulation(
            Lx=length, Ly=height, Nx=grid_x, Ny=grid_y,
            inlet_velocity=inlet_vel, density=density, viscosity=viscosity,
            obstacle_params=obstacle_params, max_iterations=max_iter
        )
        
        return result_img, message
        
    except Exception as e:
        error_msg = f"CFD simulation error: {str(e)}"
        return None, error_msg

# =====================================================
# GRADIO INTERFACE
# =====================================================

def create_gradio_interface():
    """Create comprehensive Gradio interface"""
    
    with gr.Blocks(title="Kelmoid Genesis LLM Prototype", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🔧 Kelmoid Genesis LLM Prototype
        **AI-Powered CAD Engineering Suite for Design, Analysis, and Manufacturing**
        
        This suite includes:
        - 🎨 **Text-to-CAD Generator**: Create 3D models from natural language
        - 📐 **2D Plate Designer**: Generate technical drawings and G-code
        - 🌊 **CFD Simulator**: Computational fluid dynamics analysis
        - 📋 **Orthographic Views**: Generate technical drawings from 3D models
        """)
        
        with gr.Tabs():
            
            # =====================================================
            # TAB 1: TEXT TO CAD
            # =====================================================
            with gr.TabItem("🎨 Text-to-CAD Generator"):
                gr.Markdown("### Convert natural language descriptions into 3D CAD models")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        cad_prompt = gr.Textbox(
                            label="Design Prompt",
                            placeholder="Enter your CAD design description (e.g., 'Create a steel cube 20x20x20mm')",
                            lines=3
                        )
                        cad_generate_btn = gr.Button("🚀 Generate CAD Model", variant="primary", size="lg")
                        
                        gr.Markdown("**Quick Examples:**")
                        with gr.Row():
                            gr.Button("Cube 20x20x20", size="sm").click(
                                lambda: "Create a cube with 20x20x20 dimensions", 
                                outputs=cad_prompt
                            )
                            gr.Button("Cylinder r=10, h=15", size="sm").click(
                                lambda: "Design a cylinder with radius 10mm and height 15mm", 
                                outputs=cad_prompt
                            )
                            gr.Button("Bearing OD=20, ID=10", size="sm").click(
                                lambda: "Make a bearing with outer diameter 20mm and inner diameter 10mm", 
                                outputs=cad_prompt
                            )
                            gr.Button("L-Bracket 30x20x15", size="sm").click(
                                lambda: "Create an L-shaped bracket 30x20x15mm", 
                                outputs=cad_prompt
                            )
                
                with gr.Row():
                    with gr.Column():
                        cad_3d_output = gr.Plot(label="Interactive 3D Model")
                    with gr.Column():
                        cad_ortho_output = gr.Image(label="Orthographic Views", type="pil")
                
                cad_summary_output = gr.Markdown(label="Generation Summary")
                
                cad_generate_btn.click(
                    fn=process_text_to_cad,
                    inputs=[cad_prompt],
                    outputs=[cad_3d_output, cad_ortho_output, cad_summary_output]
                )
            
            # =====================================================
            # TAB 2: 2D PLATE DESIGNER
            # =====================================================
            with gr.TabItem("📐 2D Plate Designer"):
                gr.Markdown("### Generate technical drawings and G-code for 2D plates")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        plate_description = gr.Textbox(
                            lines=5,
                            label="Plate Description",
                            placeholder="e.g., A 100mm x 50mm plate with a 20mm diameter hole and a 30mm long and 10mm wide slot"
                        )
                        plate_generate_btn = gr.Button("📋 Generate Plate Design", variant="primary")
                        
                        gr.Markdown("**Examples:**")
                        with gr.Column():
                            gr.Button("150x100mm plate with 25mm hole").click(
                                lambda: "A 150mm x 100mm plate with a 25mm diameter circular cutout",
                                outputs=plate_description
                            )
                            gr.Button("100x100mm plate with slot").click(
                                lambda: "A 100mm x 100mm plate with a 50mm long and 10mm wide slot",
                                outputs=plate_description
                            )
                            gr.Button("120x80mm plate with oval").click(
                                lambda: "A 120mm x 80mm plate with an oval hole 40mm long and 20mm wide",
                                outputs=plate_description
                            )
                
                with gr.Row():
                    with gr.Column():
                        plate_drawing_output = gr.Image(label="3-View Technical Drawing", type="pil")
                    with gr.Column():
                        plate_gcode_output = gr.Code(label="Generated G-Code")
                
                plate_status_output = gr.Textbox(label="Status")
                
                plate_generate_btn.click(
                    fn=process_plate_design,
                    inputs=[plate_description],
                    outputs=[plate_drawing_output, plate_gcode_output, plate_status_output]
                )
            
            # =====================================================
            # TAB 3: CFD SIMULATOR
            # =====================================================
            with gr.TabItem("🌊 kelmoid CFD Simulator"):
                gr.Markdown("### Computational Fluid Dynamics Simulation")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Domain Parameters:**")
                        cfd_length = gr.Number(label="Channel Length (m)", value=2.0, minimum=0.1, maximum=10.0)
                        cfd_height = gr.Number(label="Channel Height (m)", value=1.0, minimum=0.1, maximum=5.0)
                        cfd_grid_x = gr.Number(label="Grid Points X", value=41, minimum=21, maximum=101)
                        cfd_grid_y = gr.Number(label="Grid Points Y", value=21, minimum=11, maximum=51)
                        
                        gr.Markdown("**Flow Parameters:**")
                        cfd_inlet_vel = gr.Number(label="Inlet Velocity (m/s)", value=0.005, minimum=0.001, maximum=0.1)
                        cfd_density = gr.Number(label="Fluid Density (kg/m³)", value=1.0, minimum=0.1, maximum=10.0)
                        cfd_viscosity = gr.Number(label="Dynamic Viscosity (Pa·s)", value=0.05, minimum=0.001, maximum=1.0)
                        
                        gr.Markdown("**Obstacle (optional):**")
                        cfd_obs_x1 = gr.Number(label="Obstacle X1", value=0.5, minimum=0.0, maximum=2.0)
                        cfd_obs_y1 = gr.Number(label="Obstacle Y1", value=0.2, minimum=0.0, maximum=1.0)
                        cfd_obs_x2 = gr.Number(label="Obstacle X2", value=0.7, minimum=0.0, maximum=2.0)
                        cfd_obs_y2 = gr.Number(label="Obstacle Y2", value=0.8, minimum=0.0, maximum=1.0)
                        
                        cfd_max_iter = gr.Number(label="Max iterations", value=1000, minimum=100, maximum=5000)
                        cfd_simulate_btn = gr.Button("🌊 Run CFD Simulation", variant="primary")
                    
                    with gr.Column(scale=2):
                        cfd_result_output = gr.Image(label="CFD Results", type="pil")
                        cfd_status_output = gr.Textbox(label="Simulation Status")
                
                cfd_simulate_btn.click(
                    fn=process_cfd_simulation,
                    inputs=[cfd_length, cfd_height, cfd_grid_x, cfd_grid_y, cfd_inlet_vel, 
                           cfd_density, cfd_viscosity, cfd_obs_x1, cfd_obs_y1, cfd_obs_x2, 
                           cfd_obs_y2, cfd_max_iter],
                    outputs=[cfd_result_output, cfd_status_output]
                )
        
        gr.Markdown("""
        ---
        ### 📚 Usage Guide:
        
        **Text-to-CAD Generator:**
        - Supports shapes: cube, sphere, cylinder, cone, gear, bracket, plate, rod, washer, screw, nut, bearing, flange, pipe
        - Use dimension keywords: length, width, height, radius, diameter, thickness
        - Specify colors: red, blue, green, yellow, orange, purple, pink, brown, black, white, gray
        
        **2D Plate Designer:**
        - Describe plates with dimensions like "100mm x 50mm"
        - Add features: "20mm diameter hole", "30mm long and 10mm wide slot", "oval hole 40mm long and 20mm wide"
        - Generates technical drawings and CNC G-code
        
        **CFD Simulator:**
        - Simulates fluid flow through channels with optional obstacles
        - Adjust grid resolution for accuracy vs. speed
        - Lower viscosity = higher Reynolds number = more turbulent flow
        
        **Note:** This application runs on CPU and is optimized for educational and prototyping purposes.
        """)
    
    return demo

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )