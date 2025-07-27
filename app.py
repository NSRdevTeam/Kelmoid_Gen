# app.py
# Combined Engineering Suite for Hugging Face Deployment

# --- 1. IMPORTS ---
# Standard library imports
import re
import os
import io
import warnings

# Third-party imports
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from shapely.geometry import box, Point
from shapely.affinity import scale
import trimesh
import pyvista as pv

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 2. CORE LOGIC FROM PDFS ---

# === MODULE 1: Text to CAD & G-Code (from g-code.pdf) ===

def parse_simple_description(description):
    """Parses a textual description to extract geometric parameters for a plate."""
    width, height = 100, 100
    holes, slots, ovals = [], [], []

    # Dimensions
    dim_match = re.search(r'(\d+)\s*mm\s*[xX×]\s*(\d+)\s*mm', description)
    if dim_match:
        width, height = int(dim_match.group(1)), int(dim_match.group(2))

    # Holes
    for hole_match in re.finditer(r'(hole|circle|circular cutout)[^\d]*(\d+)\s*mm', description):
        holes.append({'x': width / 2, 'y': height / 2, 'diameter': int(hole_match.group(2))})

    # Slots
    for slot_match in re.finditer(r'(\d+)\s*mm\s+long\s+and\s+(\d+)\s*mm\s+wide\s+slot', description):
        slots.append({'x': width / 2, 'y': height / 2, 'length': int(slot_match.group(1)), 'width': int(slot_match.group(2))})

    # Ovals
    for oval_match in re.finditer(r'oval\s+hole\s+(\d+)\s*mm\s+long\s+and\s+(\d+)\s*mm\s+wide', description):
        ovals.append({'x': width / 2, 'y': height / 2, 'length': int(oval_match.group(1)), 'width': int(oval_match.group(2))})

    return {"width": width, "height": height, "holes": holes, "slots": slots, "ovals": ovals}

def generate_simple_3_view_drawing(description):
    """Generates a 3-view engineering drawing from a simple description."""
    parsed = parse_simple_description(description)
    width, height, depth = parsed["width"], parsed["height"], 5
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.style.use('grayscale')

    views = ['Top View', 'Front View', 'Side View']
    for ax, view in zip(axes, views):
        ax.set_title(view)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')

        if view == "Top View":
            shape = box(0, 0, width, height)
            x, y = shape.exterior.xy
            ax.plot(x, y, color='black')
            for hole in parsed["holes"]:
                ax.plot(*Point(hole['x'], hole['y']).buffer(hole['diameter'] / 2).exterior.xy, color='black')
            for slot in parsed["slots"]:
                ax.plot(*scale(Point(slot['x'], slot['y']).buffer(1), slot['length']/2, slot['width']/2).exterior.xy, color='black')
            for oval in parsed["ovals"]:
                ax.plot(*scale(Point(oval['x'], oval['y']).buffer(1), oval['length']/2, oval['width']/2).exterior.xy, color='black')
            ax.set_xlim(-10, width + 10)
            ax.set_ylim(-10, height + 10)
        elif view == "Front View":
            ax.plot(*box(0, 0, width, depth).exterior.xy, color='black')
            ax.set_xlim(-10, width + 10)
            ax.set_ylim(-10, depth + 10)
        elif view == "Side View":
            ax.plot(*box(0, 0, height, depth).exterior.xy, color='black')
            ax.set_xlim(-10, height + 10)
            ax.set_ylim(-10, depth + 10)

    plt.tight_layout()
    drawing_path = "simple_3_view_drawing.png"
    fig.savefig(drawing_path)
    plt.close(fig)
    return drawing_path

def generate_gcode(description):
    """Generates G-code for milling the part based on the description."""
    parsed = parse_simple_description(description)
    w, h = parsed['width'], parsed['height']
    gcode = [
        "G21 ; Set units to mm", "G90 ; Use absolute positioning", "G17 ; Select XY plane",
        "M3 S1000 ; Start spindle", "G0 Z5 ; Lift Z to a safe height",
        "\n; --- Mill Outer Profile ---",
        "G0 X0 Y0", "G1 Z-1 F100", f"G1 X{w} F300", f"G1 Y{h}", f"G1 X0", "G1 Y0", "G0 Z5"
    ]
    for hole in parsed['holes']:
        x, y, r = hole['x'], hole['y'], hole['diameter'] / 2
        gcode.extend([f"\n; --- Mill Hole at X{x}, Y{y}, D{hole['diameter']} ---", f"G0 X{x - r} Y{y}", "G1 Z-1 F100", f"G2 I{r} J0 F200", "G0 Z5"])
    for slot in parsed['slots']:
        x, y, l, w_slot = slot['x'], slot['y'], slot['length'], slot['width']
        r = w_slot / 2
        x_start, x_end = x - (l - w_slot) / 2, x + (l - w_slot) / 2
        gcode.extend([f"\n; --- Mill Slot at center X{x}, Y{y} ---", f"G0 X{x_start} Y{y - r}", "G1 Z-1 F100", f"G1 X{x_end} F200", f"G2 I0 J{r}", f"G1 X{x_start}", f"G2 I0 J{r}", "G0 Z5"])
    for oval in parsed['ovals']:
        gcode.append(f"\n; --- Oval hole at X{oval['x']}, Y{oval['y']} (manual operation needed) ---")
    gcode.extend(["\nM5 ; Stop spindle", "G0 X0 Y0 ; Return to home", "M30 ; End of program"])
    return "\n".join(gcode)

def export_svg(description):
    """Exports a 2D drawing of the part top-view as an SVG file."""
    parsed = parse_simple_description(description)
    width, height = parsed["width"], parsed["height"]
    fig, ax = plt.subplots(figsize=(width/25.4, height/25.4)) # Inches
    ax.set_aspect('equal')
    ax.plot(*box(0, 0, width, height).exterior.xy, color='black', linewidth=2)
    for hole in parsed["holes"]:
        ax.plot(*Point(hole['x'], hole['y']).buffer(hole['diameter'] / 2).exterior.xy, color='black', linewidth=1.5)
    ax.set_xlim(-10, width + 10)
    ax.set_ylim(-10, height + 10)
    ax.axis('off')
    svg_path = "drawing.svg"
    fig.savefig(svg_path, format="svg", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return svg_path

def process_simple_cad(description):
    """Top-level function for the simple CAD generator."""
    if not description.strip():
        return None, "Please enter a description.", None
    try:
        drawing_path = generate_simple_3_view_drawing(description)
        gcode = generate_gcode(description)
        svg_path = export_svg(description)
        return drawing_path, gcode, svg_path
    except Exception as e:
        return None, f"An error occurred: {e}", None

# === MODULE 2: Advanced Text-to-CAD (from text to cad.pdf) ===
class TextToCADGenerator:
    """Text-to-CAD generator using procedural geometry."""
    def __init__(self):
        self.shapes_library = {
            'cube': self._create_cube, 'box': self._create_cube,
            'sphere': self._create_sphere, 'ball': self._create_sphere,
            'cylinder': self._create_cylinder, 'tube': self._create_cylinder,
            'cone': self._create_cone, 'plate': self._create_plate,
            'washer': self._create_washer, 'bracket': self._create_bracket,
        }

    def parse_prompt(self, prompt: str) -> dict:
        prompt = prompt.lower().strip()
        dimensions = self._extract_dimensions(prompt)
        shape_type = next((shape for shape in self.shapes_library if shape in prompt), 'cube')
        color = next((c for c in ['red', 'blue', 'green', 'yellow', 'gray'] if c in prompt), 'lightblue')
        return {'shape': shape_type, 'dimensions': dimensions, 'color': color, 'prompt': prompt}

    def _extract_dimensions(self, prompt: str) -> dict:
        dims = {'length': 10, 'width': 10, 'height': 10, 'radius': 5, 'diameter': 10}
        patterns = {
            'length': r'length\s*[:=]?\s*(\d+\.?\d*)', 'width': r'width\s*[:=]?\s*(\d+\.?\d*)',
            'height': r'height\s*[:=]?\s*(\d+\.?\d*)', 'radius': r'radius\s*[:=]?\s*(\d+\.?\d*)',
            'diameter': r'diameter\s*[:=]?\s*(\d+\.?\d*)', 'thick': r'thick\w*\s*[:=]?\s*(\d+\.?\d*)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                dims[key] = float(match.group(1))
        return dims

    def generate_3d_model(self, params: dict) -> trimesh.Trimesh:
        shape_func = self.shapes_library.get(params['shape'], self._create_cube)
        return shape_func(params['dimensions'])

    def generate_2d_drawing(self, params: dict) -> Image.Image:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Technical Drawing: {params['shape'].title()}", fontsize=16)
        dims = params['dimensions']
        # Simplified drawing logic
        self._draw_rectangular_part(ax1, ax2, ax3, dims, params['shape'])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img

    def _draw_rectangular_part(self, ax1, ax2, ax3, dims, shape):
        l, w, h = dims['length'], dims['width'], dims['height']
        views = {"Front View": (w, h), "Top View": (w, l), "Side View": (l, h)}
        axes = {"Front View": ax1, "Top View": ax2, "Side View": ax3}
        for title, (dim1, dim2) in views.items():
            ax = axes[title]
            ax.set_title(title)
            ax.add_patch(plt.Rectangle((0, 0), dim1, dim2, fill=False, edgecolor='black', linewidth=2))
            ax.set_xlim(-dim1*0.1, dim1*1.1)
            ax.set_ylim(-dim2*0.1, dim2*1.1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(f"Dim 1: {dim1:.2f}")
            ax.set_ylabel(f"Dim 2: {dim2:.2f}")

    def generate_3d_visualization(self, mesh: trimesh.Trimesh, color: str = 'lightblue') -> go.Figure:
        fig = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            color=color, opacity=0.9
        )])
        fig.update_layout(title="3D CAD Model", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        return fig

    # Shape creation methods
    def _create_cube(self, dims: dict) -> trimesh.Trimesh:
        return trimesh.creation.box(extents=[dims['length'], dims['width'], dims['height']])
    def _create_sphere(self, dims: dict) -> trimesh.Trimesh:
        return trimesh.creation.icosphere(subdivisions=3, radius=dims.get('radius', dims.get('diameter', 10) / 2))
    def _create_cylinder(self, dims: dict) -> trimesh.Trimesh:
        return trimesh.creation.cylinder(radius=dims.get('radius', 5), height=dims.get('height', 10))
    def _create_cone(self, dims: dict) -> trimesh.Trimesh:
        return trimesh.creation.cone(radius=dims.get('radius', 5), height=dims.get('height', 10))
    def _create_plate(self, dims: dict) -> trimesh.Trimesh:
        return trimesh.creation.box(extents=[dims['length'], dims['width'], dims.get('height', 2)])
    def _create_washer(self, dims: dict) -> trimesh.Trimesh:
        outer = trimesh.creation.cylinder(radius=dims.get('radius', 10), height=dims.get('height', 2))
        inner = trimesh.creation.cylinder(radius=dims.get('radius', 10) * 0.5, height=dims.get('height', 2) * 1.1)
        return outer.difference(inner)
    def _create_bracket(self, dims: dict) -> trimesh.Trimesh:
        l, w, h, t = dims.get('length', 20), dims.get('width', 15), dims.get('height', 20), dims.get('thick', 3)
        base = trimesh.creation.box(extents=[l, w, t])
        base.apply_translation([l/2, w/2, t/2])
        wall = trimesh.creation.box(extents=[t, w, h])
        wall.apply_translation([t/2, w/2, h/2])
        return base.union(wall)

cad_generator = TextToCADGenerator()

def process_advanced_text_to_cad(prompt: str):
    """Main function for the advanced CAD generator."""
    try:
        params = cad_generator.parse_prompt(prompt)
        mesh_3d = cad_generator.generate_3d_model(params)
        drawing_2d = cad_generator.generate_2d_drawing(params)
        fig_3d = cad_generator.generate_3d_visualization(mesh_3d, params['color'])
        summary = f"*Shape:* {params['shape'].title()}\n*Dimensions:* {params['dimensions']}"
        return drawing_2d, fig_3d, summary
    except Exception as e:
        return None, go.Figure().add_annotation(text=f"Error: {e}", showarrow=False), f"Error: {e}"


# === MODULE 3: 3D to Orthographic View (from cad to ortho.pdf) ===

def generate_ortho_views(uploaded_file):
    """Generates orthographic views from an uploaded 3D model file."""
    if uploaded_file is None:
        return None, "Please upload a 3D model file."
    try:
        # PyVista reads from a file path
        mesh = pv.read(uploaded_file.name)

        # Generate base views with PyVista
        views, filenames = ['xy', 'xz', 'yz'], ['front.png', 'top.png', 'side.png']
        plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
        plotter.background_color = 'white'
        plotter.enable_parallel_projection()
        for i, view in enumerate(views):
            plotter.clear()
            plotter.add_mesh(mesh.extract_feature_edges(), color='black', line_width=2)
            plotter.camera_position = view
            plotter.reset_camera()
            plotter.camera.zoom(1.2)
            plotter.screenshot(filenames[i])

        # Combine and add dimensions with Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('white')
        bounds = mesh.bounds
        x_dim, y_dim, z_dim = bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]
        dims = [(x_dim, y_dim), (x_dim, z_dim), (y_dim, z_dim)]
        view_titles = ["Front View (XY)", "Top View (XZ)", "Side View (YZ)"]

        for i, ax in enumerate(axes):
            img = plt.imread(filenames[i])
            ax.imshow(img)
            ax.set_title(view_titles[i], fontsize=12, pad=10)
            ax.axis('off')
            dim1, dim2 = dims[i]
            # Add annotations
            ax.text(0.5, 0.05, f"Width: {dim1:.2f}", transform=ax.transAxes, ha='center', va='bottom', fontsize=10)
            ax.text(0.05, 0.5, f"Height: {dim2:.2f}", transform=ax.transAxes, ha='left', va='center', rotation=90, fontsize=10)

        final_path = 'orthographic_views_with_dimensions.png'
        plt.tight_layout()
        plt.savefig(final_path, dpi=150)
        plt.close(fig)
        return final_path, f"Successfully generated views for {os.path.basename(uploaded_file.name)}."

    except Exception as e:
        return None, f"An error occurred: {e}"


# === MODULE 4: 2D CFD Simulation (from cfd.pdf) ===

def run_cfd_simulation(Lx, Ly, Nx, Ny, inlet_velocity_u, rho, mu, ox1, oy1, ox2, oy2, max_iter, p_iter):
    """Runs a 2D CFD simulation and returns the result plot."""
    try:
        # Grid and Initialization
        dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
        x, y = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)
        u, v, p = np.zeros((Ny, Nx)), np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
        
        # Obstacle
        obstacle_mask = np.zeros((Ny, Nx), dtype=bool)
        if ox1 < ox2 and oy1 < oy2:
            i1, i2 = int(ox1 / dx), int(ox2 / dx)
            j1, j2 = int(oy1 / dy), int(oy2 / dy)
            obstacle_mask[j1:j2, i1:i2] = True

        # Main Loop (SIMPLE algorithm simplified)
        nu = mu / rho
        udiff, stepcount = 1.0, 0
        alpha_p, alpha_uv = 0.1, 0.5 # Relaxation factors

        for it in range(max_iter):
            un, vn = u.copy(), v.copy()
            
            # Momentum predictor (simplified)
            # Pressure correction (simplified Poisson equation)
            p_prime = np.zeros_like(p)
            for _ in range(p_iter):
                pn_prime = p_prime.copy()
                p_prime[1:-1, 1:-1] = ((pn_prime[1:-1, 2:] + pn_prime[1:-1, :-2]) * dy**2 +
                                       (pn_prime[2:, 1:-1] + pn_prime[:-2, 1:-1]) * dx**2) / \
                                      (2 * (dx**2 + dy**2))
                # BCs for p_prime
                p_prime[:, -1] = 0 # Outlet
                p_prime[:, 0] = p_prime[:, 1]
                p_prime[0, :] = p_prime[1, :]
                p_prime[-1, :] = p_prime[-2, :]
                
            p += alpha_p * p_prime
            
            # Velocity corrector
            u[1:-1, 1:-1] -= alpha_uv * (p_prime[1:-1, 1:-1] - p_prime[1:-1, :-2]) / dx
            v[1:-1, 1:-1] -= alpha_uv * (p_prime[1:-1, 1:-1] - p_prime[:-2, 1:-1]) / dy

            # Boundary Conditions
            u[:, 0], v[:, 0] = inlet_velocity_u, 0
            u[:, -1], v[:, -1] = u[:, -2], v[:, -2]
            u[0, :], v[0, :] = 0, 0
            u[-1, :], v[-1, :] = 0, 0
            u[obstacle_mask], v[obstacle_mask] = 0, 0

            udiff = np.linalg.norm(u - un) / (np.linalg.norm(un) + 1e-6)
            if udiff < 1e-6: break

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        velocity_magnitude = np.sqrt(u**2 + v**2)
        velocity_magnitude[obstacle_mask] = np.nan
        
        cf = ax.contourf(X, Y, velocity_magnitude, levels=50, cmap=cm.jet)
        fig.colorbar(cf, label='Velocity Magnitude (m/s)')
        ax.streamplot(X, Y, u, v, color='black', linewidth=0.7, density=1.5)
        if np.any(obstacle_mask):
            ax.contour(X, Y, obstacle_mask, levels=[0.5], colors='grey', linewidths=3)
            
        ax.set_title(f'CFD Simulation Results after {it+1} iterations')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error during CFD simulation:\n{e}", ha='center', va='center')
        return fig

# --- 3. GRADIO UI ---

def create_gradio_interface():
    """Create and launch the main Gradio interface."""
    
    css = """
    .gradio-container { max-width: 1280px !important; margin: auto; }
    .gr-tabs { border: 1px solid #E0E0E0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer { display: none !important; }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Engineering Suite") as demo:
        gr.Markdown("# 🛠️ Engineering Suite")
        gr.Markdown("A collection of tools for CAD generation, G-Code, and simulation. Created by combining multiple code sources.")

        with gr.Tabs():
            # --- Tab 1: Simple Text to CAD & G-Code ---
            with gr.TabItem("Text to CAD & G-Code"):
                gr.Markdown("### Describe a simple 2D part to generate drawings and G-Code.")
                with gr.Row():
                    with gr.Column(scale=1):
                        simple_cad_input = gr.Textbox(
                            lines=4,
                            label="Part Description",
                            placeholder="e.g., A 100mm x 50mm plate with a 20mm diameter hole and a 30mm long and 10mm wide slot"
                        )
                        simple_cad_button = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        simple_cad_img_output = gr.Image(label="3-View Drawing")
                with gr.Row():
                    simple_cad_gcode_output = gr.Code(label="Generated G-Code", language="gcode")
                    simple_cad_svg_output = gr.File(label="Download SVG Drawing")
                
                gr.Examples(
                    examples=[
                        ["A 150mm x 100mm plate with a 25mm diameter circular cutout"],
                        ["A 100mm x 100mm plate with a 50mm long and 10mm wide slot"],
                    ],
                    inputs=simple_cad_input
                )

            # --- Tab 2: Advanced Text-to-CAD ---
            with gr.TabItem("Advanced Text-to-CAD"):
                gr.Markdown("### Generate a 3D model and technical drawing from a more detailed description.")
                with gr.Row():
                    with gr.Column(scale=1):
                        adv_cad_input = gr.Textbox(lines=4, label="Design Prompt", placeholder="e.g., Create a blue bracket with length 50, width 30, height 40 and thickness 5")
                        adv_cad_button = gr.Button("Generate 3D Model", variant="primary")
                        adv_cad_summary = gr.Markdown(label="Generation Summary")
                    with gr.Column(scale=2):
                        adv_cad_img_output = gr.Image(label="2D Technical Drawing")
                        adv_cad_3d_output = gr.Plot(label="Interactive 3D Model")

                gr.Examples(
                    examples=[
                        ["A red cube with length 20, width 30, height 15"],
                        ["A green cylinder with radius 10 and height 40"],
                        ["A gray washer with radius 15 and height 3"],
                    ],
                    inputs=adv_cad_input
                )

            # --- Tab 3: 3D to Orthographic View ---
            with gr.TabItem("3D to Orthographic View"):
                gr.Markdown("### Upload a 3D model file (.stl, .obj) to generate its dimensioned orthographic views.")
                with gr.Row():
                    with gr.Column(scale=1):
                        ortho_input = gr.File(label="Upload 3D Model (.stl, .obj, .ply)")
                        ortho_button = gr.Button("Generate Ortho Views", variant="primary")
                        ortho_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=2):
                        ortho_output = gr.Image(label="Orthographic Views with Dimensions")

            # --- Tab 4: 2D CFD Simulation ---
            with gr.TabItem("2D CFD Simulation"):
                gr.Markdown("### Configure and run a 2D channel flow simulation.")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Domain & Grid**")
                        cfd_Lx = gr.Slider(1.0, 5.0, value=2.0, label="Channel Length (Lx)")
                        cfd_Ly = gr.Slider(0.5, 2.0, value=1.0, label="Channel Height (Ly)")
                        cfd_Nx = gr.Slider(31, 101, value=61, step=10, label="Grid Points X (Nx)")
                        cfd_Ny = gr.Slider(21, 81, value=41, step=10, label="Grid Points Y (Ny)")
                        gr.Markdown("**Fluid Properties**")
                        cfd_vel = gr.Slider(0.001, 0.1, value=0.01, label="Inlet Velocity (m/s)")
                        cfd_rho = gr.Slider(1.0, 1000.0, value=1.0, label="Density (kg/m^3)")
                        cfd_mu = gr.Slider(0.001, 0.1, value=0.02, label="Viscosity (Pa.s)")
                    with gr.Column():
                        gr.Markdown("**Obstacle (relative to domain)**")
                        cfd_ox1 = gr.Slider(0.0, 1.0, value=0.4, label="Obstacle X1")
                        cfd_oy1 = gr.Slider(0.0, 1.0, value=0.4, label="Obstacle Y1")
                        cfd_ox2 = gr.Slider(0.0, 1.0, value=0.6, label="Obstacle X2")
                        cfd_oy2 = gr.Slider(0.0, 1.0, value=0.6, label="Obstacle Y2")
                        gr.Markdown("**Solver Settings**")
                        cfd_max_iter = gr.Slider(100, 5000, value=500, step=100, label="Max Iterations")
                        cfd_p_iter = gr.Slider(20, 200, value=50, step=10, label="Pressure Solver Iterations")
                        cfd_button = gr.Button("Run Simulation", variant="primary")
                cfd_output = gr.Plot(label="CFD Result")

        # --- Event Handlers ---
        simple_cad_button.click(
            fn=process_simple_cad,
            inputs=[simple_cad_input],
            outputs=[simple_cad_img_output, simple_cad_gcode_output, simple_cad_svg_output]
        )
        adv_cad_button.click(
            fn=process_advanced_text_to_cad,
            inputs=[adv_cad_input],
            outputs=[adv_cad_img_output, adv_cad_3d_output, adv_cad_summary]
        )
        ortho_button.click(
            fn=generate_ortho_views,
            inputs=[ortho_input],
            outputs=[ortho_output, ortho_status]
        )
        cfd_button.click(
            fn=run_cfd_simulation,
            inputs=[cfd_Lx, cfd_Ly, cfd_Nx, cfd_Ny, cfd_vel, cfd_rho, cfd_mu, cfd_ox1, cfd_oy1, cfd_ox2, cfd_oy2, cfd_max_iter, cfd_p_iter],
            outputs=[cfd_output]
        )

    return demo

# --- 4. LAUNCH THE APP ---
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()
