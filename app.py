import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import trimesh
from shapely.geometry import box, Point
from shapely.affinity import scale
from scipy.spatial import ConvexHull
import re
import io
import os
import tempfile
import base64
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Optional LLM Text-to-CAD engine
try:
    from llm_engine import LLMTextToCAD, LLM_AVAILABLE
except Exception:
    LLMTextToCAD = None
    LLM_AVAILABLE = False

# Enhanced imports for parametric CAD
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
    print("✅ CadQuery available - Parametric CAD enabled")
except ImportError:
    CADQUERY_AVAILABLE = False
    print("⚠️  CadQuery not available - Using Trimesh fallbacks")

# Properly setup boolean backend with manifold3d
BOOL_BACKEND = None
try:
    import manifold3d
    print("✅ Manifold3D imported successfully")
    
    # Test if manifold3d is working by creating a simple boolean operation
    try:
        cube1 = trimesh.creation.box(extents=[1, 1, 1])
        cube2 = trimesh.creation.box(extents=[1, 1, 1])
        cube2.apply_translation([0.5, 0.5, 0.5])
        
        # Try manifold engine specifically
        result = trimesh.boolean.union([cube1, cube2], engine='manifold')
        if result is not None and hasattr(result, 'vertices'):
            BOOL_BACKEND = 'manifold'
            print("✅ Manifold3D boolean backend working correctly")
        else:
            raise Exception("Manifold returned None or invalid result")
            
    except Exception as e:
        print(f"⚠️ Manifold3D specific test failed: {e}")
        # Try auto-detection (trimesh will choose best available backend)
        try:
            result = trimesh.boolean.union([cube1, cube2])
            if result is not None and hasattr(result, 'vertices'):
                BOOL_BACKEND = 'auto'
                print("✅ Trimesh boolean backend (auto) working")
            else:
                print("⚠️ Auto-detection returned None result")
        except Exception as e2:
            print(f"⚠️ Auto boolean backend also failed: {e2}")
            # Set a fallback - we'll use safe operations
            BOOL_BACKEND = 'fallback'
            print("⚠️ Using fallback boolean operations")
            
except ImportError as e:
    print(f"⚠️ Manifold3D import failed: {e}")
    print("To install: pip install manifold3d")
    BOOL_BACKEND = 'fallback'
    print("⚠️ Using fallback boolean operations")

print(f"✅ Boolean backend active: {BOOL_BACKEND}")

# =====================================================
# ENHANCED CAD UTILITIES
# =====================================================

def cq_to_trimesh(cq_obj):
    """Export CadQuery object to temporary STL then load as trimesh.Trimesh"""
    if not CADQUERY_AVAILABLE:
        raise RuntimeError("CadQuery not available")
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cq.exporters.export(cq_obj, tmp_path, exportType="STL")
        mesh = trimesh.load(tmp_path)
        if hasattr(mesh, "is_watertight") and not mesh.is_watertight:
            try:
                mesh = mesh.fill_holes()
            except Exception:
                pass
        return mesh
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def safe_union(a, b):
    """Enhanced union with multiple fallback strategies"""
    if not isinstance(a, trimesh.Trimesh) or not isinstance(b, trimesh.Trimesh):
        print("Warning: Non-Trimesh objects in union operation")
        if hasattr(a, "union"):
            try:
                return a.union(b)
            except Exception:
                pass
        return a
    
    # Try with the detected backend first
    if BOOL_BACKEND == 'manifold':
        try:
            return trimesh.boolean.union([a, b], engine='manifold')
        except Exception as e:
            print(f"Manifold union failed: {e}, trying fallback")
    
    # Try auto-detection if not fallback mode
    if BOOL_BACKEND != 'fallback':
        try:
            return trimesh.boolean.union([a, b])
        except Exception as e:
            print(f"Auto union failed: {e}, using concatenation fallback")
    
    # Fallback: concatenate meshes
    try:
        combined = trimesh.util.concatenate([a, b])
        return combined
    except Exception as e:
        print(f"Concatenation fallback failed: {e}, returning first mesh")
        return a

def safe_difference(a, b):
    """Enhanced difference with multiple fallback strategies"""
    if not isinstance(a, trimesh.Trimesh) or not isinstance(b, trimesh.Trimesh):
        print("Warning: Non-Trimesh objects in difference operation")
        if hasattr(a, "difference"):
            try:
                return a.difference(b)
            except Exception:
                pass
        return a
    
    # Try with the detected backend first
    if BOOL_BACKEND == 'manifold':
        try:
            return trimesh.boolean.difference([a, b], engine='manifold')
        except Exception as e:
            print(f"Manifold difference failed: {e}, trying fallback")
    
    # Try auto-detection if not fallback mode
    if BOOL_BACKEND != 'fallback':
        try:
            return trimesh.boolean.difference([a, b])
        except Exception as e:
            print(f"Auto difference failed: {e}, returning original mesh")
    
    # Fallback: return original mesh (can't do difference without boolean backend)
    print("Warning: No boolean backend available for difference operation, returning original mesh")
    return a

def export_mesh(mesh, filename):
    """Export mesh to various formats"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.stl', '.obj', '.ply', '.glb']:
        try:
            mesh.export(filename)
            return filename
        except Exception as e:
            raise RuntimeError(f"Export failed: {str(e)}")
    else:
        raise RuntimeError("Unsupported export format. Use .stl, .obj, .ply, .glb")

# =====================================================
# ENHANCED PROMPT PARSING
# =====================================================

DIM_FIELDS = [
    'length', 'width', 'height', 'radius', 'diameter', 'thickness',
    'outer_radius', 'inner_radius', 'depth', 'size', 'panel_count', 
    'frame_width', 'spacing', 'num_shelves', 'rail_width', 'leg_size', 
    'panel_thickness', 'hole_radius', 'wall_thickness'
]

def extract_key_values(prompt):
    """Extract key=value pairs (numeric) into dict"""
    kv = {}
    for m in re.finditer(r'([a-zA-Z_]+)\s*[:=]\s*([0-9]+\.?[0-9]*)', prompt):
        k = m.group(1).lower()
        v = float(m.group(2))
        kv[k] = v
    return kv

def extract_x_pattern(prompt):
    """Extract patterns like 10x20x30 (assumes order length x width x height)"""
    m = re.search(r'(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)', prompt)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None

def filter_numeric(dims):
    """Return filtered dict containing only numeric values expected by builders"""
    out = {}
    for k, v in dims.items():
        if v is None:
            continue
        if k in ('panel_count', 'mullions_v', 'mullions_h', 'num_shelves'):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out

# =====================================================
# TEXT TO CAD PARSER AND GENERATOR
# =====================================================

class TextToCADGenerator:
    """Enhanced Text-to-CAD generator with all shapes"""
    def __init__(self):
        self.shapes_library = {
            # Basic shapes
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
            # Mechanical parts
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
            'pipe': self._create_pipe,
            # Architectural frames
            'doorframe': self._create_door_frame,
            'door_frame': self._create_door_frame,
            'windowframe': self._create_window_frame,
            'window_frame': self._create_window_frame,
            'gypsumframe': self._create_gypsum_frame,
            'gypsum_frame': self._create_gypsum_frame,
            'drywall_frame': self._create_gypsum_frame,
            # Furniture frames
            'bedframe': self._create_bed_frame,
            'bed_frame': self._create_bed_frame,
            'tableframe': self._create_table_frame,
            'table_frame': self._create_table_frame,
            'chairframe': self._create_chair_frame,
            'chair_frame': self._create_chair_frame,
            'shelfframe': self._create_shelf_frame,
            'shelf_frame': self._create_shelf_frame,
            'cabinetframe': self._create_cabinet_frame,
            'cabinet_frame': self._create_cabinet_frame,
            # Enhanced parametric builders
            'water_tank': self._create_water_tank,
            'watertank': self._create_water_tank,
            'tank': self._create_water_tank,
            'parametric_washer': self._create_parametric_washer,
            'parametric_nut': self._create_parametric_nut,
            'parametric_bracket': self._create_parametric_bracket,
            'parametric_door': self._create_parametric_door,
            'parametric_window': self._create_parametric_window
        }

    def parse_prompt(self, prompt: str):
        """Enhanced prompt parsing with key=value and pattern recognition"""
        p = prompt.lower().strip()
        
        # Start with enhanced dimension extraction
        dimensions = self._extract_dimensions(p)
        
        # Add key=value parsing
        kv_pairs = extract_key_values(p)
        for k, v in kv_pairs.items():
            if k in dimensions:
                dimensions[k] = v
            elif k == 'panels':
                dimensions['panel_count'] = int(v)
            elif k == 'frames':
                dimensions['frame_width'] = v
        
        # Add NxMxP pattern recognition
        pattern = extract_x_pattern(p)
        if pattern:
            dimensions['length'], dimensions['width'], dimensions['height'] = pattern
        
        # Enhanced shape identification
        shape_type = None
        for shape in self.shapes_library.keys():
            if shape.replace('_', ' ') in p or shape in p:
                shape_type = shape
                break
        
        # Additional keyword mapping
        if shape_type is None:
            for keyword in ['tank', 'panel']:
                if keyword in p:
                    if keyword == 'tank':
                        shape_type = 'water_tank' if 'water_tank' in self.shapes_library else 'cylinder'
                    elif keyword == 'panel':
                        shape_type = 'gypsum_panel' if 'gypsum_panel' in self.shapes_library else 'plate'
        
        if not shape_type:
            shape_type = 'cube'
        
        color = self._extract_color(p)
        
        # Detect precision preference
        precision = 'high' if any(word in p for word in ['precise', 'parametric', 'high', 'accurate']) else 'fast'
        if any(word in p for word in ['fast', 'approx', 'quick']):
            precision = 'fast'
        
        return {
            'shape': shape_type,
            'dimensions': dimensions,
            'color': color,
            'precision': precision,
            'prompt': prompt
        }

    def _extract_dimensions(self, prompt: str):
        """Extract dimensions from prompt"""
        dimensions = {
            'length': 10, 'width': 10, 'height': 10, 'radius': 5, 'diameter': 10,
            'depth': 10, 'thickness': 5, 'spacing': 400, 'num_shelves': 4,
            'rail_width': 80, 'leg_size': 50, 'panel_thickness': 18
        }
        
        patterns = {
            'length': r'length\s*[:=]?\s*(\d+\.?\d*)',
            'width': r'width\s*[:=]?\s*(\d+\.?\d*)',
            'height': r'height\s*[:=]?\s*(\d+\.?\d*)',
            'radius': r'radius\s*[:=]?\s*(\d+\.?\d*)',
            'diameter': r'diameter\s*[:=]?\s*(\d+\.?\d*)',
            'depth': r'depth\s*[:=]?\s*(\d+\.?\d*)',
            'size': r'size\s*[:=]?\s*(\d+\.?\d*)',
            'thick': r'thick\w*\s*[:=]?\s*(\d+\.?\d*)',
            'spacing': r'spacing\s*[:=]?\s*(\d+\.?\d*)',
            'shelves': r'shelves\s*[:=]?\s*(\d+)',
            'rail': r'rail\s*[:=]?\s*(\d+\.?\d*)',
            'leg': r'leg\s*[:=]?\s*(\d+\.?\d*)',
            'panel': r'panel\s*[:=]?\s*(\d+\.?\d*)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if key == 'size':
                    dimensions['length'] = dimensions['width'] = dimensions['height'] = value
                elif key == 'thick':
                    dimensions['thickness'] = value
                elif key == 'shelves':
                    dimensions['num_shelves'] = int(value)
                elif key == 'rail':
                    dimensions['rail_width'] = value
                elif key == 'leg':
                    dimensions['leg_size'] = value
                elif key == 'panel':
                    dimensions['panel_thickness'] = value
                else:
                    dimensions[key] = value
        
        # Handle patterns like "10x20x30" or "10mm x 20mm x 30mm"
        dimension_patterns = [
            r'(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)',  # "10x20x30"
            r'(\d+\.?\d*)\s*(?:mm|cm|inches?)\s*[x×]\s*(\d+\.?\d*)\s*(?:mm|cm|inches?)\s*[x×]\s*(\d+\.?\d*)\s*(?:mm|cm|inches?)',  # "10mm x 20mm x 30mm"
        ]
        
        for pattern in dimension_patterns:
            dimension_match = re.search(pattern, prompt, re.IGNORECASE)
            if dimension_match:
                dimensions['length'] = float(dimension_match.group(1))
                dimensions['width'] = float(dimension_match.group(2))
                dimensions['height'] = float(dimension_match.group(3))
                break
        
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
        try:
            outer = trimesh.creation.cylinder(radius=outer_radius, height=height)
            inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
            return outer.difference(inner)
        except Exception:
            # Fallback: return outer cylinder if boolean operation fails
            return trimesh.creation.cylinder(radius=outer_radius, height=height)

    def _create_screw(self, dims):
        radius = dims.get('radius', 3)
        length = dims.get('length', 20)
        head_radius = radius * 1.5
        head_height = radius
        
        try:
            body = trimesh.creation.cylinder(radius=radius, height=length)
            head = trimesh.creation.cylinder(radius=head_radius, height=head_height)
            head = head.apply_translation([0, 0, length/2 + head_height/2])
            return body.union(head)
        except Exception:
            # Fallback: return just the body cylinder if union fails
            return trimesh.creation.cylinder(radius=radius, height=length)

    def _create_nut(self, dims):
        radius = dims.get('radius', 5)
        height = dims.get('height', 4)
        inner_radius = radius * 0.4
        
        try:
            outer = trimesh.creation.cylinder(radius=radius, height=height, sections=6)
            inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
            return outer.difference(inner)
        except Exception:
            # Fallback: return hexagonal cylinder without hole
            return trimesh.creation.cylinder(radius=radius, height=height, sections=6)

    def _create_bearing(self, dims):
        outer_radius = dims.get('radius', 10)
        inner_radius = outer_radius * 0.6
        height = dims.get('height', 5)
        try:
            outer = trimesh.creation.cylinder(radius=outer_radius, height=height)
            inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
            return outer.difference(inner)
        except Exception:
            # Fallback: return outer cylinder without hole
            return trimesh.creation.cylinder(radius=outer_radius, height=height)

    def _create_flange(self, dims):
        outer_radius = dims.get('radius', 15)
        inner_radius = outer_radius * 0.4
        height = dims.get('height', 5)
        try:
            outer = trimesh.creation.cylinder(radius=outer_radius, height=height)
            inner = trimesh.creation.cylinder(radius=inner_radius, height=height * 1.1)
            return outer.difference(inner)
        except Exception:
            # Fallback: return outer cylinder without hole
            return trimesh.creation.cylinder(radius=outer_radius, height=height)

    def _create_pipe(self, dims):
        outer_radius = dims.get('radius', 10)
        inner_radius = outer_radius * 0.8
        length = dims.get('length', 30)
        try:
            outer = trimesh.creation.cylinder(radius=outer_radius, height=length)
            inner = trimesh.creation.cylinder(radius=inner_radius, height=length * 1.1)
            return outer.difference(inner)
        except Exception:
            # Fallback: return solid cylinder without hole
            return trimesh.creation.cylinder(radius=outer_radius, height=length)

    # =====================================================
    # ARCHITECTURAL FRAMES
    # =====================================================
    
    def _create_door_frame(self, dims):
        """Create a door frame with header and side jambs"""
        width = dims.get('width', 900)  # Door opening width (mm)
        height = dims.get('height', 2100)  # Door opening height (mm)
        depth = dims.get('depth', 150)  # Frame depth (mm)
        thickness = dims.get('thickness', 50)  # Frame thickness (mm)
        
        try:
            # Create the outer frame box
            outer_width = width + 2 * thickness
            outer_height = height + thickness  # No bottom piece
            frame_box = trimesh.creation.box(extents=[outer_width, depth, outer_height])
            
            # Create the opening to subtract
            opening = trimesh.creation.box(extents=[width, depth * 1.1, height])
            opening = opening.apply_translation([0, 0, -thickness/2])
            
            return frame_box.difference(opening)
        except Exception:
            # Fallback: create L-shaped frame pieces
            # Left jamb
            left = trimesh.creation.box(extents=[thickness, depth, height])
            left = left.apply_translation([-(width/2 + thickness/2), 0, 0])
            # Right jamb  
            right = trimesh.creation.box(extents=[thickness, depth, height])
            right = right.apply_translation([(width/2 + thickness/2), 0, 0])
            # Header
            header = trimesh.creation.box(extents=[width + 2*thickness, depth, thickness])
            header = header.apply_translation([0, 0, height/2 + thickness/2])
            
            try:
                return left.union(right).union(header)
            except Exception:
                return trimesh.creation.box(extents=[outer_width, depth, outer_height])
    
    def _create_window_frame(self, dims):
        """Create a window frame with sill"""
        width = dims.get('width', 1200)  # Window opening width (mm)
        height = dims.get('height', 1000)  # Window opening height (mm)
        depth = dims.get('depth', 100)  # Frame depth (mm)
        thickness = dims.get('thickness', 50)  # Frame thickness (mm)
        sill_height = dims.get('sill_height', 20)  # Window sill height (mm)
        
        try:
            # Create the outer frame box
            outer_width = width + 2 * thickness
            outer_height = height + 2 * thickness
            frame_box = trimesh.creation.box(extents=[outer_width, depth, outer_height])
            
            # Create the opening to subtract
            opening = trimesh.creation.box(extents=[width, depth * 1.1, height])
            
            # Create window sill (extended bottom piece)
            sill = trimesh.creation.box(extents=[outer_width + 100, depth + 50, sill_height])
            sill = sill.apply_translation([0, 25, -(outer_height/2 + sill_height/2)])
            
            frame_with_opening = frame_box.difference(opening)
            return frame_with_opening.union(sill)
            
        except Exception:
            # Fallback: create frame pieces separately
            # Create 4 sides of the window frame
            left = trimesh.creation.box(extents=[thickness, depth, height + 2*thickness])
            left = left.apply_translation([-(width/2 + thickness/2), 0, 0])
            right = trimesh.creation.box(extents=[thickness, depth, height + 2*thickness])
            right = right.apply_translation([(width/2 + thickness/2), 0, 0])
            top = trimesh.creation.box(extents=[width, depth, thickness])
            top = top.apply_translation([0, 0, height/2 + thickness/2])
            bottom = trimesh.creation.box(extents=[width, depth, thickness])
            bottom = bottom.apply_translation([0, 0, -(height/2 + thickness/2)])
            
            try:
                return left.union(right).union(top).union(bottom)
            except Exception:
                return trimesh.creation.box(extents=[outer_width, depth, outer_height])
    
    def _create_gypsum_frame(self, dims):
        """Create a gypsum/drywall frame structure"""
        width = dims.get('width', 2400)  # Frame width (mm)
        height = dims.get('height', 2700)  # Frame height (mm)
        depth = dims.get('depth', 100)  # Stud depth (mm)
        stud_width = dims.get('stud_width', 50)  # Stud width (mm)
        spacing = dims.get('spacing', 400)  # Stud spacing (mm)
        
        try:
            # Create top and bottom plates
            top_plate = trimesh.creation.box(extents=[width, depth, stud_width])
            top_plate = top_plate.apply_translation([0, 0, height/2 - stud_width/2])
            bottom_plate = trimesh.creation.box(extents=[width, depth, stud_width])
            bottom_plate = bottom_plate.apply_translation([0, 0, -height/2 + stud_width/2])
            
            # Create vertical studs
            stud_height = height - 2 * stud_width
            num_studs = int(width / spacing) + 1
            studs = []
            
            for i in range(num_studs):
                x_pos = -width/2 + i * spacing
                if x_pos <= width/2:
                    stud = trimesh.creation.box(extents=[stud_width, depth, stud_height])
                    stud = stud.apply_translation([x_pos, 0, 0])
                    studs.append(stud)
            
            # Combine all pieces
            frame = top_plate.union(bottom_plate)
            for stud in studs:
                frame = frame.union(stud)
            
            return frame
            
        except Exception:
            # Fallback: simple rectangular frame
            return trimesh.creation.box(extents=[width, depth, height])

    # =====================================================
    # FURNITURE FRAMES
    # =====================================================
    
    def _create_bed_frame(self, dims):
        """Create a bed frame structure"""
        length = dims.get('length', 2000)  # Bed length (mm)
        width = dims.get('width', 1500)   # Bed width (mm)
        height = dims.get('height', 400)  # Frame height (mm)
        rail_width = dims.get('rail_width', 80)  # Rail thickness (mm)
        rail_height = dims.get('rail_height', 200)  # Rail height (mm)
        
        try:
            # Create head rail
            head_rail = trimesh.creation.box(extents=[width, rail_width, rail_height])
            head_rail = head_rail.apply_translation([0, length/2 - rail_width/2, rail_height/2 - height/2])
            
            # Create foot rail
            foot_rail = trimesh.creation.box(extents=[width, rail_width, rail_height * 0.6])
            foot_rail = foot_rail.apply_translation([0, -length/2 + rail_width/2, rail_height*0.3 - height/2])
            
            # Create side rails
            left_rail = trimesh.creation.box(extents=[rail_width, length - 2*rail_width, rail_width])
            left_rail = left_rail.apply_translation([-width/2 + rail_width/2, 0, -height/2 + rail_width/2])
            
            right_rail = trimesh.creation.box(extents=[rail_width, length - 2*rail_width, rail_width])
            right_rail = right_rail.apply_translation([width/2 - rail_width/2, 0, -height/2 + rail_width/2])
            
            # Create support slats (simplified as a platform)
            platform = trimesh.creation.box(extents=[width - 2*rail_width, length - 2*rail_width, 20])
            platform = platform.apply_translation([0, 0, -height/2 + 20])
            
            return head_rail.union(foot_rail).union(left_rail).union(right_rail).union(platform)
            
        except Exception:
            # Fallback: simple platform
            return trimesh.creation.box(extents=[width, length, height])
    
    def _create_table_frame(self, dims):
        """Create a table frame structure"""
        length = dims.get('length', 1200)  # Table length (mm)
        width = dims.get('width', 800)    # Table width (mm)
        height = dims.get('height', 750)  # Table height (mm)
        top_thickness = dims.get('top_thickness', 30)  # Top thickness (mm)
        leg_size = dims.get('leg_size', 50)  # Leg cross-section (mm)
        
        try:
            # Create table top
            table_top = trimesh.creation.box(extents=[length, width, top_thickness])
            table_top = table_top.apply_translation([0, 0, height/2 - top_thickness/2])
            
            # Create legs
            leg_height = height - top_thickness
            leg_positions = [
                [-length/2 + leg_size, -width/2 + leg_size, -top_thickness/2],
                [length/2 - leg_size, -width/2 + leg_size, -top_thickness/2],
                [-length/2 + leg_size, width/2 - leg_size, -top_thickness/2],
                [length/2 - leg_size, width/2 - leg_size, -top_thickness/2]
            ]
            
            frame = table_top
            for pos in leg_positions:
                leg = trimesh.creation.box(extents=[leg_size, leg_size, leg_height])
                leg = leg.apply_translation(pos)
                frame = frame.union(leg)
            
            return frame
            
        except Exception:
            # Fallback: solid block
            return trimesh.creation.box(extents=[length, width, height])
    
    def _create_chair_frame(self, dims):
        """Create a chair frame structure"""
        width = dims.get('width', 450)     # Seat width (mm)
        depth = dims.get('depth', 400)     # Seat depth (mm)
        seat_height = dims.get('seat_height', 450)  # Seat height (mm)
        back_height = dims.get('back_height', 350)  # Back height above seat (mm)
        frame_size = dims.get('frame_size', 30)     # Frame member size (mm)
        
        try:
            # Create seat frame
            seat = trimesh.creation.box(extents=[width, depth, frame_size])
            seat = seat.apply_translation([0, 0, seat_height - frame_size/2])
            
            # Create backrest
            back = trimesh.creation.box(extents=[width, frame_size, back_height])
            back = back.apply_translation([0, depth/2 - frame_size/2, seat_height + back_height/2])
            
            # Create legs
            leg_positions = [
                [-width/2 + frame_size/2, -depth/2 + frame_size/2],
                [width/2 - frame_size/2, -depth/2 + frame_size/2],
                [-width/2 + frame_size/2, depth/2 - frame_size/2],
                [width/2 - frame_size/2, depth/2 - frame_size/2]
            ]
            
            frame = seat.union(back)
            for x, y in leg_positions:
                leg = trimesh.creation.box(extents=[frame_size, frame_size, seat_height])
                leg = leg.apply_translation([x, y, seat_height/2 - frame_size/2])
                frame = frame.union(leg)
            
            return frame
            
        except Exception:
            # Fallback: simple chair block
            total_height = seat_height + back_height
            return trimesh.creation.box(extents=[width, depth, total_height])
    
    def _create_shelf_frame(self, dims):
        """Create a shelf frame structure"""
        width = dims.get('width', 800)     # Shelf width (mm)
        depth = dims.get('depth', 300)     # Shelf depth (mm)
        height = dims.get('height', 1800)  # Total height (mm)
        shelf_thickness = dims.get('shelf_thickness', 20)  # Shelf thickness (mm)
        num_shelves = dims.get('num_shelves', 4)  # Number of shelves
        
        try:
            # Create vertical sides
            left_side = trimesh.creation.box(extents=[shelf_thickness, depth, height])
            left_side = left_side.apply_translation([-width/2 + shelf_thickness/2, 0, 0])
            
            right_side = trimesh.creation.box(extents=[shelf_thickness, depth, height])
            right_side = right_side.apply_translation([width/2 - shelf_thickness/2, 0, 0])
            
            # Create shelves
            shelf_spacing = (height - shelf_thickness) / (num_shelves - 1)
            frame = left_side.union(right_side)
            
            for i in range(num_shelves):
                z_pos = -height/2 + shelf_thickness/2 + i * shelf_spacing
                shelf = trimesh.creation.box(extents=[width - 2*shelf_thickness, depth, shelf_thickness])
                shelf = shelf.apply_translation([0, 0, z_pos])
                frame = frame.union(shelf)
            
            return frame
            
        except Exception:
            # Fallback: solid block
            return trimesh.creation.box(extents=[width, depth, height])
    
    def _create_cabinet_frame(self, dims):
        """Create a cabinet frame structure"""
        width = dims.get('width', 600)     # Cabinet width (mm)
        depth = dims.get('depth', 350)     # Cabinet depth (mm)
        height = dims.get('height', 720)   # Cabinet height (mm)
        panel_thickness = dims.get('panel_thickness', 18)  # Panel thickness (mm)
        
        try:
            # Create cabinet box
            # Left side
            left = trimesh.creation.box(extents=[panel_thickness, depth, height])
            left = left.apply_translation([-width/2 + panel_thickness/2, 0, 0])
            
            # Right side
            right = trimesh.creation.box(extents=[panel_thickness, depth, height])
            right = right.apply_translation([width/2 - panel_thickness/2, 0, 0])
            
            # Top
            top = trimesh.creation.box(extents=[width, depth, panel_thickness])
            top = top.apply_translation([0, 0, height/2 - panel_thickness/2])
            
            # Bottom
            bottom = trimesh.creation.box(extents=[width, depth, panel_thickness])
            bottom = bottom.apply_translation([0, 0, -height/2 + panel_thickness/2])
            
            # Back panel
            back_panel = trimesh.creation.box(extents=[width - 2*panel_thickness, panel_thickness, height - 2*panel_thickness])
            back_panel = back_panel.apply_translation([0, depth/2 - panel_thickness/2, 0])
            
            return left.union(right).union(top).union(bottom).union(back_panel)
            
        except Exception:
            # Fallback: solid block
            return trimesh.creation.box(extents=[width, depth, height])

    # =====================================================
    # PARAMETRIC BUILDERS (CadQuery + Trimesh Fallbacks)
    # =====================================================
    
    def _create_parametric_washer(self, dims):
        """Enhanced washer with CadQuery precision"""
        outer_radius = dims.get('outer_radius', dims.get('radius', 20))
        inner_radius = dims.get('inner_radius', outer_radius * 0.4)
        thickness = dims.get('thickness', 3)
        
        if CADQUERY_AVAILABLE:
            try:
                outer = cq.Workplane("XY").circle(outer_radius).extrude(thickness)
                inner = cq.Workplane("XY").circle(inner_radius).extrude(thickness + 0.01)
                washer = outer.cut(inner)
                return cq_to_trimesh(washer)
            except Exception:
                pass
        
        # Fallback to existing method
        return self._create_washer(dims)
    
    def _create_parametric_nut(self, dims):
        """Enhanced nut with CadQuery precision"""
        radius = dims.get('radius', 10)
        thickness = dims.get('thickness', dims.get('height', 6))
        hole_radius = dims.get('hole_radius', radius * 0.4)
        
        if CADQUERY_AVAILABLE:
            try:
                nut = cq.Workplane("XY").polygon(6, radius * 2).extrude(thickness)
                nut = nut.faces(">Z").workplane().hole(hole_radius * 2)
                return cq_to_trimesh(nut)
            except Exception:
                pass
        
        # Fallback to existing method
        return self._create_nut(dims)
    
    def _create_parametric_bracket(self, dims):
        """Enhanced bracket with CadQuery precision"""
        leg1 = dims.get('length', 100)
        leg2 = dims.get('height', 80)
        thickness = dims.get('thickness', 6)
        hole_diam = dims.get('hole_radius', 4) * 2
        hole_offset = dims.get('depth', 20)
        
        if CADQUERY_AVAILABLE:
            try:
                legA = cq.Workplane("XY").box(leg1, thickness, thickness).translate((leg1/2 - leg1, 0, 0))
                legB = cq.Workplane("XY").box(thickness, thickness, leg2).translate((0, 0, leg2/2))
                combined = legA.union(legB)
                if hole_diam > 0:
                    combined = combined.faces(">Z").workplane().pushPoints([(hole_offset - leg1/2, 0), (leg1 - hole_offset - leg1/2, 0)]).hole(hole_diam)
                return cq_to_trimesh(combined)
            except Exception:
                pass
        
        # Fallback to existing method
        return self._create_bracket(dims)
    
    def _create_parametric_door(self, dims):
        """Enhanced door with CadQuery precision"""
        width = dims.get('width', 900)
        height = dims.get('height', 2100)
        thickness = dims.get('thickness', 40)
        panel_count = dims.get('panel_count', 2)
        frame_width = dims.get('frame_width', 60)
        
        if CADQUERY_AVAILABLE:
            try:
                door = cq.Workplane("XY").box(width, thickness, height)
                # Add panel insets
                if panel_count > 0:
                    panel_h = (height - 2*frame_width - (panel_count-1)*frame_width) / panel_count
                    z0 = -height/2 + frame_width + panel_h/2
                    for i in range(int(panel_count)):
                        door = door.faces(">Y").workplane().center(0, z0 - (-height/2)).rect(width - 2*frame_width, panel_h - frame_width/2).cutBlind(-frame_width/4)
                        z0 += panel_h + frame_width
                return cq_to_trimesh(door)
            except Exception:
                pass
        
        # Fallback to existing method
        return self._create_door_frame(dims)
    
    def _create_parametric_window(self, dims):
        """Enhanced window with CadQuery precision"""
        width = dims.get('width', 1200)
        height = dims.get('height', 1200)
        frame_thickness = dims.get('thickness', dims.get('depth', 60))
        glass_thickness = dims.get('glass_thickness', 6)
        
        if CADQUERY_AVAILABLE:
            try:
                # Create frame
                outer = cq.Workplane("XY").box(width, frame_thickness, height)
                inner = cq.Workplane("XY").box(width - 2*frame_thickness, frame_thickness + 2, height - 2*frame_thickness)
                frame = outer.cut(inner)
                
                # Add glass
                glass = cq.Workplane("XY").box(width - 2*frame_thickness - 2, glass_thickness, height - 2*frame_thickness - 2)
                return cq_to_trimesh(frame.union(glass))
            except Exception:
                pass
        
        # Fallback to existing method
        return self._create_window_frame(dims)
    
    def _create_water_tank(self, dims):
        """Create a cylindrical water tank"""
        diameter = dims.get('diameter', 1000)
        height = dims.get('height', 1200)
        wall_thickness = dims.get('wall_thickness', dims.get('thickness', 8))
        
        outer_radius = diameter / 2
        inner_radius = outer_radius - wall_thickness
        
        if CADQUERY_AVAILABLE:
            try:
                outer = cq.Workplane("XY").circle(outer_radius).extrude(height)
                inner = cq.Workplane("XY").circle(inner_radius).extrude(height + 1)
                tank = outer.cut(inner)
                # Add lid
                lid = cq.Workplane("XY").circle(outer_radius + 10).extrude(5).translate((0, 0, height/2 + 2.5))
                return cq_to_trimesh(tank.union(lid))
            except Exception:
                pass
        
        # Trimesh fallback
        try:
            outer = trimesh.creation.cylinder(radius=outer_radius, height=height, sections=128)
            inner = trimesh.creation.cylinder(radius=inner_radius, height=height*1.01, sections=128)
            return safe_difference(outer, inner)
        except Exception:
            return trimesh.creation.cylinder(radius=outer_radius, height=height)

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
# ORTHOGRAPHIC VIEWS GENERATOR (6 views + flexible grid)
# =====================================================

def generate_orthographic_views(mesh, layout="2x3"):
    """
    Generate 6 orthographic views (Front, Back, Top, Bottom, Left, Right)
    and return a combined grid image.

    Args:
        mesh: Trimesh.Trimesh object with `.vertices`
        layout (str): Grid layout e.g. "2x3", "3x2", "1x6", "6x1"
    
    Returns:
        PIL.Image: Combined orthographic views
    
    Raises:
        RuntimeError: If mesh processing fails
    """
    # Validate mesh before processing
    try:
        validate_mesh(mesh)
    except Exception as validation_error:
        raise RuntimeError(f"Invalid mesh for orthographic views: {validation_error}")
    
    try:
        projections = [
            ("Front (XY)",  [0, 1]),
            ("Back (XY)",   [0, 1]),
            ("Top (XZ)",    [0, 2]),
            ("Bottom (XZ)", [0, 2]),
            ("Left (YZ)",   [1, 2]),
            ("Right (YZ)",  [1, 2]),
        ]
        
        views = []
        titles = []

        # Generate individual view images
        for title, axes_idx in projections:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            vertices_2d = mesh.vertices[:, axes_idx]

            if len(vertices_2d) > 3:
                try:
                    hull = ConvexHull(vertices_2d)
                    hull_points = vertices_2d[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.plot(hull_points[:, 0], hull_points[:, 1], 'b-', linewidth=2.5)
                    ax.fill(hull_points[:, 0], hull_points[:, 1], 'lightblue', alpha=0.3)
                except Exception:
                    # Fallback: scatter plot if ConvexHull fails
                    ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='blue', s=0.5, alpha=0.6)
            else:
                ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='blue', s=2)

            # Add dimension annotations
            if len(vertices_2d) > 0:
                x_range = np.ptp(vertices_2d[:, 0])
                y_range = np.ptp(vertices_2d[:, 1])
                ax.text(0.02, 0.98, f'{x_range:.1f}mm', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                ax.text(0.98, 0.02, f'{y_range:.1f}mm', transform=ax.transAxes, 
                       horizontalalignment='right', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf).convert('RGBA')
            plt.close(fig)

            views.append(img)
            titles.append(title)

        # Parse layout
        try:
            rows, cols = map(int, layout.lower().split("x"))
            if rows * cols < 6:
                raise ValueError("Layout grid too small for 6 views")
        except Exception:
            rows, cols = 2, 3

        # Create combined grid
        fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        fig.suptitle('Engineering Orthographic Views', fontsize=16, fontweight='bold')
        axs = np.array(axs).reshape(-1)

        for ax, view_img, title in zip(axs[:len(views)], views, titles):
            ax.imshow(view_img)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        # Hide unused subplots
        for ax in axs[len(views):]:
            ax.axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        combined_img = Image.open(buf).convert('RGBA')
        plt.close(fig)

        return combined_img
    
    except Exception as e:
        raise RuntimeError(f"Orthographic Views Error: {str(e)}")

def render_error_to_image(error_message, width=800, height=400, title="Error"):
    """Create an error image for display purposes"""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.text(0.5, 0.5, str(error_message), ha='center', va='center', 
            fontsize=12, color='red', wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, color='red')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert('RGBA')
    plt.close(fig)
    
    return img

def validate_mesh(mesh):
    """Validate that the mesh object is valid for processing"""
    if mesh is None:
        raise ValueError("Mesh is None")
    
    if not hasattr(mesh, 'vertices'):
        raise ValueError("Mesh missing vertices attribute")
    
    if not hasattr(mesh, 'faces'):
        raise ValueError("Mesh missing faces attribute")
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    if vertices is None or len(vertices) == 0:
        raise ValueError("Mesh has no vertices")
    
    if faces is None or len(faces) == 0:
        raise ValueError("Mesh has no faces")
    
    if vertices.shape[1] != 3:
        raise ValueError(f"Vertices must be 3D, got shape {vertices.shape}")
    
    if faces.shape[1] != 3:
        raise ValueError(f"Faces must be triangular, got shape {faces.shape}")
    
    # Check for invalid face indices
    max_vertex_index = len(vertices) - 1
    if np.any(faces > max_vertex_index):
        raise ValueError("Faces reference non-existent vertices")
    
    if np.any(faces < 0):
        raise ValueError("Faces contain negative indices")
    
    return True

def render_mesh_preview(mesh, title="Mesh Preview", width=600, height=400):
    """Render a basic mesh preview image"""
    try:
        validate_mesh(mesh)
        
        fig = plt.figure(figsize=(width/100, height/100))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           triangles=mesh.faces, alpha=0.8, shade=True)
        else:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c='blue', s=1, alpha=0.6)
        
        ax.set_title(title)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert('RGBA')
        plt.close(fig)
        
        return img
        
    except Exception as e:
        return render_error_to_image(f"Mesh preview error: {str(e)}", width, height, "Preview Error")

# =====================================================
# MAIN APPLICATION FUNCTIONS
# =====================================================

# Initialize CAD generator
cad_generator = TextToCADGenerator()

def process_text_to_cad(prompt, precision_choice="High (parametric)", export_format="stl", grid_layout="2x3", use_llm=False):
    """Enhanced CAD processing with precision modes and optional LLM parsing."""
    try:
        params = None

        # If LLM is enabled and available, attempt to use it to parse prompt
        if use_llm and LLM_AVAILABLE and LLMTextToCAD is not None:
            try:
                llm = LLMTextToCAD()
                if llm.available():
                    llm_result = llm.compile(prompt)
                    if llm_result.get("success"):
                        schema = llm_result["schema"]
                        # Map schema to our internal params
                        dims = schema.get("dimensions", {})
                        # Ensure numeric and filter
                        dims = filter_numeric(dims)
                        color = schema.get("color") or 'lightblue'
                        shape = (schema.get("shape") or 'cube').replace(' ', '_')
                        params = {
                            'shape': shape if shape in cad_generator.shapes_library else 'cube',
                            'dimensions': dims,
                            'color': color,
                            'precision': 'high',
                            'prompt': prompt
                        }
            except Exception as e:
                print(f"LLM parse failed, falling back: {e}")

        # Fallback to legacy parsing if no LLM params
        if params is None:
            params = cad_generator.parse_prompt(prompt)

        # Override precision based on UI choice
        params['precision'] = 'high' if precision_choice.lower().startswith('h') else 'fast'

        # Use parametric builders for high precision when available
        if params['precision'] == 'high' and params['shape'] in ['washer', 'nut', 'bracket', 'door', 'window']:
            # Try parametric version first
            parametric_shape = f"parametric_{params['shape']}"
            if parametric_shape in cad_generator.shapes_library:
                original_shape = params['shape']
                params['shape'] = parametric_shape
                try:
                    mesh_3d = cad_generator.generate_3d_model(params)
                except Exception:
                    # Fallback to original shape
                    params['shape'] = original_shape
                    mesh_3d = cad_generator.generate_3d_model(params)
            else:
                mesh_3d = cad_generator.generate_3d_model(params)
        else:
            mesh_3d = cad_generator.generate_3d_model(params)

        # Validate the generated mesh
        try:
            validate_mesh(mesh_3d)
        except Exception as validation_error:
            raise RuntimeError(f"Generated mesh is invalid: {validation_error}")

        # Generate 3D visualization
        fig_3d = cad_generator.generate_3d_visualization(mesh_3d, params['color'])

        # Generate enhanced orthographic views
        try:
            ortho_views = generate_orthographic_views(mesh_3d, layout=grid_layout)
        except Exception as ortho_error:
            print(f"Orthographic views failed: {ortho_error}")
            ortho_views = render_error_to_image(
                f"Failed to generate orthographic views: {str(ortho_error)}",
                title="Orthographic Views Error"
            )

        # Enhanced summary
        dims = params['dimensions']
        dim_summary = []
        for key, value in dims.items():
            if value is not None and key in ['length', 'width', 'height', 'radius', 'diameter', 'thickness']:
                dim_summary.append(f"{key.title()}: {value}mm")

        backend_info = "✅ CadQuery (parametric)" if CADQUERY_AVAILABLE and params['precision'] == 'high' else "⚡ Trimesh (fast)"
        boolean_info = f"Boolean backend: {BOOL_BACKEND}" if BOOL_BACKEND else "No boolean backend"

        summary = f"""
**🔧 Generated CAD Model Summary:**
- **Shape:** {params['shape'].replace('_', ' ').replace('parametric ', '').title()}
- **Dimensions:** {', '.join(dim_summary) if dim_summary else 'Default dimensions'}
- **Color:** {params['color'].title()}
- **Precision Mode:** {params['precision'].title()} precision
- **CAD Backend:** {backend_info}
- **{boolean_info}**
- **Original Prompt:** "{params['prompt']}"

✅ The model has been successfully generated with 6-view orthographic projections.
"""

        # Optional export
        export_path = None
        if export_format and mesh_3d:
            try:
                tmpfile = tempfile.NamedTemporaryFile(suffix=f'.{export_format}', delete=False)
                tmpname = tmpfile.name
                tmpfile.close()
                export_mesh(mesh_3d, tmpname)
                export_path = tmpname
            except Exception as e:
                export_path = None
                print(f"Export error: {e}")

        return fig_3d, ortho_views, summary, export_path

    except Exception as e:
        error_msg = f"Error generating CAD model: {str(e)}"
        print(f"CAD Generation Error: {error_msg}")  # Log the error

        # Create placeholder 3D figure
        placeholder_fig = go.Figure()
        placeholder_fig.add_annotation(
            text=error_msg,
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        placeholder_fig.update_layout(
            title="CAD Generation Failed",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        # Create standardized error image
        error_img = render_error_to_image(
            error_msg,
            width=800,
            height=400,
            title="CAD Generation Error"
        )

        return placeholder_fig, error_img, f"❌ **Error:** {error_msg}", None

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
    
    with gr.Blocks(title="KelmoidAI_Genesis LLM prototype", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🔧 KelmoidAI_Genesis LLM prototype
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
                            placeholder="e.g., 'Create a door width=900 height=2100 thickness=40 panels=2' or 'washer radius=20 thickness=3'",
                            lines=3
                        )
                        
                        with gr.Row():
                            precision_choice = gr.Radio(
                                ["High (parametric)", "Fast (approximate)"], 
                                value="High (parametric)", 
                                label="Precision Mode",
                                info="High uses CadQuery for accuracy, Fast uses Trimesh for speed"
                            )
                            export_format = gr.Dropdown(
                                ["stl", "obj", "ply", "glb"], 
                                value="stl", 
                                label="Export Format"
                            )
                            grid_layout = gr.Dropdown(
                                ["2x3", "3x2", "1x6", "6x1"], 
                                value="2x3", 
                                label="Orthographic Layout"
                            )
                        use_llm = gr.Checkbox(label="Use LLM (Text-to-CAD compiler)", value=False, info="Requires OPENAI_API_KEY or HF_API_TOKEN in environment. Falls back automatically if unavailable.")
                        
                        cad_generate_btn = gr.Button("🚀 Generate CAD Model", variant="primary", size="lg")
                        download_file = gr.File(label="Download CAD File", visible=False)
                        
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
                            gr.Button("Parametric Door", size="sm").click(
                                lambda: "Create parametric door width=900 height=2100 thickness=40 panels=2", 
                                outputs=cad_prompt
                            )
                            gr.Button("Water Tank", size="sm").click(
                                lambda: "Design water tank diameter=1000 height=1200 wall_thickness=8", 
                                outputs=cad_prompt
                            )
                            gr.Button("Precision Washer", size="sm").click(
                                lambda: "Create parametric washer outer_radius=25 inner_radius=10 thickness=3", 
                                outputs=cad_prompt
                            )
                            gr.Button("Hex Nut M12", size="sm").click(
                                lambda: "Make parametric nut radius=12 thickness=10 hole_radius=6", 
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
                    inputs=[cad_prompt, precision_choice, export_format, grid_layout, use_llm],
                    outputs=[cad_3d_output, cad_ortho_output, cad_summary_output, download_file]
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
        
        **🔧 Enhanced Text-to-CAD Generator:**
        - **Basic Shapes**: cube, sphere, cylinder, cone, pyramid, torus, gear, plate, rod
        - **Mechanical Parts**: bracket, washer, screw, bolt, nut, bearing, flange, pipe
        - **Architectural Frames**: door frame, window frame, gypsum frame, drywall frame, water tank
        - **Furniture Frames**: bed frame, table frame, chair frame, shelf frame, cabinet frame
        - **Parametric Models**: parametric_door, parametric_window, parametric_washer, parametric_nut, parametric_bracket
        - **Precision Modes**: High (CadQuery parametric) vs Fast (Trimesh approximate)
        - **Key=Value Syntax**: Use `width=900 height=2100 thickness=40` for precise control
        - **Export Formats**: STL, OBJ, PLY, GLB for 3D printing and CAD software
        - **Dimension Keywords**: length, width, height, radius, diameter, thickness, depth, spacing, panels, frames
        - **Colors**: red, blue, green, yellow, orange, purple, pink, brown, black, white, gray
        
        **🎯 Pro Tips:**
        - Use `parametric_` prefix for high-precision mechanical parts
        - Include `panels=4` for doors, `wall_thickness=8` for tanks
        - Try `NxMxP` patterns like `100x50x25` for quick dimensions
        
        **2D Plate Designer:**
        - Describe plates with dimensions like "100mm x 50mm"
        - Add features: "20mm diameter hole", "30mm long and 10mm wide slot", "oval hole 40mm long and 20mm wide"
        - Generates technical drawings and CNC G-code
        
        **CFD Simulator:**
        - Simulates fluid flow through channels with optional obstacles
        - Adjust grid resolution for accuracy vs. speed
        - Lower viscosity = higher Reynolds number = more turbulent flow
        
        **🚀 Enhanced Features:**
        - **CadQuery Integration**: Precise parametric CAD when available
        - **Boolean Backend**: Advanced geometry operations with manifold3d
        - **6-View Orthographics**: Professional engineering drawings
        - **Export Support**: Direct download of STL/OBJ/PLY/GLB files
        - **Key=Value Parsing**: `width=900 thickness=40` syntax support
        
        **Note:** This application supports both CPU-based fast prototyping and precision parametric CAD modeling.
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
