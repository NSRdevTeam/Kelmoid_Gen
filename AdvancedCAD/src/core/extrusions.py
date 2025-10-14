"""
2D Primitives and Extrusion Operations
Advanced 2D shape generation and extrusion with enhanced features
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .primitives import MeshData


@dataclass
class Shape2D:
    """2D shape representation"""
    points: np.ndarray      # Nx2 array of 2D points
    closed: bool = True     # Whether the shape is closed
    holes: List[np.ndarray] = None  # List of hole contours


class Shape2DGenerator:
    """Base class for 2D shape generators"""
    
    def __init__(self):
        self.default_resolution = 32
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters"""
        return params
    
    def generate(self, params: Dict[str, Any]) -> Shape2D:
        """Generate 2D shape"""
        raise NotImplementedError


class CircleGenerator(Shape2DGenerator):
    """Generate 2D circle shapes"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate circle parameters"""
        validated = {
            'radius': 1.0,
            'resolution': self.default_resolution,
            'center': True
        }
        
        if 'r' in params:
            validated['radius'] = float(params['r'])
        elif 'radius' in params:
            validated['radius'] = float(params['radius'])
        
        if 'd' in params:
            validated['radius'] = float(params['d']) / 2.0
        elif 'diameter' in params:
            validated['radius'] = float(params['diameter']) / 2.0
        
        if 'fn' in params:
            validated['resolution'] = int(params['fn'])
        elif 'resolution' in params:
            validated['resolution'] = int(params['resolution'])
        
        # Ensure minimum resolution
        validated['resolution'] = max(6, validated['resolution'])
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> Shape2D:
        """Generate circle shape"""
        params = self.validate_params(params)
        radius = params['radius']
        resolution = params['resolution']
        
        # Generate circle points
        angles = np.linspace(0, 2 * math.pi, resolution, endpoint=False)
        points = np.array([
            [radius * math.cos(angle), radius * math.sin(angle)]
            for angle in angles
        ], dtype=np.float32)
        
        return Shape2D(points=points, closed=True)


class SquareGenerator(Shape2DGenerator):
    """Generate 2D square/rectangle shapes"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate square parameters"""
        validated = {
            'size': [1.0, 1.0],
            'center': True
        }
        
        if 'size' in params:
            size = params['size']
            if isinstance(size, (int, float)):
                validated['size'] = [float(size), float(size)]
            elif isinstance(size, (list, tuple)) and len(size) >= 2:
                validated['size'] = [float(size[0]), float(size[1])]
        
        if 'center' in params:
            validated['center'] = bool(params['center'])
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> Shape2D:
        """Generate square/rectangle shape"""
        params = self.validate_params(params)
        size = params['size']
        center = params['center']
        
        width, height = size
        
        if center:
            # Centered square
            points = np.array([
                [-width/2, -height/2],
                [ width/2, -height/2],
                [ width/2,  height/2],
                [-width/2,  height/2]
            ], dtype=np.float32)
        else:
            # Square with corner at origin
            points = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)
        
        return Shape2D(points=points, closed=True)


class PolygonGenerator(Shape2DGenerator):
    """Generate polygon shapes from points"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate polygon parameters"""
        validated = {
            'points': [],
            'paths': None,
            'convexity': 1
        }
        
        if 'points' in params:
            validated['points'] = params['points']
        
        if 'paths' in params:
            validated['paths'] = params['paths']
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> Shape2D:
        """Generate polygon shape"""
        params = self.validate_params(params)
        points_list = params['points']
        
        if not points_list:
            # Default to unit square if no points provided
            points_list = [[0, 0], [1, 0], [1, 1], [0, 1]]
        
        points = np.array(points_list, dtype=np.float32)
        
        return Shape2D(points=points, closed=True)


class RegularPolygonGenerator(Shape2DGenerator):
    """Generate regular polygons"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regular polygon parameters"""
        validated = {
            'sides': 6,
            'radius': 1.0,
            'inscribed': True  # True for inscribed circle, False for circumscribed
        }
        
        if 'sides' in params:
            validated['sides'] = max(3, int(params['sides']))
        
        if 'r' in params:
            validated['radius'] = float(params['r'])
        elif 'radius' in params:
            validated['radius'] = float(params['radius'])
        
        if 'inscribed' in params:
            validated['inscribed'] = bool(params['inscribed'])
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> Shape2D:
        """Generate regular polygon"""
        params = self.validate_params(params)
        sides = params['sides']
        radius = params['radius']
        inscribed = params['inscribed']
        
        if inscribed:
            # Adjust radius for inscribed circle
            actual_radius = radius / math.cos(math.pi / sides)
        else:
            actual_radius = radius
        
        # Generate polygon points
        angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
        points = np.array([
            [actual_radius * math.cos(angle), actual_radius * math.sin(angle)]
            for angle in angles
        ], dtype=np.float32)
        
        return Shape2D(points=points, closed=True)


class ExtrusionEngine:
    """Engine for performing extrusion operations"""
    
    def __init__(self):
        pass
    
    def linear_extrude(self, shape: Shape2D, **params) -> MeshData:
        """Perform linear extrusion of 2D shape"""
        validated_params = self._validate_linear_params(params)
        
        height = validated_params['height']
        center = validated_params['center']
        twist = validated_params['twist']
        scale_factor = validated_params['scale']
        slices = validated_params['slices']
        
        return self._perform_linear_extrude(shape, height, center, twist, scale_factor, slices)
    
    def rotate_extrude(self, shape: Shape2D, **params) -> MeshData:
        """Perform rotational extrusion of 2D shape"""
        validated_params = self._validate_rotate_params(params)
        
        angle = validated_params['angle']
        resolution = validated_params['resolution']
        
        return self._perform_rotate_extrude(shape, angle, resolution)
    
    def path_extrude(self, shape: Shape2D, path: List[Tuple[float, float, float]], **params) -> MeshData:
        """Extrude shape along a 3D path"""
        # Advanced feature - extrude along arbitrary 3D path
        return self._perform_path_extrude(shape, path, params)
    
    def _validate_linear_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate linear extrusion parameters"""
        validated = {
            'height': 1.0,
            'center': False,
            'twist': 0.0,
            'scale': 1.0,
            'slices': 1
        }
        
        if 'h' in params:
            validated['height'] = float(params['h'])
        elif 'height' in params:
            validated['height'] = float(params['height'])
        
        if 'center' in params:
            validated['center'] = bool(params['center'])
        
        if 'twist' in params:
            validated['twist'] = float(params['twist'])
        
        if 'scale' in params:
            scale = params['scale']
            if isinstance(scale, (int, float)):
                validated['scale'] = [float(scale), float(scale)]
            elif isinstance(scale, (list, tuple)) and len(scale) >= 2:
                validated['scale'] = [float(scale[0]), float(scale[1])]
            else:
                validated['scale'] = [1.0, 1.0]
        
        if 'slices' in params:
            validated['slices'] = max(1, int(params['slices']))
        
        # If twist is non-zero, increase slices for smooth twisting
        if validated['twist'] != 0.0 and validated['slices'] == 1:
            validated['slices'] = max(10, int(abs(validated['twist']) / 10))
        
        return validated
    
    def _validate_rotate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rotational extrusion parameters"""
        validated = {
            'angle': 360.0,
            'resolution': 32
        }
        
        if 'angle' in params:
            validated['angle'] = float(params['angle'])
        
        if 'fn' in params:
            validated['resolution'] = int(params['fn'])
        elif 'resolution' in params:
            validated['resolution'] = int(params['resolution'])
        
        validated['resolution'] = max(6, validated['resolution'])
        
        return validated
    
    def _perform_linear_extrude(self, shape: Shape2D, height: float, center: bool, 
                              twist: float, scale: List[float], slices: int) -> MeshData:
        """Perform the actual linear extrusion"""
        vertices = []
        faces = []
        
        points = shape.points
        num_points = len(points)
        
        # Calculate z positions for slices
        if center:
            z_positions = np.linspace(-height/2, height/2, slices + 1)
        else:
            z_positions = np.linspace(0, height, slices + 1)
        
        # Generate vertices for each slice
        for i, z in enumerate(z_positions):
            # Calculate twist angle and scale for this slice
            t = i / slices if slices > 0 else 0
            slice_twist = twist * t * math.pi / 180  # Convert to radians
            slice_scale_x = 1.0 + (scale[0] - 1.0) * t
            slice_scale_y = 1.0 + (scale[1] - 1.0) * t
            
            # Transform points for this slice
            for point in points:
                x, y = point
                
                # Apply scaling
                x *= slice_scale_x
                y *= slice_scale_y
                
                # Apply twist
                if slice_twist != 0:
                    cos_twist = math.cos(slice_twist)
                    sin_twist = math.sin(slice_twist)
                    x_new = x * cos_twist - y * sin_twist
                    y_new = x * sin_twist + y * cos_twist
                    x, y = x_new, y_new
                
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate faces
        for slice_idx in range(slices):
            for point_idx in range(num_points):
                # Current slice vertices
                v0 = slice_idx * num_points + point_idx
                v1 = slice_idx * num_points + ((point_idx + 1) % num_points)
                
                # Next slice vertices
                v2 = (slice_idx + 1) * num_points + point_idx
                v3 = (slice_idx + 1) * num_points + ((point_idx + 1) % num_points)
                
                # Create two triangles for the quad
                faces.extend([[v0, v1, v2], [v1, v3, v2]])
        
        # Add bottom cap (if not centered or first slice)
        if not center or slices == 0:
            center_idx = len(vertices)
            vertices = np.vstack([vertices, [0, 0, z_positions[0]]])
            
            for i in range(num_points):
                v0 = i
                v1 = (i + 1) % num_points
                faces.append([center_idx, v1, v0])  # Reverse winding for bottom
        
        # Add top cap
        top_center_idx = len(vertices)
        vertices = np.vstack([vertices, [0, 0, z_positions[-1]]])
        
        last_slice_offset = slices * num_points
        for i in range(num_points):
            v0 = last_slice_offset + i
            v1 = last_slice_offset + ((i + 1) % num_points)
            faces.append([top_center_idx, v0, v1])  # Normal winding for top
        
        faces = np.array(faces, dtype=np.int32)
        
        return MeshData(vertices=vertices, faces=faces)
    
    def _perform_rotate_extrude(self, shape: Shape2D, angle: float, resolution: int) -> MeshData:
        """Perform the actual rotational extrusion"""
        vertices = []
        faces = []
        
        points = shape.points
        num_points = len(points)
        
        # Number of rotational steps
        if abs(angle) >= 360.0:
            steps = resolution
            closed_revolution = True
        else:
            steps = max(2, int(resolution * abs(angle) / 360.0))
            closed_revolution = False
        
        # Generate vertices
        for step in range(steps + (0 if closed_revolution else 1)):
            rotation_angle = (angle * step / steps) * math.pi / 180
            cos_angle = math.cos(rotation_angle)
            sin_angle = math.sin(rotation_angle)
            
            for point in points:
                x, y = point
                # Rotate around Z-axis
                new_x = x * cos_angle
                new_y = x * sin_angle
                new_z = y
                vertices.append([new_x, new_y, new_z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate faces
        actual_steps = steps if closed_revolution else steps
        for step in range(actual_steps):
            next_step = (step + 1) % (steps + (0 if closed_revolution else 1))
            
            for point_idx in range(num_points - 1):  # Assuming open profile
                # Current step vertices
                v0 = step * num_points + point_idx
                v1 = step * num_points + point_idx + 1
                
                # Next step vertices
                v2 = next_step * num_points + point_idx
                v3 = next_step * num_points + point_idx + 1
                
                # Create two triangles for the quad
                faces.extend([[v0, v2, v1], [v1, v2, v3]])
        
        faces = np.array(faces, dtype=np.int32)
        
        return MeshData(vertices=vertices, faces=faces)
    
    def _perform_path_extrude(self, shape: Shape2D, path: List[Tuple[float, float, float]], 
                            params: Dict[str, Any]) -> MeshData:
        """Perform extrusion along a 3D path"""
        # This is an advanced feature that would require complex 3D transformations
        # For now, return a simple linear extrusion
        return self._perform_linear_extrude(shape, 1.0, False, 0.0, [1.0, 1.0], 1)


class Shape2DFactory:
    """Factory for creating 2D shapes"""
    
    def __init__(self):
        self.generators = {
            'circle': CircleGenerator(),
            'square': SquareGenerator(),
            'rectangle': SquareGenerator(),  # Alias
            'polygon': PolygonGenerator(),
            'regular_polygon': RegularPolygonGenerator(),
        }
        
        self.extrusion_engine = ExtrusionEngine()
    
    def create_2d_shape(self, name: str, params: Dict[str, Any]) -> Shape2D:
        """Create a 2D shape"""
        if name not in self.generators:
            raise ValueError(f"Unknown 2D shape type: {name}")
        
        generator = self.generators[name]
        return generator.generate(params)
    
    def linear_extrude(self, shape_name: str, shape_params: Dict[str, Any], 
                      extrude_params: Dict[str, Any]) -> MeshData:
        """Create 2D shape and perform linear extrusion"""
        shape = self.create_2d_shape(shape_name, shape_params)
        return self.extrusion_engine.linear_extrude(shape, **extrude_params)
    
    def rotate_extrude(self, shape_name: str, shape_params: Dict[str, Any], 
                      extrude_params: Dict[str, Any]) -> MeshData:
        """Create 2D shape and perform rotational extrusion"""
        shape = self.create_2d_shape(shape_name, shape_params)
        return self.extrusion_engine.rotate_extrude(shape, **extrude_params)
    
    def get_available_shapes(self) -> List[str]:
        """Get list of available 2D shapes"""
        return list(self.generators.keys())


# Global factory instance
shape_2d_factory = Shape2DFactory()


# Convenience functions
def create_2d_shape(name: str, params: Dict[str, Any]) -> Shape2D:
    """Create a 2D shape"""
    return shape_2d_factory.create_2d_shape(name, params)


def linear_extrude(shape_name: str, shape_params: Dict[str, Any], **extrude_params) -> MeshData:
    """Create 2D shape and extrude it linearly"""
    return shape_2d_factory.linear_extrude(shape_name, shape_params, extrude_params)


def rotate_extrude(shape_name: str, shape_params: Dict[str, Any], **extrude_params) -> MeshData:
    """Create 2D shape and extrude it rotationally"""
    return shape_2d_factory.rotate_extrude(shape_name, shape_params, extrude_params)
