"""
3D Primitive Shapes Engine
High-precision parametric 3D primitives for AdvancedCAD
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import trimesh


@dataclass
class MeshData:
    """Container for mesh data"""
    vertices: np.ndarray  # Nx3 array of vertices
    faces: np.ndarray     # Mx3 array of face indices
    normals: Optional[np.ndarray] = None  # Nx3 array of vertex normals
    colors: Optional[np.ndarray] = None   # Nx4 array of vertex colors (RGBA)
    uvs: Optional[np.ndarray] = None      # Nx2 array of texture coordinates


class PrimitiveGenerator:
    """Base class for primitive shape generators"""
    
    def __init__(self):
        self.default_resolution = 32
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters"""
        return params
    
    def generate(self, params: Dict[str, Any]) -> MeshData:
        """Generate mesh data for the primitive"""
        raise NotImplementedError


class CubeGenerator(PrimitiveGenerator):
    """Generate cube/rectangular box primitives"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cube parameters"""
        validated = {
            'size': [1.0, 1.0, 1.0],
            'center': True
        }
        
        if 'size' in params:
            size = params['size']
            if isinstance(size, (int, float)):
                validated['size'] = [float(size)] * 3
            elif isinstance(size, (list, tuple)) and len(size) == 3:
                validated['size'] = [float(x) for x in size]
            elif isinstance(size, (list, tuple)) and len(size) == 1:
                validated['size'] = [float(size[0])] * 3
        
        if 'center' in params:
            validated['center'] = bool(params['center'])
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> MeshData:
        """Generate cube mesh"""
        params = self.validate_params(params)
        size = params['size']
        center = params['center']
        
        # Define cube vertices (unit cube first)
        vertices = np.array([
            # Bottom face (z = -0.5)
            [-0.5, -0.5, -0.5],  # 0
            [ 0.5, -0.5, -0.5],  # 1
            [ 0.5,  0.5, -0.5],  # 2
            [-0.5,  0.5, -0.5],  # 3
            # Top face (z = 0.5)
            [-0.5, -0.5,  0.5],  # 4
            [ 0.5, -0.5,  0.5],  # 5
            [ 0.5,  0.5,  0.5],  # 6
            [-0.5,  0.5,  0.5],  # 7
        ], dtype=np.float32)
        
        # Scale vertices
        vertices *= np.array(size)
        
        # Center or not
        if not center:
            vertices += np.array(size) / 2
        
        # Define faces (triangles)
        faces = np.array([
            # Bottom face
            [0, 1, 2], [0, 2, 3],
            # Top face
            [4, 6, 5], [4, 7, 6],
            # Front face
            [0, 4, 5], [0, 5, 1],
            # Back face
            [2, 6, 7], [2, 7, 3],
            # Left face
            [0, 3, 7], [0, 7, 4],
            # Right face
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)
        
        # Generate normals
        normals = self._generate_cube_normals()
        
        return MeshData(vertices=vertices, faces=faces, normals=normals)
    
    def _generate_cube_normals(self) -> np.ndarray:
        """Generate vertex normals for cube"""
        # Each vertex belongs to 3 faces, so we average the normals
        normals = np.array([
            [-1, -1, -1],  # 0
            [ 1, -1, -1],  # 1
            [ 1,  1, -1],  # 2
            [-1,  1, -1],  # 3
            [-1, -1,  1],  # 4
            [ 1, -1,  1],  # 5
            [ 1,  1,  1],  # 6
            [-1,  1,  1],  # 7
        ], dtype=np.float32)
        
        # Normalize
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)


class SphereGenerator(PrimitiveGenerator):
    """Generate sphere primitives"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sphere parameters"""
        validated = {
            'radius': 1.0,
            'resolution': self.default_resolution,
            'center': True
        }
        
        if 'r' in params:
            validated['radius'] = float(params['r'])
        elif 'radius' in params:
            validated['radius'] = float(params['radius'])
        
        if 'fn' in params:
            validated['resolution'] = int(params['fn'])
        elif 'resolution' in params:
            validated['resolution'] = int(params['resolution'])
        
        if 'center' in params:
            validated['center'] = bool(params['center'])
        
        # Ensure minimum resolution
        validated['resolution'] = max(8, validated['resolution'])
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> MeshData:
        """Generate sphere mesh using UV sphere topology"""
        params = self.validate_params(params)
        radius = params['radius']
        resolution = params['resolution']
        center = params['center']
        
        # Create UV sphere
        lat_divisions = resolution // 2
        lon_divisions = resolution
        
        vertices = []
        faces = []
        
        # Generate vertices
        for i in range(lat_divisions + 1):
            lat = math.pi * i / lat_divisions - math.pi / 2  # -π/2 to π/2
            
            for j in range(lon_divisions):
                lon = 2 * math.pi * j / lon_divisions  # 0 to 2π
                
                x = radius * math.cos(lat) * math.cos(lon)
                y = radius * math.cos(lat) * math.sin(lon)
                z = radius * math.sin(lat)
                
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Position sphere
        if not center:
            vertices[:, 2] += radius  # Move to sit on XY plane
        
        # Generate faces
        for i in range(lat_divisions):
            for j in range(lon_divisions):
                # Current vertex indices
                curr = i * lon_divisions + j
                next_j = i * lon_divisions + (j + 1) % lon_divisions
                next_i = (i + 1) * lon_divisions + j
                next_both = (i + 1) * lon_divisions + (j + 1) % lon_divisions
                
                # Skip degenerate triangles at poles
                if i > 0:
                    faces.append([curr, next_both, next_j])
                
                if i < lat_divisions - 1:
                    faces.append([curr, next_i, next_both])
        
        faces = np.array(faces, dtype=np.int32)
        
        # Normals for sphere are just normalized position vectors
        if center:
            normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        else:
            # Adjust for non-centered sphere
            centered_vertices = vertices.copy()
            centered_vertices[:, 2] -= radius
            normals = centered_vertices / np.linalg.norm(centered_vertices, axis=1, keepdims=True)
        
        return MeshData(vertices=vertices, faces=faces, normals=normals)


class CylinderGenerator(PrimitiveGenerator):
    """Generate cylinder primitives"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cylinder parameters"""
        validated = {
            'height': 1.0,
            'radius1': 1.0,
            'radius2': None,  # If None, use radius1 (perfect cylinder)
            'resolution': self.default_resolution,
            'center': True
        }
        
        if 'h' in params:
            validated['height'] = float(params['h'])
        elif 'height' in params:
            validated['height'] = float(params['height'])
        
        if 'r' in params:
            validated['radius1'] = float(params['r'])
            validated['radius2'] = float(params['r'])
        elif 'radius' in params:
            validated['radius1'] = float(params['radius'])
            validated['radius2'] = float(params['radius'])
        
        if 'r1' in params:
            validated['radius1'] = float(params['r1'])
        elif 'radius1' in params:
            validated['radius1'] = float(params['radius1'])
        
        if 'r2' in params:
            validated['radius2'] = float(params['r2'])
        elif 'radius2' in params:
            validated['radius2'] = float(params['radius2'])
        
        # If radius2 not specified, use radius1
        if validated['radius2'] is None:
            validated['radius2'] = validated['radius1']
        
        if 'fn' in params:
            validated['resolution'] = int(params['fn'])
        elif 'resolution' in params:
            validated['resolution'] = int(params['resolution'])
        
        if 'center' in params:
            validated['center'] = bool(params['center'])
        
        # Ensure minimum resolution
        validated['resolution'] = max(6, validated['resolution'])
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> MeshData:
        """Generate cylinder/cone mesh"""
        params = self.validate_params(params)
        height = params['height']
        r1 = params['radius1']
        r2 = params['radius2']
        resolution = params['resolution']
        center = params['center']
        
        vertices = []
        faces = []
        
        # Generate vertices for two circular ends
        for layer in range(2):
            z = height * layer  # 0 for bottom, height for top
            radius = r1 if layer == 0 else r2
            
            # Center vertex for each end
            vertices.append([0, 0, z])
            center_idx = len(vertices) - 1
            
            # Rim vertices
            rim_start = len(vertices)
            for i in range(resolution):
                angle = 2 * math.pi * i / resolution
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                vertices.append([x, y, z])
            
            # Triangular faces for end caps
            for i in range(resolution):
                next_i = (i + 1) % resolution
                if layer == 0:  # Bottom face (inward normal)
                    faces.append([center_idx, rim_start + next_i, rim_start + i])
                else:  # Top face (outward normal)
                    faces.append([center_idx, rim_start + i, rim_start + next_i])
        
        # Side faces (connecting the two rings)
        bottom_rim_start = 1  # Index of first bottom rim vertex
        top_rim_start = resolution + 2  # Index of first top rim vertex
        
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Bottom triangle
            faces.append([
                bottom_rim_start + i,
                top_rim_start + i,
                bottom_rim_start + next_i
            ])
            
            # Top triangle
            faces.append([
                bottom_rim_start + next_i,
                top_rim_start + i,
                top_rim_start + next_i
            ])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        # Center the cylinder if requested
        if center:
            vertices[:, 2] -= height / 2
        
        # Generate normals (simplified - proper normals would require more complex calculation)
        normals = np.zeros_like(vertices)
        
        # End cap normals
        normals[0] = [0, 0, -1]  # Bottom center
        normals[resolution + 1] = [0, 0, 1]  # Top center
        
        # Rim normals (approximation)
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            # Bottom rim
            normals[1 + i] = [math.cos(angle), math.sin(angle), 0]
            # Top rim
            normals[resolution + 2 + i] = [math.cos(angle), math.sin(angle), 0]
        
        return MeshData(vertices=vertices, faces=faces, normals=normals)


class ConeGenerator(CylinderGenerator):
    """Generate cone primitives (special case of cylinder with r2=0)"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cone parameters"""
        validated = super().validate_params(params)
        validated['radius2'] = 0.0  # Force top radius to 0
        return validated


class TorusGenerator(PrimitiveGenerator):
    """Generate torus primitives"""
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate torus parameters"""
        validated = {
            'major_radius': 2.0,
            'minor_radius': 0.5,
            'major_resolution': self.default_resolution,
            'minor_resolution': self.default_resolution // 2,
            'center': True
        }
        
        if 'R' in params:
            validated['major_radius'] = float(params['R'])
        elif 'major_radius' in params:
            validated['major_radius'] = float(params['major_radius'])
        
        if 'r' in params:
            validated['minor_radius'] = float(params['r'])
        elif 'minor_radius' in params:
            validated['minor_radius'] = float(params['minor_radius'])
        
        if 'fn' in params:
            validated['major_resolution'] = int(params['fn'])
            validated['minor_resolution'] = int(params['fn']) // 2
        
        if 'major_fn' in params:
            validated['major_resolution'] = int(params['major_fn'])
        
        if 'minor_fn' in params:
            validated['minor_resolution'] = int(params['minor_fn'])
        
        # Ensure minimum resolution
        validated['major_resolution'] = max(8, validated['major_resolution'])
        validated['minor_resolution'] = max(6, validated['minor_resolution'])
        
        return validated
    
    def generate(self, params: Dict[str, Any]) -> MeshData:
        """Generate torus mesh"""
        params = self.validate_params(params)
        R = params['major_radius']
        r = params['minor_radius']
        major_res = params['major_resolution']
        minor_res = params['minor_resolution']
        
        vertices = []
        faces = []
        
        # Generate vertices
        for i in range(major_res):
            u = 2 * math.pi * i / major_res  # Major angle
            
            for j in range(minor_res):
                v = 2 * math.pi * j / minor_res  # Minor angle
                
                x = (R + r * math.cos(v)) * math.cos(u)
                y = (R + r * math.cos(v)) * math.sin(u)
                z = r * math.sin(v)
                
                vertices.append([x, y, z])
        
        # Generate faces
        for i in range(major_res):
            for j in range(minor_res):
                # Current vertex indices
                curr = i * minor_res + j
                next_i = ((i + 1) % major_res) * minor_res + j
                next_j = i * minor_res + (j + 1) % minor_res
                next_both = ((i + 1) % major_res) * minor_res + (j + 1) % minor_res
                
                # Two triangles per quad
                faces.append([curr, next_i, next_j])
                faces.append([next_j, next_i, next_both])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        # Generate normals for torus
        normals = np.zeros_like(vertices)
        for i in range(len(vertices)):
            # Calculate normal based on torus geometry
            x, y, z = vertices[i]
            
            # Distance from z-axis
            dist_from_z = math.sqrt(x*x + y*y)
            
            if dist_from_z > 1e-6:  # Avoid division by zero
                # Normal vector points away from the torus center line
                nx = x * (1 - R / dist_from_z)
                ny = y * (1 - R / dist_from_z)
                nz = z
                
                # Normalize
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 1e-6:
                    normals[i] = [nx/length, ny/length, nz/length]
        
        return MeshData(vertices=vertices, faces=faces, normals=normals)


class PrimitiveFactory:
    """Factory for creating primitive shapes"""
    
    def __init__(self):
        self.generators = {
            'cube': CubeGenerator(),
            'box': CubeGenerator(),  # Alias for cube
            'sphere': SphereGenerator(),
            'cylinder': CylinderGenerator(),
            'cone': ConeGenerator(),
            'torus': TorusGenerator(),
        }
    
    def create_primitive(self, name: str, params: Dict[str, Any]) -> MeshData:
        """Create a primitive shape"""
        if name not in self.generators:
            raise ValueError(f"Unknown primitive type: {name}")
        
        generator = self.generators[name]
        return generator.generate(params)
    
    def get_available_primitives(self) -> List[str]:
        """Get list of available primitive types"""
        return list(self.generators.keys())
    
    def add_generator(self, name: str, generator: PrimitiveGenerator):
        """Add a custom primitive generator"""
        self.generators[name] = generator


# Global factory instance
primitive_factory = PrimitiveFactory()


def create_primitive(name: str, params: Dict[str, Any]) -> MeshData:
    """Convenience function to create primitives"""
    return primitive_factory.create_primitive(name, params)
