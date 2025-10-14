"""
File I/O Engine
Support for multiple 3D file formats with improved mesh quality
"""

import numpy as np
import struct
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .primitives import MeshData


@dataclass
class ExportOptions:
    """Options for exporting models"""
    format: str
    binary: bool = True
    precision: int = 6
    merge_vertices: bool = True
    remove_duplicates: bool = True
    smooth_normals: bool = False
    scale: float = 1.0
    units: str = "mm"  # mm, cm, m, in, ft


@dataclass
class ImportResult:
    """Result of importing a model"""
    mesh: MeshData
    metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


class FileHandler(ABC):
    """Abstract base class for file format handlers"""
    
    @abstractmethod
    def can_export(self) -> bool:
        """Whether this handler supports export"""
        pass
    
    @abstractmethod
    def can_import(self) -> bool:
        """Whether this handler supports import"""
        pass
    
    @abstractmethod
    def get_extensions(self) -> List[str]:
        """Get supported file extensions"""
        pass
    
    @abstractmethod
    def export(self, mesh: MeshData, filepath: str, options: ExportOptions) -> bool:
        """Export mesh to file"""
        pass
    
    @abstractmethod
    def import_file(self, filepath: str) -> ImportResult:
        """Import mesh from file"""
        pass


class STLHandler(FileHandler):
    """Handler for STL files (binary and ASCII)"""
    
    def can_export(self) -> bool:
        return True
    
    def can_import(self) -> bool:
        return True
    
    def get_extensions(self) -> List[str]:
        return ['.stl']
    
    def export(self, mesh: MeshData, filepath: str, options: ExportOptions) -> bool:
        """Export mesh to STL file"""
        try:
            # Prepare mesh data
            vertices = mesh.vertices * options.scale
            faces = mesh.faces
            
            if options.binary:
                return self._export_binary_stl(vertices, faces, filepath)
            else:
                return self._export_ascii_stl(vertices, faces, filepath)
        
        except Exception as e:
            print(f"Error exporting STL: {e}")
            return False
    
    def import_file(self, filepath: str) -> ImportResult:
        """Import mesh from STL file"""
        warnings = []
        errors = []
        
        try:
            # Determine if binary or ASCII
            with open(filepath, 'rb') as f:
                header = f.read(80)
                if header.startswith(b'solid ') and not self._is_binary_stl(filepath):
                    mesh = self._import_ascii_stl(filepath)
                else:
                    mesh = self._import_binary_stl(filepath)
            
            return ImportResult(
                mesh=mesh,
                metadata={'format': 'STL', 'file_size': os.path.getsize(filepath)},
                warnings=warnings,
                errors=errors
            )
        
        except Exception as e:
            errors.append(str(e))
            # Return empty mesh on error
            empty_mesh = MeshData(
                vertices=np.array([[0, 0, 0]], dtype=np.float32),
                faces=np.array([[0, 0, 0]], dtype=np.int32)
            )
            return ImportResult(
                mesh=empty_mesh,
                metadata={},
                warnings=warnings,
                errors=errors
            )
    
    def _export_binary_stl(self, vertices: np.ndarray, faces: np.ndarray, filepath: str) -> bool:
        """Export binary STL"""
        with open(filepath, 'wb') as f:
            # Write 80-byte header
            header = b'Binary STL exported by AdvancedCAD' + b'\0' * 47
            f.write(header)
            
            # Write triangle count
            triangle_count = len(faces)
            f.write(struct.pack('<I', triangle_count))
            
            # Write triangles
            for face in faces:
                v0, v1, v2 = vertices[face]
                
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal = normal / normal_length
                else:
                    normal = np.array([0, 0, 1], dtype=np.float32)
                
                # Write normal (3 floats)
                f.write(struct.pack('<3f', *normal))
                
                # Write vertices (9 floats)
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                
                # Write attribute byte count (2 bytes, usually 0)
                f.write(struct.pack('<H', 0))
        
        return True
    
    def _export_ascii_stl(self, vertices: np.ndarray, faces: np.ndarray, filepath: str) -> bool:
        """Export ASCII STL"""
        with open(filepath, 'w') as f:
            f.write('solid AdvancedCAD_Export\\n')
            
            for face in faces:
                v0, v1, v2 = vertices[face]
                
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal = normal / normal_length
                else:
                    normal = np.array([0, 0, 1], dtype=np.float32)
                
                f.write(f'  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\\n')
                f.write('    outer loop\\n')
                f.write(f'      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\\n')
                f.write(f'      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\\n')
                f.write(f'      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\\n')
                f.write('    endloop\\n')
                f.write('  endfacet\\n')
            
            f.write('endsolid AdvancedCAD_Export\\n')
        
        return True
    
    def _is_binary_stl(self, filepath: str) -> bool:
        """Detect if STL file is binary"""
        try:
            with open(filepath, 'rb') as f:
                header = f.read(80)
                triangle_count_bytes = f.read(4)
                if len(triangle_count_bytes) != 4:
                    return False
                
                triangle_count = struct.unpack('<I', triangle_count_bytes)[0]
                expected_size = 80 + 4 + triangle_count * 50
                actual_size = os.path.getsize(filepath)
                
                # If sizes match closely, it's probably binary
                return abs(actual_size - expected_size) < 100
        except:
            return False
    
    def _import_binary_stl(self, filepath: str) -> MeshData:
        """Import binary STL"""
        vertices = []
        faces = []
        vertex_map = {}
        vertex_index = 0
        
        with open(filepath, 'rb') as f:
            # Skip header
            f.read(80)
            
            # Read triangle count
            triangle_count = struct.unpack('<I', f.read(4))[0]
            
            # Read triangles
            for _ in range(triangle_count):
                # Skip normal (3 floats)
                f.read(12)
                
                # Read vertices
                face_indices = []
                for _ in range(3):
                    vertex = struct.unpack('<3f', f.read(12))
                    vertex_key = tuple(np.round(vertex, 6))
                    
                    if vertex_key not in vertex_map:
                        vertex_map[vertex_key] = vertex_index
                        vertices.append(vertex)
                        vertex_index += 1
                    
                    face_indices.append(vertex_map[vertex_key])
                
                faces.append(face_indices)
                
                # Skip attribute bytes
                f.read(2)
        
        return MeshData(
            vertices=np.array(vertices, dtype=np.float32),
            faces=np.array(faces, dtype=np.int32)
        )
    
    def _import_ascii_stl(self, filepath: str) -> MeshData:
        """Import ASCII STL"""
        vertices = []
        faces = []
        vertex_map = {}
        vertex_index = 0
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('facet normal'):
                # Skip to outer loop
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('outer loop'):
                    i += 1
                i += 1
                
                # Read three vertices
                face_indices = []
                for _ in range(3):
                    if i >= len(lines):
                        break
                    vertex_line = lines[i].strip()
                    if vertex_line.startswith('vertex'):
                        coords = vertex_line.split()[1:4]
                        vertex = tuple(float(c) for c in coords)
                        vertex_key = tuple(np.round(vertex, 6))
                        
                        if vertex_key not in vertex_map:
                            vertex_map[vertex_key] = vertex_index
                            vertices.append(vertex)
                            vertex_index += 1
                        
                        face_indices.append(vertex_map[vertex_key])
                    i += 1
                
                if len(face_indices) == 3:
                    faces.append(face_indices)
            else:
                i += 1
        
        return MeshData(
            vertices=np.array(vertices, dtype=np.float32),
            faces=np.array(faces, dtype=np.int32)
        )


class OBJHandler(FileHandler):
    """Handler for Wavefront OBJ files"""
    
    def can_export(self) -> bool:
        return True
    
    def can_import(self) -> bool:
        return True
    
    def get_extensions(self) -> List[str]:
        return ['.obj']
    
    def export(self, mesh: MeshData, filepath: str, options: ExportOptions) -> bool:
        """Export mesh to OBJ file"""
        try:
            vertices = mesh.vertices * options.scale
            faces = mesh.faces
            normals = mesh.normals
            
            with open(filepath, 'w') as f:
                f.write('# OBJ exported by AdvancedCAD\\n')
                f.write(f'# Vertices: {len(vertices)}\\n')
                f.write(f'# Faces: {len(faces)}\\n\\n')
                
                # Write vertices
                for vertex in vertices:
                    f.write(f'v {vertex[0]:.{options.precision}f} {vertex[1]:.{options.precision}f} {vertex[2]:.{options.precision}f}\\n')
                
                f.write('\\n')
                
                # Write normals if available
                if normals is not None:
                    for normal in normals:
                        f.write(f'vn {normal[0]:.{options.precision}f} {normal[1]:.{options.precision}f} {normal[2]:.{options.precision}f}\\n')
                    f.write('\\n')
                
                # Write faces (OBJ uses 1-based indexing)
                for face in faces:
                    if normals is not None:
                        f.write(f'f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\\n')
                    else:
                        f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\\n')
            
            return True
        
        except Exception as e:
            print(f"Error exporting OBJ: {e}")
            return False
    
    def import_file(self, filepath: str) -> ImportResult:
        """Import mesh from OBJ file"""
        warnings = []
        errors = []
        
        try:
            vertices = []
            normals = []
            faces = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':
                        # Vertex
                        if len(parts) >= 4:
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
                    elif parts[0] == 'vn':
                        # Normal
                        if len(parts) >= 4:
                            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
                    elif parts[0] == 'f':
                        # Face
                        face_vertices = []
                        for i in range(1, min(4, len(parts))):  # Only support triangles
                            vertex_data = parts[i].split('/')
                            vertex_idx = int(vertex_data[0]) - 1  # Convert to 0-based
                            face_vertices.append(vertex_idx)
                        
                        if len(face_vertices) == 3:
                            faces.append(face_vertices)
                        elif len(face_vertices) > 3:
                            warnings.append(f"Face with {len(face_vertices)} vertices found, only triangles supported")
            
            mesh = MeshData(
                vertices=np.array(vertices, dtype=np.float32),
                faces=np.array(faces, dtype=np.int32),
                normals=np.array(normals, dtype=np.float32) if normals else None
            )
            
            return ImportResult(
                mesh=mesh,
                metadata={'format': 'OBJ', 'file_size': os.path.getsize(filepath)},
                warnings=warnings,
                errors=errors
            )
        
        except Exception as e:
            errors.append(str(e))
            empty_mesh = MeshData(
                vertices=np.array([[0, 0, 0]], dtype=np.float32),
                faces=np.array([[0, 0, 0]], dtype=np.int32)
            )
            return ImportResult(
                mesh=empty_mesh,
                metadata={},
                warnings=warnings,
                errors=errors
            )


class PLYHandler(FileHandler):
    """Handler for PLY files"""
    
    def can_export(self) -> bool:
        return True
    
    def can_import(self) -> bool:
        return False  # Import not implemented for PLY yet
    
    def get_extensions(self) -> List[str]:
        return ['.ply']
    
    def export(self, mesh: MeshData, filepath: str, options: ExportOptions) -> bool:
        """Export mesh to PLY file"""
        try:
            vertices = mesh.vertices * options.scale
            faces = mesh.faces
            normals = mesh.normals
            colors = mesh.colors
            
            with open(filepath, 'w') as f:
                f.write('ply\\n')
                f.write('format ascii 1.0\\n')
                f.write(f'element vertex {len(vertices)}\\n')
                f.write('property float x\\n')
                f.write('property float y\\n')
                f.write('property float z\\n')
                
                if normals is not None:
                    f.write('property float nx\\n')
                    f.write('property float ny\\n')
                    f.write('property float nz\\n')
                
                if colors is not None:
                    f.write('property uchar red\\n')
                    f.write('property uchar green\\n')
                    f.write('property uchar blue\\n')
                
                f.write(f'element face {len(faces)}\\n')
                f.write('property list uchar int vertex_indices\\n')
                f.write('end_header\\n')
                
                # Write vertices
                for i, vertex in enumerate(vertices):
                    f.write(f'{vertex[0]:.{options.precision}f} {vertex[1]:.{options.precision}f} {vertex[2]:.{options.precision}f}')
                    
                    if normals is not None:
                        normal = normals[i]
                        f.write(f' {normal[0]:.{options.precision}f} {normal[1]:.{options.precision}f} {normal[2]:.{options.precision}f}')
                    
                    if colors is not None:
                        color = colors[i]
                        r, g, b = (color[:3] * 255).astype(int)
                        f.write(f' {r} {g} {b}')
                    
                    f.write('\\n')
                
                # Write faces
                for face in faces:
                    f.write(f'3 {face[0]} {face[1]} {face[2]}\\n')
            
            return True
        
        except Exception as e:
            print(f"Error exporting PLY: {e}")
            return False
    
    def import_file(self, filepath: str) -> ImportResult:
        """Import mesh from PLY file (not implemented)"""
        return ImportResult(
            mesh=MeshData(
                vertices=np.array([[0, 0, 0]], dtype=np.float32),
                faces=np.array([[0, 0, 0]], dtype=np.int32)
            ),
            metadata={},
            warnings=[],
            errors=["PLY import not implemented"]
        )


class AdvancedCADHandler(FileHandler):
    """Handler for AdvancedCAD native format (JSON-based)"""
    
    def can_export(self) -> bool:
        return True
    
    def can_import(self) -> bool:
        return True
    
    def get_extensions(self) -> List[str]:
        return ['.acad', '.json']
    
    def export(self, mesh: MeshData, filepath: str, options: ExportOptions) -> bool:
        """Export mesh to AdvancedCAD format"""
        try:
            data = {
                'format': 'AdvancedCAD',
                'version': '1.0',
                'metadata': {
                    'units': options.units,
                    'scale': options.scale,
                    'export_options': {
                        'precision': options.precision,
                        'merge_vertices': options.merge_vertices,
                        'remove_duplicates': options.remove_duplicates
                    }
                },
                'geometry': {
                    'vertices': mesh.vertices.tolist(),
                    'faces': mesh.faces.tolist(),
                    'normals': mesh.normals.tolist() if mesh.normals is not None else None,
                    'colors': mesh.colors.tolist() if mesh.colors is not None else None,
                    'uvs': mesh.uvs.tolist() if mesh.uvs is not None else None
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error exporting AdvancedCAD format: {e}")
            return False
    
    def import_file(self, filepath: str) -> ImportResult:
        """Import mesh from AdvancedCAD format"""
        warnings = []
        errors = []
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            geometry = data.get('geometry', {})
            
            vertices = np.array(geometry.get('vertices', []), dtype=np.float32)
            faces = np.array(geometry.get('faces', []), dtype=np.int32)
            normals = None
            colors = None
            uvs = None
            
            if geometry.get('normals'):
                normals = np.array(geometry['normals'], dtype=np.float32)
            
            if geometry.get('colors'):
                colors = np.array(geometry['colors'], dtype=np.float32)
            
            if geometry.get('uvs'):
                uvs = np.array(geometry['uvs'], dtype=np.float32)
            
            mesh = MeshData(
                vertices=vertices,
                faces=faces,
                normals=normals,
                colors=colors,
                uvs=uvs
            )
            
            metadata = data.get('metadata', {})
            metadata['format'] = 'AdvancedCAD'
            metadata['file_size'] = os.path.getsize(filepath)
            
            return ImportResult(
                mesh=mesh,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )
        
        except Exception as e:
            errors.append(str(e))
            empty_mesh = MeshData(
                vertices=np.array([[0, 0, 0]], dtype=np.float32),
                faces=np.array([[0, 0, 0]], dtype=np.int32)
            )
            return ImportResult(
                mesh=empty_mesh,
                metadata={},
                warnings=warnings,
                errors=errors
            )


class FileIOEngine:
    """Main file I/O engine"""
    
    def __init__(self):
        self.handlers = {
            'stl': STLHandler(),
            'obj': OBJHandler(),
            'ply': PLYHandler(),
            'acad': AdvancedCADHandler(),
        }
    
    def get_supported_export_formats(self) -> Dict[str, List[str]]:
        """Get supported export formats"""
        formats = {}
        for name, handler in self.handlers.items():
            if handler.can_export():
                formats[name] = handler.get_extensions()
        return formats
    
    def get_supported_import_formats(self) -> Dict[str, List[str]]:
        """Get supported import formats"""
        formats = {}
        for name, handler in self.handlers.items():
            if handler.can_import():
                formats[name] = handler.get_extensions()
        return formats
    
    def export_mesh(self, mesh: MeshData, filepath: str, options: ExportOptions = None) -> bool:
        """Export mesh to file"""
        if options is None:
            options = ExportOptions(format='stl')
        
        # Determine format from file extension
        file_ext = Path(filepath).suffix.lower()
        handler = self._get_handler_by_extension(file_ext)
        
        if handler is None or not handler.can_export():
            print(f"No export handler available for {file_ext}")
            return False
        
        return handler.export(mesh, filepath, options)
    
    def import_mesh(self, filepath: str) -> ImportResult:
        """Import mesh from file"""
        file_ext = Path(filepath).suffix.lower()
        handler = self._get_handler_by_extension(file_ext)
        
        if handler is None or not handler.can_import():
            return ImportResult(
                mesh=MeshData(
                    vertices=np.array([[0, 0, 0]], dtype=np.float32),
                    faces=np.array([[0, 0, 0]], dtype=np.int32)
                ),
                metadata={},
                warnings=[],
                errors=[f"No import handler available for {file_ext}"]
            )
        
        return handler.import_file(filepath)
    
    def _get_handler_by_extension(self, ext: str) -> Optional[FileHandler]:
        """Get handler for file extension"""
        for handler in self.handlers.values():
            if ext in handler.get_extensions():
                return handler
        return None
    
    def optimize_mesh(self, mesh: MeshData, options: ExportOptions) -> MeshData:
        """Optimize mesh before export"""
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        normals = mesh.normals.copy() if mesh.normals is not None else None
        
        if options.merge_vertices:
            vertices, faces, normals = self._merge_vertices(vertices, faces, normals)
        
        if options.remove_duplicates:
            faces = self._remove_duplicate_faces(faces)
        
        return MeshData(
            vertices=vertices,
            faces=faces,
            normals=normals,
            colors=mesh.colors,
            uvs=mesh.uvs
        )
    
    def _merge_vertices(self, vertices: np.ndarray, faces: np.ndarray, 
                       normals: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Merge duplicate vertices"""
        # Simple vertex merging based on position
        unique_vertices = []
        vertex_map = {}
        
        for i, vertex in enumerate(vertices):
            key = tuple(np.round(vertex, 6))
            if key not in vertex_map:
                vertex_map[key] = len(unique_vertices)
                unique_vertices.append(vertex)
        
        # Remap faces
        new_faces = []
        for face in faces:
            new_face = []
            for vertex_idx in face:
                vertex = vertices[vertex_idx]
                key = tuple(np.round(vertex, 6))
                new_face.append(vertex_map[key])
            new_faces.append(new_face)
        
        # Remap normals if available
        new_normals = None
        if normals is not None:
            new_normals = []
            for vertex in unique_vertices:
                # Find original index
                for i, orig_vertex in enumerate(vertices):
                    if np.allclose(vertex, orig_vertex, atol=1e-6):
                        new_normals.append(normals[i])
                        break
            new_normals = np.array(new_normals, dtype=np.float32)
        
        return (np.array(unique_vertices, dtype=np.float32),
                np.array(new_faces, dtype=np.int32),
                new_normals)
    
    def _remove_duplicate_faces(self, faces: np.ndarray) -> np.ndarray:
        """Remove duplicate faces"""
        unique_faces = []
        face_set = set()
        
        for face in faces:
            # Sort face indices to detect duplicates regardless of winding order
            sorted_face = tuple(sorted(face))
            if sorted_face not in face_set:
                face_set.add(sorted_face)
                unique_faces.append(face)
        
        return np.array(unique_faces, dtype=np.int32)


# Global file I/O engine instance
file_io_engine = FileIOEngine()


# Convenience functions
def export_mesh(mesh: MeshData, filepath: str, **options) -> bool:
    """Export mesh to file"""
    export_options = ExportOptions(format=Path(filepath).suffix[1:], **options)
    return file_io_engine.export_mesh(mesh, filepath, export_options)


def import_mesh(filepath: str) -> ImportResult:
    """Import mesh from file"""
    return file_io_engine.import_mesh(filepath)
