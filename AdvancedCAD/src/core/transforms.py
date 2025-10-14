"""
Advanced Transformation and Animation System
Comprehensive 3D transformations with animation support
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .primitives import MeshData


class TransformType(Enum):
    """Types of transformations"""
    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"
    MIRROR = "mirror"
    MATRIX = "matrix"


@dataclass
class Transform:
    """Single transformation"""
    type: TransformType
    params: Dict[str, Any]
    matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.matrix is None:
            self.matrix = self._compute_matrix()
    
    def _compute_matrix(self) -> np.ndarray:
        """Compute transformation matrix"""
        if self.type == TransformType.TRANSLATE:
            return create_translation_matrix(**self.params)
        elif self.type == TransformType.ROTATE:
            return create_rotation_matrix(**self.params)
        elif self.type == TransformType.SCALE:
            return create_scale_matrix(**self.params)
        elif self.type == TransformType.MIRROR:
            return create_mirror_matrix(**self.params)
        elif self.type == TransformType.MATRIX:
            return self.params.get('matrix', np.eye(4))
        else:
            return np.eye(4)


@dataclass
class KeyFrame:
    """Animation keyframe"""
    time: float
    transforms: List[Transform]
    
    def get_combined_matrix(self) -> np.ndarray:
        """Get combined transformation matrix for this keyframe"""
        result = np.eye(4)
        for transform in self.transforms:
            result = np.dot(result, transform.matrix)
        return result


@dataclass
class Animation:
    """Animation sequence"""
    name: str
    duration: float
    keyframes: List[KeyFrame]
    loop: bool = False
    
    def get_transform_at_time(self, time: float) -> np.ndarray:
        """Get transformation matrix at specific time"""
        if self.loop and self.duration > 0:
            time = time % self.duration
        
        time = max(0, min(time, self.duration))
        
        if not self.keyframes:
            return np.eye(4)
        
        # Find surrounding keyframes
        before_frame = None
        after_frame = None
        
        for keyframe in self.keyframes:
            if keyframe.time <= time:
                before_frame = keyframe
            if keyframe.time >= time and after_frame is None:
                after_frame = keyframe
                break
        
        if before_frame is None:
            return self.keyframes[0].get_combined_matrix()
        
        if after_frame is None or before_frame == after_frame:
            return before_frame.get_combined_matrix()
        
        # Interpolate between keyframes
        t = (time - before_frame.time) / (after_frame.time - before_frame.time)
        return self._interpolate_matrices(
            before_frame.get_combined_matrix(),
            after_frame.get_combined_matrix(),
            t
        )
    
    def _interpolate_matrices(self, matrix1: np.ndarray, matrix2: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two transformation matrices"""
        # Simple linear interpolation for translation
        # For rotation, this is a simplified approach - proper interpolation would use quaternions
        return (1 - t) * matrix1 + t * matrix2


class TransformationEngine:
    """Engine for applying transformations to meshes"""
    
    def __init__(self):
        pass
    
    def translate(self, mesh: MeshData, translation: List[float]) -> MeshData:
        """Translate mesh"""
        matrix = create_translation_matrix(translation)
        return self.apply_matrix(mesh, matrix)
    
    def rotate(self, mesh: MeshData, rotation: List[float], center: Optional[List[float]] = None) -> MeshData:
        """Rotate mesh"""
        matrix = create_rotation_matrix(rotation, center)
        return self.apply_matrix(mesh, matrix)
    
    def scale(self, mesh: MeshData, scale: List[float], center: Optional[List[float]] = None) -> MeshData:
        """Scale mesh"""
        matrix = create_scale_matrix(scale, center)
        return self.apply_matrix(mesh, matrix)
    
    def mirror(self, mesh: MeshData, plane_normal: List[float], plane_point: Optional[List[float]] = None) -> MeshData:
        """Mirror mesh across plane"""
        matrix = create_mirror_matrix(plane_normal, plane_point)
        return self.apply_matrix(mesh, matrix)
    
    def apply_matrix(self, mesh: MeshData, matrix: np.ndarray) -> MeshData:
        """Apply transformation matrix to mesh"""
        # Transform vertices
        vertices_homogeneous = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
        transformed_vertices = np.dot(matrix, vertices_homogeneous.T).T
        new_vertices = transformed_vertices[:, :3]
        
        # Transform normals (use inverse transpose for normals)
        new_normals = None
        if mesh.normals is not None:
            normal_matrix = np.linalg.inv(matrix[:3, :3]).T
            new_normals = np.dot(mesh.normals, normal_matrix.T)
            # Renormalize normals
            norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
            new_normals = new_normals / np.where(norms > 0, norms, 1)
        
        return MeshData(
            vertices=new_vertices.astype(np.float32),
            faces=mesh.faces.copy(),
            normals=new_normals.astype(np.float32) if new_normals is not None else None,
            colors=mesh.colors.copy() if mesh.colors is not None else None,
            uvs=mesh.uvs.copy() if mesh.uvs is not None else None
        )
    
    def combine_transforms(self, transforms: List[Transform]) -> np.ndarray:
        """Combine multiple transformations into single matrix"""
        result = np.eye(4)
        for transform in transforms:
            result = np.dot(result, transform.matrix)
        return result
    
    def animate_mesh(self, mesh: MeshData, animation: Animation, time: float) -> MeshData:
        """Apply animation to mesh at specific time"""
        matrix = animation.get_transform_at_time(time)
        return self.apply_matrix(mesh, matrix)


class PathFollower:
    """Tool for following 3D paths"""
    
    def __init__(self, path_points: List[Tuple[float, float, float]]):
        self.path_points = np.array(path_points)
        self.path_length = self._calculate_path_length()
        self.segment_lengths = self._calculate_segment_lengths()
    
    def _calculate_path_length(self) -> float:
        """Calculate total path length"""
        if len(self.path_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.path_points)):
            segment = self.path_points[i] - self.path_points[i-1]
            total_length += np.linalg.norm(segment)
        
        return total_length
    
    def _calculate_segment_lengths(self) -> List[float]:
        """Calculate individual segment lengths"""
        if len(self.path_points) < 2:
            return []
        
        lengths = []
        for i in range(1, len(self.path_points)):
            segment = self.path_points[i] - self.path_points[i-1]
            lengths.append(np.linalg.norm(segment))
        
        return lengths
    
    def get_position_at_distance(self, distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and direction at specific distance along path"""
        if self.path_length == 0:
            return self.path_points[0] if len(self.path_points) > 0 else np.array([0, 0, 0]), np.array([1, 0, 0])
        
        # Clamp distance to path bounds
        distance = max(0, min(distance, self.path_length))
        
        # Find which segment contains this distance
        current_distance = 0.0
        for i, segment_length in enumerate(self.segment_lengths):
            if current_distance + segment_length >= distance:
                # Interpolate within this segment
                t = (distance - current_distance) / segment_length if segment_length > 0 else 0
                
                start_point = self.path_points[i]
                end_point = self.path_points[i + 1]
                
                position = start_point + t * (end_point - start_point)
                direction = end_point - start_point
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                else:
                    direction = np.array([1, 0, 0])
                
                return position, direction
            
            current_distance += segment_length
        
        # Fallback to end of path
        return self.path_points[-1], np.array([1, 0, 0])
    
    def get_position_at_time(self, time: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and direction at specific time (0 to duration)"""
        if duration <= 0:
            return self.get_position_at_distance(0)
        
        progress = time / duration
        progress = max(0, min(progress, 1))
        distance = progress * self.path_length
        
        return self.get_position_at_distance(distance)


# Matrix creation functions
def create_translation_matrix(translation: List[float]) -> np.ndarray:
    """Create translation matrix"""
    if isinstance(translation, dict) and 'v' in translation:
        translation = translation['v']
    elif isinstance(translation, dict):
        translation = [translation.get('x', 0), translation.get('y', 0), translation.get('z', 0)]
    
    if len(translation) != 3:
        raise ValueError("Translation must have 3 components")
    
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return matrix


def create_rotation_matrix(rotation: List[float], center: Optional[List[float]] = None) -> np.ndarray:
    """Create rotation matrix"""
    if isinstance(rotation, dict) and 'v' in rotation:
        rotation = rotation['v']
    elif isinstance(rotation, dict):
        rotation = [rotation.get('x', 0), rotation.get('y', 0), rotation.get('z', 0)]
    
    if len(rotation) != 3:
        raise ValueError("Rotation must have 3 components (angles in degrees)")
    
    # Convert degrees to radians
    rx, ry, rz = [math.radians(angle) for angle in rotation]
    
    # Create individual rotation matrices
    Rx = np.array([
        [1, 0, 0, 0],
        [0, math.cos(rx), -math.sin(rx), 0],
        [0, math.sin(rx), math.cos(rx), 0],
        [0, 0, 0, 1]
    ])
    
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry), 0],
        [0, 1, 0, 0],
        [-math.sin(ry), 0, math.cos(ry), 0],
        [0, 0, 0, 1]
    ])
    
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0, 0],
        [math.sin(rz), math.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Combine rotations (order: Z, Y, X)
    rotation_matrix = np.dot(np.dot(Rx, Ry), Rz)
    
    # Apply center offset if specified
    if center is not None and len(center) == 3:
        translate_to_origin = create_translation_matrix([-c for c in center])
        translate_back = create_translation_matrix(center)
        rotation_matrix = np.dot(np.dot(translate_back, rotation_matrix), translate_to_origin)
    
    return rotation_matrix


def create_scale_matrix(scale: List[float], center: Optional[List[float]] = None) -> np.ndarray:
    """Create scale matrix"""
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]
    elif isinstance(scale, dict) and 'v' in scale:
        scale = scale['v']
    elif isinstance(scale, dict):
        scale = [scale.get('x', 1), scale.get('y', 1), scale.get('z', 1)]
    
    if len(scale) == 1:
        scale = [scale[0], scale[0], scale[0]]
    elif len(scale) != 3:
        raise ValueError("Scale must have 1 or 3 components")
    
    matrix = np.eye(4)
    matrix[0, 0] = scale[0]
    matrix[1, 1] = scale[1]
    matrix[2, 2] = scale[2]
    
    # Apply center offset if specified
    if center is not None and len(center) == 3:
        translate_to_origin = create_translation_matrix([-c for c in center])
        translate_back = create_translation_matrix(center)
        matrix = np.dot(np.dot(translate_back, matrix), translate_to_origin)
    
    return matrix


def create_mirror_matrix(plane_normal: List[float], plane_point: Optional[List[float]] = None) -> np.ndarray:
    """Create mirror matrix across a plane"""
    if isinstance(plane_normal, dict) and 'v' in plane_normal:
        plane_normal = plane_normal['v']
    elif isinstance(plane_normal, dict):
        plane_normal = [plane_normal.get('x', 1), plane_normal.get('y', 0), plane_normal.get('z', 0)]
    
    if len(plane_normal) != 3:
        raise ValueError("Plane normal must have 3 components")
    
    # Normalize the plane normal
    normal = np.array(plane_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    
    # Create reflection matrix
    I = np.eye(3)
    mirror_3x3 = I - 2 * np.outer(normal, normal)
    
    matrix = np.eye(4)
    matrix[:3, :3] = mirror_3x3
    
    # Apply plane point offset if specified
    if plane_point is not None and len(plane_point) == 3:
        # Translate to put plane at origin, mirror, then translate back
        translate_to_origin = create_translation_matrix([-p for p in plane_point])
        translate_back = create_translation_matrix(plane_point)
        matrix = np.dot(np.dot(translate_back, matrix), translate_to_origin)
    
    return matrix


# Global transformation engine
transform_engine = TransformationEngine()


# Convenience functions
def translate_mesh(mesh: MeshData, translation: List[float]) -> MeshData:
    """Translate mesh"""
    return transform_engine.translate(mesh, translation)


def rotate_mesh(mesh: MeshData, rotation: List[float], center: Optional[List[float]] = None) -> MeshData:
    """Rotate mesh"""
    return transform_engine.rotate(mesh, rotation, center)


def scale_mesh(mesh: MeshData, scale: List[float], center: Optional[List[float]] = None) -> MeshData:
    """Scale mesh"""
    return transform_engine.scale(mesh, scale, center)


def mirror_mesh(mesh: MeshData, plane_normal: List[float], plane_point: Optional[List[float]] = None) -> MeshData:
    """Mirror mesh"""
    return transform_engine.mirror(mesh, plane_normal, plane_point)


def apply_matrix_to_mesh(mesh: MeshData, matrix: np.ndarray) -> MeshData:
    """Apply transformation matrix to mesh"""
    return transform_engine.apply_matrix(mesh, matrix)
