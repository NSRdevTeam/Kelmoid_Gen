#!/usr/bin/env python3
"""
Advanced Constraint Types
=========================

Extended constraint library for sophisticated geometric relationships
including tangency, concentricity, equal constraints, and manufacturing-specific constraints.

Author: KelmoidAI Genesis Team  
License: MIT
"""

import numpy as np
import math
from typing import List, Optional
from constraint_solver import (
    GeometricConstraint, ConstraintType, GeometricEntity, EntityType,
    Point3D, Vector3D, Line3D, Circle3D, DistanceConstraint, create_distance_constraint
)

class PerpendicularConstraint(GeometricConstraint):
    """Perpendicular constraint between two lines"""
    
    def __init__(self, constraint_id: str, line1: Line3D, line2: Line3D):
        super().__init__(constraint_id, ConstraintType.PERPENDICULAR, [line1, line2])
        self.line1 = line1
        self.line2 = line2
    
    def evaluate(self) -> float:
        """Return dot product (should be 0 for perpendicular lines)"""
        dir1 = self.line1.get_direction().to_array()
        dir2 = self.line2.get_direction().to_array()
        return abs(np.dot(dir1, dir2))
    
    def get_gradient(self) -> List[np.ndarray]:
        return [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]

class TangentConstraint(GeometricConstraint):
    """Tangency constraint between circle and line"""
    
    def __init__(self, constraint_id: str, circle: Circle3D, line: Line3D):
        super().__init__(constraint_id, ConstraintType.TANGENT, [circle, line])
        self.circle = circle
        self.line = line
    
    def evaluate(self) -> float:
        """Return difference between distance from center to line and radius"""
        center = self.circle.center.to_array()
        line_start = self.line.start.to_array()
        line_dir = self.line.get_direction().to_array()
        
        # Vector from line start to circle center
        to_center = center - line_start
        
        # Project onto line direction
        projection = np.dot(to_center, line_dir) * line_dir
        
        # Distance from center to line
        distance_to_line = np.linalg.norm(to_center - projection)
        
        # Should equal radius for tangency
        return abs(distance_to_line - self.circle.get_parameter('radius'))
    
    def get_gradient(self) -> List[np.ndarray]:
        return [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]

class ConcentricConstraint(GeometricConstraint):
    """Concentric constraint between two circles"""
    
    def __init__(self, constraint_id: str, circle1: Circle3D, circle2: Circle3D):
        super().__init__(constraint_id, ConstraintType.CONCENTRIC, [circle1, circle2])
        self.circle1 = circle1
        self.circle2 = circle2
    
    def evaluate(self) -> float:
        """Return distance between centers (should be 0 for concentric circles)"""
        return self.circle1.center.distance_to(self.circle2.center)
    
    def get_gradient(self) -> List[np.ndarray]:
        return [np.zeros(3), np.zeros(3)]

class EqualConstraint(GeometricConstraint):
    """Equal constraint for dimensions (radius, length, etc.)"""
    
    def __init__(self, constraint_id: str, entity1: GeometricEntity, entity2: GeometricEntity, parameter_name: str):
        super().__init__(constraint_id, ConstraintType.EQUAL, [entity1, entity2])
        self.entity1 = entity1
        self.entity2 = entity2
        self.parameter_name = parameter_name
    
    def evaluate(self) -> float:
        """Return difference between parameter values"""
        value1 = self.entity1.get_parameter(self.parameter_name)
        value2 = self.entity2.get_parameter(self.parameter_name)
        return abs(value1 - value2)
    
    def get_gradient(self) -> List[np.ndarray]:
        return [np.zeros(3), np.zeros(3)]

class CoincidentConstraint(GeometricConstraint):
    """Coincident constraint between two points"""
    
    def __init__(self, constraint_id: str, point1: GeometricEntity, point2: GeometricEntity):
        super().__init__(constraint_id, ConstraintType.COINCIDENT, [point1, point2])
        self.point1 = point1
        self.point2 = point2
    
    def evaluate(self) -> float:
        """Return distance between points (should be 0 for coincident points)"""
        return self.point1.position.distance_to(self.point2.position)
    
    def get_gradient(self) -> List[np.ndarray]:
        p1 = self.point1.position.to_array()
        p2 = self.point2.position.to_array()
        diff = p1 - p2
        dist = np.linalg.norm(diff)
        
        if dist == 0:
            return [np.zeros(3), np.zeros(3)]
        
        unit_vector = diff / dist
        return [unit_vector, -unit_vector]

class HorizontalConstraint(GeometricConstraint):
    """Horizontal constraint for lines"""
    
    def __init__(self, constraint_id: str, line: Line3D):
        super().__init__(constraint_id, ConstraintType.HORIZONTAL, [line])
        self.line = line
    
    def evaluate(self) -> float:
        """Return Z-component of line direction (should be 0 for horizontal)"""
        direction = self.line.get_direction()
        return abs(direction.z)
    
    def get_gradient(self) -> List[np.ndarray]:
        return [np.zeros(3), np.zeros(3)]

class VerticalConstraint(GeometricConstraint):
    """Vertical constraint for lines"""
    
    def __init__(self, constraint_id: str, line: Line3D):
        super().__init__(constraint_id, ConstraintType.VERTICAL, [line])
        self.line = line
    
    def evaluate(self) -> float:
        """Return XY-component magnitude (should be 0 for vertical)"""
        direction = self.line.get_direction()
        xy_magnitude = math.sqrt(direction.x**2 + direction.y**2)
        return xy_magnitude
    
    def get_gradient(self) -> List[np.ndarray]:
        return [np.zeros(3), np.zeros(3)]

class FixConstraint(GeometricConstraint):
    """Fix constraint to lock entity position"""
    
    def __init__(self, constraint_id: str, entity: GeometricEntity, fixed_position: Optional[Point3D] = None):
        super().__init__(constraint_id, ConstraintType.FIX, [entity])
        self.entity = entity
        self.fixed_position = fixed_position or entity.position
    
    def evaluate(self) -> float:
        """Return distance from fixed position"""
        return self.entity.position.distance_to(self.fixed_position)
    
    def get_gradient(self) -> List[np.ndarray]:
        current = self.entity.position.to_array()
        fixed = self.fixed_position.to_array()
        diff = current - fixed
        dist = np.linalg.norm(diff)
        
        if dist == 0:
            return [np.zeros(3)]
        
        return [diff / dist]

class CollinearConstraint(GeometricConstraint):
    """Collinear constraint for three or more points"""
    
    def __init__(self, constraint_id: str, points: List[GeometricEntity]):
        super().__init__(constraint_id, ConstraintType.COLLINEAR, points)
        self.points = points
    
    def evaluate(self) -> float:
        """Return sum of distances from points to best-fit line"""
        if len(self.points) < 3:
            return 0.0
        
        # Use first two points to define line
        p1 = self.points[0].position.to_array()
        p2 = self.points[1].position.to_array()
        
        line_dir = p2 - p1
        line_length = np.linalg.norm(line_dir)
        
        if line_length == 0:
            return float('inf')  # Degenerate case
        
        line_dir = line_dir / line_length
        
        total_error = 0.0
        
        # Check remaining points
        for i in range(2, len(self.points)):
            point = self.points[i].position.to_array()
            
            # Vector from p1 to point
            to_point = point - p1
            
            # Project onto line
            projection = np.dot(to_point, line_dir) * line_dir
            
            # Distance from point to line
            distance = np.linalg.norm(to_point - projection)
            total_error += distance**2
        
        return math.sqrt(total_error)
    
    def get_gradient(self) -> List[np.ndarray]:
        # Simplified - full gradient would be complex
        return [np.zeros(3) for _ in self.points]

# Manufacturing-specific constraints
class ManufacturingConstraints:
    """Manufacturing-specific constraint utilities"""
    
    @staticmethod
    def create_hole_pattern_constraint(constraint_id: str, center: GeometricEntity, 
                                     holes: List[GeometricEntity], radius: float, 
                                     num_holes: int) -> List[GeometricConstraint]:
        """Create constraints for circular hole pattern"""
        constraints = []
        angle_step = 2 * math.pi / num_holes
        
        for i, hole in enumerate(holes):
            # Distance from center
            dist_constraint = DistanceConstraint(f"{constraint_id}_dist_{i}", center, hole, radius)
            constraints.append(dist_constraint)
        
        return constraints
    
    @staticmethod
    def create_bolt_circle_constraint(constraint_id: str, center: GeometricEntity, 
                                    bolts: List[GeometricEntity], diameter: float) -> List[GeometricConstraint]:
        """Create constraints for bolt circle pattern"""
        return ManufacturingConstraints.create_hole_pattern_constraint(
            constraint_id, center, bolts, diameter/2, len(bolts)
        )

# Utility functions for creating advanced constraints
def create_perpendicular_constraint(id: str, line1: Line3D, line2: Line3D) -> PerpendicularConstraint:
    return PerpendicularConstraint(id, line1, line2)

def create_tangent_constraint(id: str, circle: Circle3D, line: Line3D) -> TangentConstraint:
    return TangentConstraint(id, circle, line)

def create_concentric_constraint(id: str, circle1: Circle3D, circle2: Circle3D) -> ConcentricConstraint:
    return ConcentricConstraint(id, circle1, circle2)

def create_equal_constraint(id: str, entity1: GeometricEntity, entity2: GeometricEntity, parameter: str) -> EqualConstraint:
    return EqualConstraint(id, entity1, entity2, parameter)

def create_coincident_constraint(id: str, point1: GeometricEntity, point2: GeometricEntity) -> CoincidentConstraint:
    return CoincidentConstraint(id, point1, point2)

def create_horizontal_constraint(id: str, line: Line3D) -> HorizontalConstraint:
    return HorizontalConstraint(id, line)

def create_vertical_constraint(id: str, line: Line3D) -> VerticalConstraint:
    return VerticalConstraint(id, line)

def create_fix_constraint(id: str, entity: GeometricEntity, position: Optional[Point3D] = None) -> FixConstraint:
    return FixConstraint(id, entity, position)

def create_collinear_constraint(id: str, points: List[GeometricEntity]) -> CollinearConstraint:
    return CollinearConstraint(id, points)

# Test advanced constraints
if __name__ == "__main__":
    from constraint_solver import ConstraintSolver, EntityType
    
    print("🔧 Testing Advanced Constraints")
    print("=" * 50)
    
    solver = ConstraintSolver()
    
    # Create test entities
    center = GeometricEntity("center", EntityType.POINT, position=Point3D(0, 0, 0))
    p1 = GeometricEntity("p1", EntityType.POINT, position=Point3D(10, 0, 0))
    p2 = GeometricEntity("p2", EntityType.POINT, position=Point3D(0, 10, 0))
    p3 = GeometricEntity("p3", EntityType.POINT, position=Point3D(5, 5, 0))
    
    # Create lines
    line1 = Line3D("line1", Point3D(0, 0, 0), Point3D(10, 0, 0))
    line2 = Line3D("line2", Point3D(0, 0, 0), Point3D(0, 10, 0))
    
    # Create circles
    circle1 = Circle3D("circle1", Point3D(0, 0, 0), 5.0)
    circle2 = Circle3D("circle2", Point3D(1, 1, 0), 3.0)
    
    solver.add_entity(center)
    solver.add_entity(p1)
    solver.add_entity(p2)
    solver.add_entity(p3)
    solver.add_entity(line1)
    solver.add_entity(line2)
    solver.add_entity(circle1)
    solver.add_entity(circle2)
    
    # Test perpendicular constraint
    perp_constraint = create_perpendicular_constraint("perp1", line1, line2)
    solver.add_constraint(perp_constraint)
    
    print("Testing perpendicular constraint:")
    print(f"Initial violation: {perp_constraint.evaluate():.6f}")
    print(f"Is satisfied: {perp_constraint.is_satisfied()}")
    
    # Test equal distance constraint (make p1 and p2 equidistant from center)
    dist1 = create_distance_constraint("dist1", center, p1, 10.0)
    dist2 = create_distance_constraint("dist2", center, p2, 10.0)
    
    solver.add_constraint(dist1)
    solver.add_constraint(dist2)
    
    # Test fix constraint (keep center fixed)
    fix_center = create_fix_constraint("fix_center", center)
    solver.add_constraint(fix_center)
    
    print("\nSolving with advanced constraints...")
    result = solver.solve()
    
    print(f"Success: {result['success']}")
    print(f"Final error: {result['final_error']:.8f}")
    print(f"Satisfied constraints: {len(result['satisfied_constraints'])}")
    print(f"Violated constraints: {len(result['violated_constraints'])}")
    
    # Test manufacturing constraints
    print("\n🏭 Testing Manufacturing Constraints")
    
    # Create hole pattern
    hole_center = GeometricEntity("hole_center", EntityType.POINT, position=Point3D(0, 0, 0))
    holes = [
        GeometricEntity("hole1", EntityType.POINT, position=Point3D(20, 0, 0)),
        GeometricEntity("hole2", EntityType.POINT, position=Point3D(0, 20, 0)),
        GeometricEntity("hole3", EntityType.POINT, position=Point3D(-20, 0, 0)),
        GeometricEntity("hole4", EntityType.POINT, position=Point3D(0, -20, 0))
    ]
    
    pattern_constraints = ManufacturingConstraints.create_bolt_circle_constraint(
        "bolt_pattern", hole_center, holes, 40.0  # 40mm diameter bolt circle
    )
    
    print(f"Created {len(pattern_constraints)} bolt circle constraints")
    
    # Test collinear constraint
    collinear_points = [p1, p3, p2]  # These should be made collinear
    collinear_constraint = create_collinear_constraint("collinear1", collinear_points)
    
    print(f"Collinear constraint violation: {collinear_constraint.evaluate():.6f}")
    
    print("\n✅ Advanced constraint system ready!")
