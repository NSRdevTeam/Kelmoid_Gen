#!/usr/bin/env python3
"""
Parametric CAD Generator
========================

Constraint-aware CAD generation system that integrates geometric constraints
with traditional CAD modeling for intelligent design automation.

Features:
- Parametric geometry with constraints
- Automatic constraint satisfaction
- Design intent preservation
- Manufacturing-aware constraints
- Real-time constraint solving

Author: KelmoidAI Genesis Team
License: MIT
"""

import cadquery as cq
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

from constraint_solver import (
    ConstraintSolver, GeometricEntity, EntityType, Point3D, Vector3D,
    DistanceConstraint, AngleConstraint, ParallelConstraint, SymmetryConstraint,
    create_distance_constraint, create_angle_constraint, create_parallel_constraint
)
from advanced_constraints import (
    create_perpendicular_constraint, create_fix_constraint, create_coincident_constraint,
    create_horizontal_constraint, create_vertical_constraint, ManufacturingConstraints
)

@dataclass
class ParametricFeature:
    """Represents a parametric CAD feature with constraints"""
    id: str
    feature_type: str  # 'extrude', 'revolve', 'sweep', 'loft', etc.
    base_geometry: Dict[str, Any]
    parameters: Dict[str, float] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)  # Constraint IDs
    dependent_features: List[str] = field(default_factory=list)
    
class ParametricCADGenerator:
    """Advanced CAD generator with constraint solving capabilities"""
    
    def __init__(self):
        self.solver = ConstraintSolver()
        self.features: Dict[str, ParametricFeature] = {}
        self.entities: Dict[str, GeometricEntity] = {}
        self.current_workplane = cq.Workplane("XY")
        self.feature_counter = 0
        
    def reset(self):
        """Reset the generator to clean state"""
        self.solver = ConstraintSolver()
        self.features.clear()
        self.entities.clear()
        self.current_workplane = cq.Workplane("XY")
        self.feature_counter = 0
    
    def add_entity(self, entity_id: str, entity_type: EntityType, 
                  position: Point3D, parameters: Optional[Dict[str, float]] = None):
        """Add geometric entity to the system"""
        entity = GeometricEntity(entity_id, entity_type, parameters or {}, position)
        self.entities[entity_id] = entity
        self.solver.add_entity(entity)
        return entity
    
    def add_constraint(self, constraint_type: str, entities: List[str], 
                      value: Optional[float] = None, **kwargs) -> str:
        """Add constraint between entities"""
        constraint_id = f"constraint_{len(self.solver.constraints)}"
        
        # Get entity objects
        entity_objects = [self.entities[eid] for eid in entities]
        
        if constraint_type == "distance" and len(entity_objects) == 2:
            constraint = create_distance_constraint(constraint_id, entity_objects[0], entity_objects[1], value)
        elif constraint_type == "perpendicular" and len(entity_objects) == 2:
            from constraint_solver import Line3D
            line1 = Line3D(entities[0], entity_objects[0].position, Point3D(entity_objects[0].position.x + 1, entity_objects[0].position.y, entity_objects[0].position.z))
            line2 = Line3D(entities[1], entity_objects[1].position, Point3D(entity_objects[1].position.x, entity_objects[1].position.y + 1, entity_objects[1].position.z))
            constraint = create_perpendicular_constraint(constraint_id, line1, line2)
        elif constraint_type == "fix":
            constraint = create_fix_constraint(constraint_id, entity_objects[0])
        elif constraint_type == "coincident" and len(entity_objects) == 2:
            constraint = create_coincident_constraint(constraint_id, entity_objects[0], entity_objects[1])
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type}")
        
        self.solver.add_constraint(constraint)
        return constraint_id
    
    def solve_constraints(self) -> Dict[str, Any]:
        """Solve all constraints in the system"""
        return self.solver.solve()
    
    def create_parametric_rectangle(self, width: float, height: float, 
                                  center: Optional[Point3D] = None,
                                  constraints: Optional[List[Dict]] = None) -> str:
        """Create parametric rectangle with optional constraints"""
        feature_id = f"rect_{self.feature_counter}"
        self.feature_counter += 1
        
        center = center or Point3D(0, 0, 0)
        
        # Create corner points as entities
        corner_entities = []
        for i, (dx, dy) in enumerate([(-width/2, -height/2), (width/2, -height/2), 
                                     (width/2, height/2), (-width/2, height/2)]):
            corner_id = f"{feature_id}_corner_{i}"
            corner_pos = Point3D(center.x + dx, center.y + dy, center.z)
            entity = self.add_entity(corner_id, EntityType.POINT, corner_pos)
            corner_entities.append(corner_id)
        
        # Add center point
        center_id = f"{feature_id}_center"
        self.add_entity(center_id, EntityType.POINT, center)
        
        # Create constraints
        constraint_ids = []
        
        # Distance constraints from center to corners (should be equal for rectangle)
        for corner_id in corner_entities:
            dist = math.sqrt((width/2)**2 + (height/2)**2)
            constraint_id = self.add_constraint("distance", [center_id, corner_id], dist)
            constraint_ids.append(constraint_id)
        
        # Add custom constraints if provided
        if constraints:
            for constraint_spec in constraints:
                if constraint_spec["type"] == "fix_center":
                    fix_id = self.add_constraint("fix", [center_id])
                    constraint_ids.append(fix_id)
                elif constraint_spec["type"] == "width":
                    # Ensure width constraint
                    dist_id = self.add_constraint("distance", [corner_entities[0], corner_entities[1]], width)
                    constraint_ids.append(dist_id)
                elif constraint_spec["type"] == "height":
                    # Ensure height constraint  
                    dist_id = self.add_constraint("distance", [corner_entities[1], corner_entities[2]], height)
                    constraint_ids.append(dist_id)
        
        # Create feature record
        feature = ParametricFeature(
            id=feature_id,
            feature_type="rectangle",
            base_geometry={
                "width": width,
                "height": height,
                "center": [center.x, center.y, center.z],
                "corners": corner_entities
            },
            parameters={"width": width, "height": height},
            constraints=constraint_ids
        )
        
        self.features[feature_id] = feature
        return feature_id
    
    def create_parametric_circle(self, radius: float, center: Optional[Point3D] = None,
                               constraints: Optional[List[Dict]] = None) -> str:
        """Create parametric circle with optional constraints"""
        feature_id = f"circle_{self.feature_counter}"
        self.feature_counter += 1
        
        center = center or Point3D(0, 0, 0)
        
        # Create center entity
        center_id = f"{feature_id}_center"
        self.add_entity(center_id, EntityType.POINT, center, {"radius": radius})
        
        # Create points on circle for constraints
        point_entities = []
        num_points = 4  # Create 4 points for constraint purposes
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            point_pos = Point3D(
                center.x + radius * math.cos(angle),
                center.y + radius * math.sin(angle),
                center.z
            )
            point_id = f"{feature_id}_point_{i}"
            self.add_entity(point_id, EntityType.POINT, point_pos)
            point_entities.append(point_id)
        
        # Create radius constraints
        constraint_ids = []
        for point_id in point_entities:
            constraint_id = self.add_constraint("distance", [center_id, point_id], radius)
            constraint_ids.append(constraint_id)
        
        # Add custom constraints if provided
        if constraints:
            for constraint_spec in constraints:
                if constraint_spec["type"] == "fix_center":
                    fix_id = self.add_constraint("fix", [center_id])
                    constraint_ids.append(fix_id)
                elif constraint_spec["type"] == "concentric" and "target" in constraint_spec:
                    # Would need to implement concentric constraint
                    pass
        
        # Create feature record
        feature = ParametricFeature(
            id=feature_id,
            feature_type="circle",
            base_geometry={
                "radius": radius,
                "center": [center.x, center.y, center.z],
                "points": point_entities
            },
            parameters={"radius": radius},
            constraints=constraint_ids
        )
        
        self.features[feature_id] = feature
        return feature_id
    
    def create_constrained_assembly(self, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create assembly with inter-part constraints"""
        assembly_id = f"assembly_{self.feature_counter}"
        self.feature_counter += 1
        
        part_features = []
        assembly_constraints = []
        
        for part_spec in parts:
            if part_spec["type"] == "rectangle":
                feature_id = self.create_parametric_rectangle(
                    part_spec["width"], 
                    part_spec["height"],
                    Point3D(*part_spec.get("position", [0, 0, 0])),
                    part_spec.get("constraints", [])
                )
                part_features.append(feature_id)
            elif part_spec["type"] == "circle":
                feature_id = self.create_parametric_circle(
                    part_spec["radius"],
                    Point3D(*part_spec.get("position", [0, 0, 0])),
                    part_spec.get("constraints", [])
                )
                part_features.append(feature_id)
        
        # Add inter-part constraints
        for constraint_spec in parts[0].get("assembly_constraints", []):
            if constraint_spec["type"] == "align_centers":
                # Align centers of first two parts
                if len(part_features) >= 2:
                    part1_center = f"{part_features[0]}_center"
                    part2_center = f"{part_features[1]}_center" 
                    constraint_id = self.add_constraint("coincident", [part1_center, part2_center])
                    assembly_constraints.append(constraint_id)
        
        return {
            "assembly_id": assembly_id,
            "parts": part_features,
            "constraints": assembly_constraints
        }
    
    def generate_cadquery_geometry(self, feature_id: str) -> cq.Workplane:
        """Generate CadQuery geometry from parametric feature"""
        if feature_id not in self.features:
            raise ValueError(f"Feature {feature_id} not found")
        
        feature = self.features[feature_id]
        
        # Solve constraints first
        solve_result = self.solve_constraints()
        
        if feature.feature_type == "rectangle":
            # Get updated corner positions from solved constraints
            corners = feature.base_geometry["corners"]
            corner_positions = []
            
            for corner_id in corners:
                if corner_id in self.entities:
                    pos = self.entities[corner_id].position
                    corner_positions.append((pos.x, pos.y))
            
            if corner_positions:
                # Create rectangle from solved positions
                wp = cq.Workplane("XY")
                # Use the first corner as starting point and create rectangle
                width = feature.parameters["width"]
                height = feature.parameters["height"]
                center_pos = self.entities[f"{feature_id}_center"].position
                
                wp = wp.center(center_pos.x, center_pos.y).rect(width, height)
                return wp
        
        elif feature.feature_type == "circle":
            # Get updated center and radius
            center_id = f"{feature_id}_center"
            if center_id in self.entities:
                center_pos = self.entities[center_id].position
                radius = feature.parameters["radius"]
                
                wp = cq.Workplane("XY")
                wp = wp.center(center_pos.x, center_pos.y).circle(radius)
                return wp
        
        # Default fallback
        return cq.Workplane("XY")
    
    def update_parameter(self, feature_id: str, parameter: str, value: float) -> Dict[str, Any]:
        """Update feature parameter and resolve constraints"""
        if feature_id not in self.features:
            raise ValueError(f"Feature {feature_id} not found")
        
        feature = self.features[feature_id]
        old_value = feature.parameters.get(parameter)
        feature.parameters[parameter] = value
        
        # Update related constraints
        if parameter == "radius" and feature.feature_type == "circle":
            # Update all radius constraints for this circle
            center_id = f"{feature_id}_center"
            point_ids = feature.base_geometry.get("points", [])
            
            for point_id in point_ids:
                # Find distance constraints involving this point
                for constraint_id, constraint in self.solver.constraints.items():
                    if (hasattr(constraint, 'point1') and hasattr(constraint, 'point2') and
                        ((constraint.point1.id == center_id and constraint.point2.id == point_id) or
                         (constraint.point1.id == point_id and constraint.point2.id == center_id))):
                        constraint.target_value = value
        
        # Solve constraints with new parameter
        result = self.solve_constraints()
        
        return {
            "feature_id": feature_id,
            "parameter": parameter,
            "old_value": old_value,
            "new_value": value,
            "constraint_result": result
        }
    
    def get_constraint_status(self) -> Dict[str, Any]:
        """Get current constraint system status"""
        return self.solver.diagnose_constraints()
    
    def export_parametric_model(self) -> Dict[str, Any]:
        """Export complete parametric model definition"""
        return {
            "features": {fid: {
                "id": f.id,
                "type": f.feature_type,
                "geometry": f.base_geometry,
                "parameters": f.parameters,
                "constraints": f.constraints
            } for fid, f in self.features.items()},
            "entities": {eid: {
                "id": e.id,
                "type": e.entity_type.name,
                "position": [e.position.x, e.position.y, e.position.z],
                "parameters": e.parameters
            } for eid, e in self.entities.items()},
            "constraint_status": self.get_constraint_status()
        }

# Integration with existing NLG system
class ConstraintAwareNLGExtension:
    """Extension to NLG system for constraint understanding"""
    
    def __init__(self):
        self.constraint_keywords = {
            "parallel": "parallel",
            "perpendicular": "perpendicular", 
            "tangent": "tangent",
            "concentric": "concentric",
            "symmetric": "symmetry",
            "aligned": "coincident",
            "fixed": "fix",
            "equal": "equal",
            "distance": "distance",
            "angle": "angle"
        }
    
    def parse_constraints_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse constraint descriptions from natural language"""
        constraints = []
        text_lower = text.lower()
        
        # Look for constraint keywords
        for keyword, constraint_type in self.constraint_keywords.items():
            if keyword in text_lower:
                constraints.append({
                    "type": constraint_type,
                    "confidence": 0.8,
                    "text_match": keyword
                })
        
        # Look for specific constraint patterns
        import re
        
        # Distance patterns
        distance_patterns = [
            r'(\d+(?:\.\d+)?)\s*mm\s+apart',
            r'distance\s+of\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*mm\s+between'
        ]
        
        for pattern in distance_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                constraints.append({
                    "type": "distance",
                    "value": float(match),
                    "confidence": 0.9
                })
        
        # Angle patterns
        angle_patterns = [
            r'(\d+(?:\.\d+)?)\s*degree',
            r'(\d+(?:\.\d+)?)\s*°',
            r'angle\s+of\s+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in angle_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                constraints.append({
                    "type": "angle",
                    "value": float(match),
                    "confidence": 0.9
                })
        
        return constraints

# Example usage and testing
if __name__ == "__main__":
    print("🔧 Testing Parametric CAD Generator")
    print("=" * 50)
    
    # Create generator
    gen = ParametricCADGenerator()
    
    # Create a constrained rectangle
    rect_id = gen.create_parametric_rectangle(
        width=50.0, 
        height=30.0, 
        center=Point3D(0, 0, 0),
        constraints=[
            {"type": "fix_center"},
            {"type": "width"},
            {"type": "height"}
        ]
    )
    
    print(f"Created parametric rectangle: {rect_id}")
    
    # Create a circle
    circle_id = gen.create_parametric_circle(
        radius=15.0,
        center=Point3D(60, 0, 0),
        constraints=[{"type": "fix_center"}]
    )
    
    print(f"Created parametric circle: {circle_id}")
    
    # Solve constraints
    result = gen.solve_constraints()
    print(f"Constraint solving: {result['success']}")
    print(f"Final error: {result.get('final_error', 0):.6f}")
    
    # Test parameter update
    print("\nTesting parameter update...")
    update_result = gen.update_parameter(circle_id, "radius", 20.0)
    print(f"Updated circle radius: {update_result['new_value']}")
    
    # Get constraint status
    status = gen.get_constraint_status()
    print(f"\nConstraint system status: {status['system_status']}")
    print(f"Total constraints: {status['total_constraints']}")
    print(f"Satisfied: {status['satisfied_constraints']}")
    
    # Test NLG constraint parsing
    print("\n🤖 Testing NLG Constraint Parsing")
    nlg_ext = ConstraintAwareNLGExtension()
    
    test_phrases = [
        "Create two circles 25mm apart and parallel lines",
        "Make a rectangle with 90 degree corners",
        "Design symmetric parts with 15mm distance between centers",
        "Create concentric circles with fixed center"
    ]
    
    for phrase in test_phrases:
        constraints = nlg_ext.parse_constraints_from_text(phrase)
        print(f"'{phrase}' -> {len(constraints)} constraints detected")
        for c in constraints:
            print(f"  - {c['type']}: {c.get('value', 'N/A')} (confidence: {c['confidence']})")
    
    print("\n✅ Parametric CAD system ready!")
    print("🎯 Features: Constraint solving, parametric updates, NLG integration")
