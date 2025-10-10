#!/usr/bin/env python3
"""
Constraint Visualization System
===============================

Visual feedback system for geometric constraints in 3D CAD environment.
Provides intuitive visual representation of constraint relationships.

Features:
- Constraint symbols and indicators
- Color-coded constraint status
- Interactive constraint manipulation
- Real-time constraint feedback
- Manufacturing constraint visualization

Author: KelmoidAI Genesis Team
License: MIT
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import math

from constraint_solver import (
    ConstraintSolver, ConstraintType, GeometricConstraint,
    Point3D, Vector3D, EntityType
)
from parametric_cad import ParametricCADGenerator

class ConstraintVisualizer:
    """Visualizes geometric constraints in 3D space"""
    
    def __init__(self):
        self.constraint_colors = {
            ConstraintType.DISTANCE: 'blue',
            ConstraintType.ANGLE: 'orange',
            ConstraintType.PARALLEL: 'green',
            ConstraintType.PERPENDICULAR: 'red',
            ConstraintType.TANGENT: 'purple',
            ConstraintType.SYMMETRY: 'cyan',
            ConstraintType.COINCIDENT: 'magenta',
            ConstraintType.CONCENTRIC: 'yellow',
            ConstraintType.HORIZONTAL: 'lime',
            ConstraintType.VERTICAL: 'pink',
            ConstraintType.FIX: 'gray',
            ConstraintType.EQUAL: 'brown'
        }
        
        self.status_colors = {
            'satisfied': 'green',
            'violated': 'red',
            'warning': 'orange'
        }
    
    def create_constraint_symbols(self, solver: ConstraintSolver) -> List[go.Scatter3d]:
        """Create 3D constraint symbols for visualization"""
        traces = []
        
        for constraint_id, constraint in solver.constraints.items():
            if constraint.is_satisfied():
                color = self.status_colors['satisfied']
                opacity = 0.7
            else:
                color = self.status_colors['violated']
                opacity = 0.9
            
            # Get constraint visualization based on type
            if constraint.type == ConstraintType.DISTANCE:
                trace = self._create_distance_symbol(constraint, color, opacity)
            elif constraint.type == ConstraintType.ANGLE:
                trace = self._create_angle_symbol(constraint, color, opacity)
            elif constraint.type == ConstraintType.PARALLEL:
                trace = self._create_parallel_symbol(constraint, color, opacity)
            elif constraint.type == ConstraintType.PERPENDICULAR:
                trace = self._create_perpendicular_symbol(constraint, color, opacity)
            elif constraint.type == ConstraintType.FIX:
                trace = self._create_fix_symbol(constraint, color, opacity)
            elif constraint.type == ConstraintType.COINCIDENT:
                trace = self._create_coincident_symbol(constraint, color, opacity)
            else:
                trace = self._create_generic_symbol(constraint, color, opacity)
            
            if trace:
                trace.name = f"{constraint.type.name}: {constraint_id}"
                traces.append(trace)
        
        return traces
    
    def _create_distance_symbol(self, constraint, color: str, opacity: float) -> Optional[go.Scatter3d]:
        """Create distance constraint visualization"""
        if not hasattr(constraint, 'point1') or not hasattr(constraint, 'point2'):
            return None
        
        p1 = constraint.point1.position
        p2 = constraint.point2.position
        
        # Create line between points
        x_coords = [p1.x, p2.x]
        y_coords = [p1.y, p2.y]
        z_coords = [p1.z, p2.z]
        
        # Add dimension arrows at midpoint
        mid_x = (p1.x + p2.x) / 2
        mid_y = (p1.y + p2.y) / 2
        mid_z = (p1.z + p2.z) / 2
        
        return go.Scatter3d(
            x=x_coords + [mid_x],
            y=y_coords + [mid_y],
            z=z_coords + [mid_z],
            mode='lines+markers+text',
            line=dict(color=color, width=4, dash='dash'),
            marker=dict(size=8, color=color, opacity=opacity),
            text=['', '', f'{constraint.target_value:.1f}mm'],
            textposition='middle center',
            opacity=opacity
        )
    
    def _create_angle_symbol(self, constraint, color: str, opacity: float) -> Optional[go.Scatter3d]:
        """Create angle constraint visualization"""
        if not hasattr(constraint, 'line1') or not hasattr(constraint, 'line2'):
            return None
        
        # Create arc to show angle
        line1_dir = constraint.line1.get_direction().to_array()
        line2_dir = constraint.line2.get_direction().to_array()
        
        # Find intersection point (simplified - assume lines intersect at origin)
        intersection = np.array([0, 0, 0])
        
        # Create arc points
        arc_radius = 20.0  # Visual arc radius
        num_points = 20
        angle_rad = constraint.target_value
        
        arc_x, arc_y, arc_z = [], [], []
        for i in range(num_points):
            t = i / (num_points - 1) * angle_rad
            # Simplified 2D arc in XY plane
            arc_x.append(intersection[0] + arc_radius * math.cos(t))
            arc_y.append(intersection[1] + arc_radius * math.sin(t))
            arc_z.append(intersection[2])
        
        return go.Scatter3d(
            x=arc_x,
            y=arc_y,
            z=arc_z,
            mode='lines+text',
            line=dict(color=color, width=3),
            text=[f'{math.degrees(angle_rad):.1f}°' if i == len(arc_x)//2 else '' for i in range(len(arc_x))],
            textposition='middle center',
            opacity=opacity
        )
    
    def _create_parallel_symbol(self, constraint, color: str, opacity: float) -> Optional[go.Scatter3d]:
        """Create parallel constraint visualization"""
        # Create parallel arrow symbols
        x_coords = [0, 5, 10, 15]
        y_coords = [0, 0, 5, 5]
        z_coords = [0, 0, 0, 0]
        
        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=6, symbol='arrow', color=color),
            opacity=opacity
        )
    
    def _create_perpendicular_symbol(self, constraint, color: str, opacity: float) -> Optional[go.Scatter3d]:
        """Create perpendicular constraint visualization"""
        # Create right angle symbol
        size = 5
        x_coords = [0, size, size, 0]
        y_coords = [0, 0, size, size]
        z_coords = [0, 0, 0, 0]
        
        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(color=color, width=4),
            opacity=opacity
        )
    
    def _create_fix_symbol(self, constraint, color: str, opacity: float) -> Optional[go.Scatter3d]:
        """Create fix constraint visualization"""
        if not hasattr(constraint, 'entity'):
            return None
        
        pos = constraint.entity.position
        
        # Create ground/fix symbol (triangle)
        size = 3
        x_coords = [pos.x - size, pos.x + size, pos.x, pos.x - size]
        y_coords = [pos.y - size, pos.y - size, pos.y + size, pos.y - size]
        z_coords = [pos.z, pos.z, pos.z, pos.z]
        
        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=8, symbol='square', color=color),
            opacity=opacity
        )
    
    def _create_coincident_symbol(self, constraint, color: str, opacity: float) -> Optional[go.Scatter3d]:
        """Create coincident constraint visualization"""
        if not hasattr(constraint, 'point1') or not hasattr(constraint, 'point2'):
            return None
        
        p1 = constraint.point1.position
        p2 = constraint.point2.position
        
        # Create overlapping circles
        return go.Scatter3d(
            x=[p1.x, p2.x],
            y=[p1.y, p2.y],
            z=[p1.z, p2.z],
            mode='markers',
            marker=dict(size=12, symbol='circle', color=color, opacity=opacity, 
                       line=dict(width=3, color='black')),
        )
    
    def _create_generic_symbol(self, constraint, color: str, opacity: float) -> Optional[go.Scatter3d]:
        """Create generic constraint symbol"""
        # Default constraint indicator
        return go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers+text',
            marker=dict(size=10, color=color, opacity=opacity),
            text=[constraint.type.name],
            textposition='top center'
        )
    
    def create_entity_visualization(self, solver: ConstraintSolver) -> List[go.Scatter3d]:
        """Create visualization for geometric entities"""
        traces = []
        
        for entity_id, entity in solver.entities.items():
            pos = entity.position
            
            if entity.entity_type == EntityType.POINT:
                trace = go.Scatter3d(
                    x=[pos.x],
                    y=[pos.y], 
                    z=[pos.z],
                    mode='markers+text',
                    marker=dict(size=8, color='black'),
                    text=[entity_id],
                    textposition='top center',
                    name=f"Point: {entity_id}"
                )
                traces.append(trace)
            
            elif entity.entity_type == EntityType.LINE:
                # For lines, we'd need start and end points
                # This is a simplified representation
                trace = go.Scatter3d(
                    x=[pos.x, pos.x + 10],
                    y=[pos.y, pos.y + 10],
                    z=[pos.z, pos.z],
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6),
                    name=f"Line: {entity_id}"
                )
                traces.append(trace)
            
            elif entity.entity_type == EntityType.CIRCLE:
                # Create circle representation
                radius = entity.get_parameter('radius', 5.0)
                theta = np.linspace(0, 2*np.pi, 50)
                
                x_circle = pos.x + radius * np.cos(theta)
                y_circle = pos.y + radius * np.sin(theta)
                z_circle = np.full_like(theta, pos.z)
                
                trace = go.Scatter3d(
                    x=x_circle,
                    y=y_circle,
                    z=z_circle,
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=3),
                    name=f"Circle: {entity_id}"
                )
                traces.append(trace)
        
        return traces
    
    def create_constraint_status_panel(self, solver: ConstraintSolver) -> Dict[str, Any]:
        """Create constraint status information panel"""
        diagnosis = solver.diagnose_constraints()
        satisfied = solver.get_satisfied_constraints()
        violated = solver.get_violated_constraints()
        
        status_info = {
            "system_status": diagnosis['system_status'],
            "total_constraints": diagnosis['total_constraints'],
            "total_entities": diagnosis['total_entities'],
            "degrees_of_freedom": diagnosis['degrees_of_freedom'],
            "satisfied_count": len(satisfied),
            "violated_count": len(violated),
            "satisfied_constraints": satisfied,
            "violated_constraints": [{"id": v[0], "violation": f"{v[1]:.6f}"} for v in violated]
        }
        
        return status_info
    
    def create_interactive_plot(self, cad_generator: ParametricCADGenerator, 
                              width: int = 800, height: int = 600) -> go.Figure:
        """Create interactive 3D plot with constraints and entities"""
        fig = go.Figure()
        
        # Add entity visualizations
        entity_traces = self.create_entity_visualization(cad_generator.solver)
        for trace in entity_traces:
            fig.add_trace(trace)
        
        # Add constraint visualizations
        constraint_traces = self.create_constraint_symbols(cad_generator.solver)
        for trace in constraint_traces:
            fig.add_trace(trace)
        
        # Configure layout
        fig.update_layout(
            title="Parametric CAD with Constraints",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            width=width,
            height=height,
            showlegend=True
        )
        
        return fig
    
    def create_constraint_legend(self) -> Dict[str, str]:
        """Create legend for constraint symbols"""
        legend = {
            "Distance": "Dashed line with dimension label",
            "Angle": "Arc with angle measurement",
            "Parallel": "Parallel arrows",
            "Perpendicular": "Right angle symbol",
            "Fix": "Triangle ground symbol",
            "Coincident": "Overlapping circles",
            "Tangent": "Tangent line indicator",
            "Symmetry": "Mirror line",
            "Equal": "Equal symbol (=)",
            "Status Colors": {
                "Green": "Constraint satisfied",
                "Red": "Constraint violated",
                "Orange": "Warning/near violation"
            }
        }
        return legend

def create_manufacturing_constraint_visualizer():
    """Create specialized visualizer for manufacturing constraints"""
    
    class ManufacturingConstraintVisualizer(ConstraintVisualizer):
        """Specialized visualizer for manufacturing constraints"""
        
        def create_bolt_pattern_visualization(self, center_pos: Point3D, 
                                           bolt_positions: List[Point3D],
                                           diameter: float) -> List[go.Scatter3d]:
            """Visualize bolt circle pattern"""
            traces = []
            
            # Create bolt circle
            theta = np.linspace(0, 2*np.pi, 100)
            radius = diameter / 2
            x_circle = center_pos.x + radius * np.cos(theta)
            y_circle = center_pos.y + radius * np.sin(theta)
            z_circle = np.full_like(theta, center_pos.z)
            
            circle_trace = go.Scatter3d(
                x=x_circle,
                y=y_circle,
                z=z_circle,
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                name="Bolt Circle"
            )
            traces.append(circle_trace)
            
            # Create bolt hole indicators
            bolt_x = [pos.x for pos in bolt_positions]
            bolt_y = [pos.y for pos in bolt_positions] 
            bolt_z = [pos.z for pos in bolt_positions]
            
            bolt_trace = go.Scatter3d(
                x=bolt_x,
                y=bolt_y,
                z=bolt_z,
                mode='markers',
                marker=dict(size=10, color='red', symbol='circle-open', 
                          line=dict(width=3, color='red')),
                name="Bolt Holes"
            )
            traces.append(bolt_trace)
            
            return traces
        
        def create_tolerance_visualization(self, nominal_dimension: float,
                                        tolerance_plus: float,
                                        tolerance_minus: float) -> Dict[str, Any]:
            """Create tolerance band visualization"""
            return {
                "nominal": nominal_dimension,
                "upper_limit": nominal_dimension + tolerance_plus,
                "lower_limit": nominal_dimension - tolerance_minus,
                "tolerance_grade": f"+{tolerance_plus:.3f}/-{tolerance_minus:.3f}",
                "color_coding": {
                    "nominal": "blue",
                    "tolerance_band": "lightblue", 
                    "critical": "red"
                }
            }
    
    return ManufacturingConstraintVisualizer()

# Testing and example usage
if __name__ == "__main__":
    print("🎨 Testing Constraint Visualization System")
    print("=" * 50)
    
    # Create parametric CAD generator with constraints
    generator = ParametricCADGenerator()
    
    # Create some constrained geometry
    rect_id = generator.create_parametric_rectangle(
        width=50.0, 
        height=30.0,
        constraints=[{"type": "fix_center"}]
    )
    
    circle_id = generator.create_parametric_circle(
        radius=15.0,
        center=Point3D(60, 0, 0),
        constraints=[{"type": "fix_center"}]
    )
    
    # Solve constraints
    result = generator.solve_constraints()
    print(f"Constraints solved: {result['success']}")
    
    # Create visualizer
    visualizer = ConstraintVisualizer()
    
    # Create status panel
    status = visualizer.create_constraint_status_panel(generator.solver)
    print(f"System status: {status['system_status']}")
    print(f"Total constraints: {status['total_constraints']}")
    print(f"Satisfied: {status['satisfied_count']}")
    print(f"Violated: {status['violated_count']}")
    
    # Create interactive plot
    fig = visualizer.create_interactive_plot(generator)
    print(f"Created interactive plot with {len(fig.data)} traces")
    
    # Test manufacturing visualizer
    print("\n🏭 Testing Manufacturing Constraint Visualizer")
    mfg_viz = create_manufacturing_constraint_visualizer()
    
    # Create bolt pattern visualization
    center = Point3D(0, 0, 0)
    bolt_holes = [
        Point3D(20, 0, 0),
        Point3D(0, 20, 0),
        Point3D(-20, 0, 0),
        Point3D(0, -20, 0)
    ]
    
    bolt_traces = mfg_viz.create_bolt_pattern_visualization(center, bolt_holes, 40.0)
    print(f"Created {len(bolt_traces)} bolt pattern traces")
    
    # Create tolerance visualization
    tolerance_info = mfg_viz.create_tolerance_visualization(50.0, 0.1, 0.05)
    print(f"Tolerance visualization: {tolerance_info['tolerance_grade']}")
    
    print("\n✅ Constraint visualization system ready!")
    print("🎯 Features: 3D constraint symbols, status indicators, manufacturing patterns")
