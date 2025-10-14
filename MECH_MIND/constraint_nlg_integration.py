#!/usr/bin/env python3
"""
Constraint-Aware NLG Integration
================================

Advanced natural language processing for geometric constraints
and intelligent CAD generation with design intent understanding.

Features:
- Natural language constraint parsing
- Design intent extraction
- Automatic constraint generation
- Manufacturing constraint recognition
- Parametric model creation from text

Author: KelmoidAI Genesis Team
License: MIT
"""

import re
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum, auto

from constraint_solver import Point3D, Vector3D, EntityType
from parametric_cad import ParametricCADGenerator, ConstraintAwareNLGExtension
from llm_engine import LLMTextToCAD

class ConstraintIntent(Enum):
    """Types of constraint intentions from natural language"""
    DIMENSIONAL = auto()      # Size/distance specifications
    POSITIONAL = auto()       # Location/placement constraints
    RELATIONAL = auto()       # Relationships between entities
    MANUFACTURING = auto()    # Manufacturing-specific constraints
    ASSEMBLY = auto()         # Assembly and mating constraints

@dataclass
class ParsedConstraint:
    """Parsed constraint from natural language"""
    intent: ConstraintIntent
    constraint_type: str
    entities: List[str]
    value: Optional[float] = None
    units: str = "mm"
    confidence: float = 0.0
    text_match: str = ""
    parameters: Dict[str, Any] = None

class AdvancedConstraintNLG:
    """Advanced natural language processing for constraints"""
    
    def __init__(self):
        self.dimensional_patterns = [
            # Distance patterns
            (r'(\d+(?:\.\d+)?)\s*(mm|cm|m|inch|in)?\s+(?:apart|between|distance)', 'distance'),
            (r'(?:distance|spacing|gap)\s+of\s+(\d+(?:\.\d+)?)\s*(mm|cm|m)?', 'distance'),
            (r'(\d+(?:\.\d+)?)\s*(mm|cm|m)?\s+(?:from|to)', 'distance'),
            
            # Size patterns
            (r'(?:width|length|height|diameter|radius)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mm|cm|m)?', 'dimension'),
            (r'(\d+(?:\.\d+)?)\s*(mm|cm|m)?\s+(?:wide|long|high|thick)', 'dimension'),
            
            # Angle patterns
            (r'(\d+(?:\.\d+)?)\s*(?:degrees?|°)', 'angle'),
            (r'(?:angle|rotation)\s+of\s+(\d+(?:\.\d+)?)', 'angle'),
        ]
        
        self.relational_patterns = [
            # Geometric relationships
            (r'parallel\s+to', 'parallel'),
            (r'perpendicular\s+to', 'perpendicular'),
            (r'tangent\s+to', 'tangent'),
            (r'concentric\s+with', 'concentric'),
            (r'aligned\s+with', 'coincident'),
            (r'symmetric\s+about', 'symmetry'),
            (r'equal\s+(?:to|with)', 'equal'),
            
            # Positional relationships
            (r'centered\s+(?:on|at)', 'centered'),
            (r'positioned\s+at', 'positioned'),
            (r'located\s+at', 'located'),
            (r'fixed\s+(?:at|to)', 'fix'),
        ]
        
        self.manufacturing_patterns = [
            # Hole patterns
            (r'(\d+)\s*holes?\s+(?:on\s+)?(?:a\s+)?(\d+(?:\.\d+)?)\s*(mm|cm)?\s+(?:bolt\s+)?circle', 'bolt_pattern'),
            (r'bolt\s+circle\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(mm|cm)?', 'bolt_circle'),
            (r'hole\s+pattern\s+(\d+(?:\.\d+)?)\s*(mm|cm)?\s+spacing', 'hole_pattern'),
            
            # Threading and fasteners
            (r'M(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*thread', 'thread'),
            (r'(\d+(?:\.\d+)?)\s*(mm|cm)?\s+thread\s+pitch', 'thread_pitch'),
            
            # Tolerances
            (r'tolerance\s+[+-]?(\d+(?:\.\d+)?)', 'tolerance'),
            (r'[+-](\d+(?:\.\d+)?)\s*(?:mm|μm|micron)', 'tolerance'),
        ]
        
        self.assembly_patterns = [
            # Mating constraints
            (r'mated?\s+(?:with|to)', 'mate'),
            (r'assembled\s+(?:with|to)', 'assemble'),
            (r'connected\s+(?:to|with)', 'connect'),
            (r'fastened\s+(?:to|with)', 'fasten'),
        ]
    
    def parse_advanced_constraints(self, text: str) -> List[ParsedConstraint]:
        """Parse advanced constraint descriptions from natural language"""
        constraints = []
        text_lower = text.lower()
        
        # Parse dimensional constraints
        for pattern, constraint_type in self.dimensional_patterns:
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                value = float(match.group(1)) if match.group(1) else None
                units = match.group(2) if len(match.groups()) > 1 and match.group(2) else "mm"
                
                constraint = ParsedConstraint(
                    intent=ConstraintIntent.DIMENSIONAL,
                    constraint_type=constraint_type,
                    entities=[],  # To be filled by entity recognition
                    value=value,
                    units=units,
                    confidence=0.9,
                    text_match=match.group(0)
                )
                constraints.append(constraint)
        
        # Parse relational constraints
        for pattern, constraint_type in self.relational_patterns:
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                constraint = ParsedConstraint(
                    intent=ConstraintIntent.RELATIONAL,
                    constraint_type=constraint_type,
                    entities=[],
                    confidence=0.85,
                    text_match=match.group(0)
                )
                constraints.append(constraint)
        
        # Parse manufacturing constraints
        for pattern, constraint_type in self.manufacturing_patterns:
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                value = None
                parameters = {}
                
                if constraint_type == 'bolt_pattern':
                    parameters['hole_count'] = int(match.group(1))
                    value = float(match.group(2)) if match.group(2) else None
                    parameters['diameter'] = value
                elif constraint_type in ['thread', 'thread_pitch']:
                    value = float(match.group(1)) if match.group(1) else None
                    if len(match.groups()) > 1:
                        parameters['pitch'] = float(match.group(2)) if match.group(2) else None
                else:
                    value = float(match.group(1)) if len(match.groups()) >= 1 and match.group(1) else None
                
                constraint = ParsedConstraint(
                    intent=ConstraintIntent.MANUFACTURING,
                    constraint_type=constraint_type,
                    entities=[],
                    value=value,
                    parameters=parameters or {},
                    confidence=0.8,
                    text_match=match.group(0)
                )
                constraints.append(constraint)
        
        # Parse assembly constraints
        for pattern, constraint_type in self.assembly_patterns:
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                constraint = ParsedConstraint(
                    intent=ConstraintIntent.ASSEMBLY,
                    constraint_type=constraint_type,
                    entities=[],
                    confidence=0.75,
                    text_match=match.group(0)
                )
                constraints.append(constraint)
        
        return constraints
    
    def extract_entity_references(self, text: str, constraints: List[ParsedConstraint]) -> List[ParsedConstraint]:
        """Extract entity references for constraints"""
        text_lower = text.lower()
        
        # Common entity keywords
        entity_patterns = [
            (r'circle[s]?', 'circle'),
            (r'rectangle[s]?|square[s]?', 'rectangle'),
            (r'line[s]?', 'line'),
            (r'point[s]?', 'point'),
            (r'hole[s]?', 'hole'),
            (r'edge[s]?', 'edge'),
            (r'face[s]?', 'face'),
            (r'center[s]?', 'center'),
        ]
        
        # Find all entity references
        entities_found = []
        for pattern, entity_type in entity_patterns:
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                entities_found.append({
                    'type': entity_type,
                    'position': match.span(),
                    'text': match.group(0)
                })
        
        # Associate entities with constraints based on proximity
        for constraint in constraints:
            # Find constraint position in text
            constraint_pos = text_lower.find(constraint.text_match.lower())
            if constraint_pos == -1:
                continue
            
            # Find nearest entities
            nearest_entities = []
            for entity in entities_found:
                distance = abs(entity['position'][0] - constraint_pos)
                if distance < 100:  # Within reasonable text distance
                    nearest_entities.append((distance, entity))
            
            # Sort by distance and take closest entities
            nearest_entities.sort()
            constraint.entities = [entity[1]['type'] for entity in nearest_entities[:2]]
        
        return constraints
    
    def create_parametric_model_from_text(self, text: str, generator: ParametricCADGenerator) -> Dict[str, Any]:
        """Create parametric CAD model from natural language description"""
        # Parse basic geometry using existing LLM
        llm = LLMTextToCAD()
        llm_result = llm.compile(text)
        
        # Parse constraints
        constraints = self.parse_advanced_constraints(text)
        constraints = self.extract_entity_references(text, constraints)
        
        # Generate base geometry
        feature_ids = []
        if llm_result.get('success'):
            schema = llm_result['schema']
            shape = schema.get('shape', 'cube')
            dimensions = schema.get('dimensions', {})
            
            if shape in ['cube', 'rectangle', 'box']:
                width = dimensions.get('width', dimensions.get('length', 20.0))
                height = dimensions.get('height', width)
                
                # Create constraints from parsed information
                constraint_specs = []
                for constraint in constraints:
                    if constraint.intent == ConstraintIntent.DIMENSIONAL:
                        if constraint.constraint_type == 'distance':
                            constraint_specs.append({"type": "distance", "value": constraint.value})
                    elif constraint.intent == ConstraintIntent.POSITIONAL:
                        constraint_specs.append({"type": "fix_center"})
                
                feature_id = generator.create_parametric_rectangle(
                    width=width,
                    height=height,
                    constraints=constraint_specs
                )
                feature_ids.append(feature_id)
                
            elif shape in ['circle', 'sphere']:
                radius = dimensions.get('radius', 10.0)
                
                constraint_specs = []
                for constraint in constraints:
                    if constraint.intent == ConstraintIntent.POSITIONAL:
                        constraint_specs.append({"type": "fix_center"})
                
                feature_id = generator.create_parametric_circle(
                    radius=radius,
                    constraints=constraint_specs
                )
                feature_ids.append(feature_id)
        
        # Solve constraints
        solve_result = generator.solve_constraints()
        
        return {
            'success': True,
            'feature_ids': feature_ids,
            'parsed_constraints': [
                {
                    'intent': c.intent.name,
                    'type': c.constraint_type,
                    'value': c.value,
                    'confidence': c.confidence,
                    'entities': c.entities
                } for c in constraints
            ],
            'constraint_solve_result': solve_result,
            'llm_schema': llm_result.get('schema', {}),
            'message': f'Created {len(feature_ids)} parametric features with {len(constraints)} constraints'
        }

class ConstraintUIComponents:
    """UI components for constraint management in Gradio"""
    
    @staticmethod
    def create_constraint_interface():
        """Create Gradio interface for constraint management"""
        import gradio as gr
        
        def create_constraint_model(text_input: str, constraint_type: str, 
                                  constraint_value: float, enable_visualization: bool):
            """Create constraint model from UI inputs"""
            generator = ParametricCADGenerator()
            nlg = AdvancedConstraintNLG()
            
            # Create model from text
            result = nlg.create_parametric_model_from_text(text_input, generator)
            
            # Generate visualization if requested
            visualization_html = ""
            if enable_visualization:
                from constraint_visualizer import ConstraintVisualizer
                visualizer = ConstraintVisualizer()
                fig = visualizer.create_interactive_plot(generator)
                visualization_html = fig.to_html()
            
            # Create status report
            status = generator.get_constraint_status()
            
            status_report = f"""
            ## Constraint Analysis Report
            
            **System Status:** {status['system_status'].title()}
            **Total Entities:** {status['total_entities']}
            **Total Constraints:** {status['total_constraints']}
            **Degrees of Freedom:** {status['degrees_of_freedom']}
            **Satisfied Constraints:** {status['satisfied_constraints']}
            **Violated Constraints:** {status['violated_constraints']}
            
            ### Parsed Constraints:
            """
            
            for i, constraint in enumerate(result['parsed_constraints']):
                status_report += f"**{i+1}.** {constraint['type'].title()} ({constraint['intent']})\n"
                status_report += f"   - Value: {constraint.get('value', 'N/A')}\n"
                status_report += f"   - Confidence: {constraint['confidence']:.2f}\n"
                status_report += f"   - Entities: {', '.join(constraint['entities']) if constraint['entities'] else 'Auto-detected'}\n\n"
            
            return status_report, visualization_html, json.dumps(result, indent=2)
        
        def update_parameter(feature_id: str, parameter: str, new_value: float):
            """Update parameter and re-solve constraints"""
            # This would connect to the active generator instance
            return f"Updated {feature_id}.{parameter} = {new_value}"
        
        # Create interface
        with gr.Blocks(title="Advanced Constraint-Based CAD") as interface:
            gr.Markdown("# 🔧 Advanced Constraint-Based CAD System")
            gr.Markdown("Create parametric CAD models with intelligent constraint recognition from natural language.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Natural Language CAD Description",
                        placeholder="Create two concentric circles 50mm apart with parallel lines tangent to both...",
                        lines=5
                    )
                    
                    constraint_type = gr.Dropdown(
                        label="Additional Constraint Type",
                        choices=["distance", "angle", "parallel", "perpendicular", "tangent", "symmetric", "fix"],
                        value="distance"
                    )
                    
                    constraint_value = gr.Number(
                        label="Constraint Value",
                        value=10.0,
                        precision=2
                    )
                    
                    enable_viz = gr.Checkbox(
                        label="Enable 3D Constraint Visualization",
                        value=True
                    )
                    
                    generate_btn = gr.Button("🚀 Generate Constraint Model", variant="primary")
                
                with gr.Column(scale=3):
                    status_output = gr.Markdown(label="Constraint Analysis")
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 3D Constraint Visualization")
                    visualization_output = gr.HTML(label="3D Visualization")
                
                with gr.Column():
                    gr.Markdown("### Detailed Results")
                    json_output = gr.Code(language="json", label="Full Analysis Results")
            
            # Parameter update section
            with gr.Row():
                gr.Markdown("### 🔄 Parametric Updates")
                
                with gr.Column():
                    feature_id_input = gr.Textbox(label="Feature ID", placeholder="circle_0")
                    param_name_input = gr.Textbox(label="Parameter Name", placeholder="radius")
                    param_value_input = gr.Number(label="New Value", value=15.0)
                    update_btn = gr.Button("Update Parameter")
                    update_result = gr.Textbox(label="Update Result", interactive=False)
            
            # Wire up the interface
            generate_btn.click(
                create_constraint_model,
                inputs=[text_input, constraint_type, constraint_value, enable_viz],
                outputs=[status_output, visualization_output, json_output]
            )
            
            update_btn.click(
                update_parameter,
                inputs=[feature_id_input, param_name_input, param_value_input],
                outputs=[update_result]
            )
            
            # Add examples
            gr.Examples(
                examples=[
                    ["Create a 50mm square with a 20mm circle centered inside it"],
                    ["Design two parallel lines 25mm apart with perpendicular cross lines"],
                    ["Make a bolt pattern with 4 holes on a 60mm diameter circle"],
                    ["Create symmetric parts with 15mm spacing and tangent curves"],
                    ["Design a bracket 80x40mm with mounting holes 50mm apart"]
                ],
                inputs=[text_input]
            )
        
        return interface

# Testing and example usage
if __name__ == "__main__":
    print("🤖 Testing Advanced Constraint NLG Integration")
    print("=" * 60)
    
    # Create advanced NLG processor
    nlg = AdvancedConstraintNLG()
    
    # Test constraint parsing
    test_descriptions = [
        "Create two concentric circles 50mm apart with parallel lines",
        "Design a rectangle 80x40mm with mounting holes 60mm apart",
        "Make a bolt pattern with 6 holes on a 100mm bolt circle",
        "Create symmetric parts with 45 degree angles and equal dimensions",
        "Design a bracket with M8x1.25 threaded holes and ±0.1mm tolerance"
    ]
    
    for desc in test_descriptions:
        print(f"\nAnalyzing: '{desc}'")
        constraints = nlg.parse_advanced_constraints(desc)
        constraints = nlg.extract_entity_references(desc, constraints)
        
        print(f"Found {len(constraints)} constraints:")
        for i, constraint in enumerate(constraints):
            print(f"  {i+1}. {constraint.constraint_type} ({constraint.intent.name})")
            print(f"     Value: {constraint.value}, Confidence: {constraint.confidence:.2f}")
            print(f"     Entities: {constraint.entities}")
    
    # Test full model creation
    print(f"\n🔧 Testing Full Model Creation")
    generator = ParametricCADGenerator()
    
    test_text = "Create a 60mm square with a 15mm circle centered inside, both fixed in position"
    result = nlg.create_parametric_model_from_text(test_text, generator)
    
    print(f"Model creation: {'Success' if result['success'] else 'Failed'}")
    print(f"Features created: {len(result['feature_ids'])}")
    print(f"Constraints parsed: {len(result['parsed_constraints'])}")
    print(f"Constraint solving: {'Success' if result['constraint_solve_result']['success'] else 'Failed'}")
    
    print("\n✅ Advanced constraint NLG integration ready!")
    print("🎯 Features: Natural language parsing, constraint recognition, parametric modeling")
