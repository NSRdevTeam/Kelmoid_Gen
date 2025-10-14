"""
Fine-Tuning Dataset Generator for Natural Language Geometry Engine
================================================================

This module generates comprehensive training datasets for fine-tuning LLMs
to better understand CAD and geometry terminology, creating a unique 
"Text-to-CAD compiler" dataset for machine learning.

Author: KelmoidAI Genesis Team
"""

import json
import random
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime
import csv
import itertools

from nlg_engine import GeometryType, Unit, Material, Dimensions, GeometricPrimitive, Vector3D

# =====================================================
# DATASET SCHEMA DEFINITIONS
# =====================================================

class DatasetType(Enum):
    """Types of training datasets"""
    BASIC_GEOMETRY = "basic_geometry"
    COMPLEX_ASSEMBLIES = "complex_assemblies"
    MANUFACTURING = "manufacturing"
    ARCHITECTURAL = "architectural"
    MECHANICAL_PARTS = "mechanical_parts"
    VALIDATION = "validation"

@dataclass
class TrainingExample:
    """Single training example for fine-tuning"""
    id: str
    input_text: str
    expected_output: Dict[str, Any]
    category: DatasetType
    difficulty: str  # "easy", "medium", "hard"
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input_text,
            "output": self.expected_output,
            "category": self.category.value,
            "difficulty": self.difficulty,
            "metadata": self.metadata
        }
    
    def to_jsonl_format(self) -> str:
        """Convert to JSONL format for fine-tuning"""
        return json.dumps({
            "messages": [
                {"role": "user", "content": self.input_text},
                {"role": "assistant", "content": json.dumps(self.expected_output, indent=2)}
            ]
        })

# =====================================================
# SYNTHETIC DATA GENERATORS
# =====================================================

class GeometricPromptGenerator:
    """Generates diverse geometric prompts with variations"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.dimension_ranges = self._initialize_dimension_ranges()
        self.materials = list(Material)
        self.colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white', 'gray', 'silver']
        self.action_verbs = ['create', 'make', 'design', 'build', 'generate', 'produce', 'construct', 'form']
        
    def _initialize_templates(self) -> Dict[GeometryType, List[str]]:
        """Initialize prompt templates for each geometry type"""
        return {
            GeometryType.CUBE: [
                "{action} a {material} {color} cube with {dimensions}",
                "{action} a rectangular block {dimensions} made of {material}",
                "I need a {color} box that is {dimensions}",
                "Design a cube {dimensions} in {material}",
                "Generate a {material} block {dimensions}",
                "Make me a square {color} cube {dimensions}",
                "Create a {color} rectangular prism {dimensions}",
                "Build a {material} cube with dimensions {dimensions}"
            ],
            GeometryType.SPHERE: [
                "{action} a {material} {color} sphere with radius {radius}mm",
                "{action} a ball with diameter {diameter}mm in {material}",
                "I need a {color} sphere {radius}mm radius",
                "Design a spherical shape with r={radius}mm using {material}",
                "Generate a {material} ball {diameter}mm diameter",
                "Make me a {color} orb with radius={radius}mm",
                "Create a spherical object {radius}mm radius in {material}",
                "Build a {material} sphere dia={diameter}mm"
            ],
            GeometryType.CYLINDER: [
                "{action} a {material} {color} cylinder radius {radius}mm height {height}mm",
                "{action} a tube {diameter}mm diameter {height}mm long in {material}",
                "I need a {color} cylinder r={radius}mm h={height}mm",
                "Design a cylindrical pipe {diameter}mm x {height}mm using {material}",
                "Generate a {material} rod diameter {diameter}mm length {height}mm",
                "Make me a {color} tube radius {radius}mm height {height}mm",
                "Create a cylindrical shape {diameter}mm dia {height}mm tall in {material}",
                "Build a {material} cylinder {radius}mm radius {height}mm high"
            ],
            GeometryType.CONE: [
                "{action} a {material} {color} cone radius {radius}mm height {height}mm",
                "{action} a conical shape {diameter}mm base diameter {height}mm tall in {material}",
                "I need a {color} cone r={radius}mm h={height}mm",
                "Design a tapered cone {diameter}mm diameter {height}mm height using {material}",
                "Generate a {material} cone base {diameter}mm height {height}mm",
                "Make me a {color} conical object radius {radius}mm height {height}mm",
                "Create a cone {diameter}mm base {height}mm tall in {material}",
                "Build a {material} tapered cone {radius}mm radius {height}mm high"
            ],
            GeometryType.TORUS: [
                "{action} a {material} {color} torus major radius {radius}mm minor radius {thickness}mm",
                "{action} a ring shape {diameter}mm diameter {thickness}mm thick in {material}",
                "I need a {color} donut r={radius}mm thickness={thickness}mm",
                "Design a toroidal ring {diameter}mm outer {thickness}mm thick using {material}",
                "Generate a {material} torus {diameter}mm diameter {thickness}mm thickness",
                "Make me a {color} ring radius {radius}mm thickness {thickness}mm",
                "Create a donut shape {diameter}mm diameter {thickness}mm thick in {material}",
                "Build a {material} torus {radius}mm radius {thickness}mm thickness"
            ]
        }
    
    def _initialize_dimension_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Initialize realistic dimension ranges for each parameter"""
        return {
            'length': (1, 500),
            'width': (1, 500),
            'height': (1, 500),
            'radius': (0.5, 100),
            'diameter': (1, 200),
            'thickness': (0.1, 50),
            'depth': (1, 100)
        }
    
    def generate_prompt(self, geometry_type: GeometryType, 
                       complexity: str = "medium") -> Tuple[str, Dict[str, Any]]:
        """
        Generate a natural language prompt and expected output
        
        Args:
            geometry_type: Type of geometry to generate
            complexity: "easy", "medium", "hard"
            
        Returns:
            Tuple of (prompt_text, expected_json_output)
        """
        # Select template
        templates = self.templates.get(geometry_type, [])
        if not templates:
            return self._generate_fallback_prompt(geometry_type)
        
        template = random.choice(templates)
        
        # Generate dimensions based on geometry type
        dimensions = self._generate_dimensions(geometry_type)
        
        # Select random attributes
        action = random.choice(self.action_verbs)
        material = random.choice(self.materials)
        color = random.choice(self.colors)
        
        # Format prompt
        try:
            prompt = template.format(
                action=action,
                material=material.value,
                color=color,
                dimensions=self._format_dimensions(dimensions, geometry_type),
                radius=dimensions.radius or 10,
                diameter=dimensions.diameter or 20,
                height=dimensions.height or 10,
                thickness=dimensions.thickness or 2
            )
        except KeyError:
            # Fallback for missing format keys
            prompt = f"{action} a {material.value} {color} {geometry_type.value}"
        
        # Add complexity variations
        if complexity == "hard":
            prompt = self._add_complexity_features(prompt, geometry_type)
        elif complexity == "easy":
            prompt = self._simplify_prompt(prompt)
        
        # Generate expected output
        expected_output = self._generate_expected_output(geometry_type, dimensions, 
                                                       material, color, prompt)
        
        return prompt, expected_output
    
    def _generate_dimensions(self, geometry_type: GeometryType) -> Dimensions:
        """Generate realistic dimensions for geometry type"""
        dims = Dimensions()
        
        if geometry_type in [GeometryType.CUBE]:
            dims.length = round(random.uniform(*self.dimension_ranges['length']), 2)
            dims.width = round(random.uniform(*self.dimension_ranges['width']), 2)
            dims.height = round(random.uniform(*self.dimension_ranges['height']), 2)
        
        elif geometry_type in [GeometryType.SPHERE]:
            dims.radius = round(random.uniform(*self.dimension_ranges['radius']), 2)
            # Sometimes use diameter instead
            if random.random() < 0.3:
                dims.diameter = dims.radius * 2
                dims.radius = None
        
        elif geometry_type in [GeometryType.CYLINDER, GeometryType.CONE]:
            dims.radius = round(random.uniform(*self.dimension_ranges['radius']), 2)
            dims.height = round(random.uniform(*self.dimension_ranges['height']), 2)
            # Sometimes use diameter instead
            if random.random() < 0.3:
                dims.diameter = dims.radius * 2
                dims.radius = None
        
        elif geometry_type in [GeometryType.TORUS]:
            dims.radius = round(random.uniform(5, 50), 2)  # Major radius
            dims.thickness = round(random.uniform(1, dims.radius/3), 2)  # Minor radius
        
        return dims
    
    def _format_dimensions(self, dimensions: Dimensions, geometry_type: GeometryType) -> str:
        """Format dimensions as text for prompt"""
        parts = []
        
        if dimensions.length:
            parts.append(f"length {dimensions.length}mm")
        if dimensions.width:
            parts.append(f"width {dimensions.width}mm")
        if dimensions.height:
            parts.append(f"height {dimensions.height}mm")
        if dimensions.radius:
            parts.append(f"radius {dimensions.radius}mm")
        if dimensions.diameter:
            parts.append(f"diameter {dimensions.diameter}mm")
        if dimensions.thickness:
            parts.append(f"thickness {dimensions.thickness}mm")
        
        if len(parts) >= 3:
            return f"{parts[0]}, {parts[1]}, and {parts[2]}"
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        elif len(parts) == 1:
            return parts[0]
        else:
            return "10mm x 10mm x 10mm"
    
    def _add_complexity_features(self, prompt: str, geometry_type: GeometryType) -> str:
        """Add complexity features for hard examples"""
        features = []
        
        if random.random() < 0.5:
            features.append("with 2mm fillet on all edges")
        if random.random() < 0.3:
            features.append("positioned at x=10, y=20, z=0")
        if random.random() < 0.4:
            features.append("rotated 45 degrees")
        if random.random() < 0.3:
            features.append("with a 5mm hole through the center")
        
        if features:
            prompt += " " + ", ".join(features)
        
        return prompt
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify prompt for easy examples"""
        # Remove material and color specifications sometimes
        if random.random() < 0.3:
            words = prompt.split()
            simplified_words = []
            skip_next = False
            
            for i, word in enumerate(words):
                if skip_next:
                    skip_next = False
                    continue
                
                if word.lower() in [m.value for m in Material]:
                    skip_next = False  # Skip material
                elif word.lower() in self.colors:
                    skip_next = False  # Skip color
                else:
                    simplified_words.append(word)
            
            prompt = " ".join(simplified_words)
        
        return prompt
    
    def _generate_expected_output(self, geometry_type: GeometryType, dimensions: Dimensions,
                                material: Material, color: str, original_prompt: str) -> Dict[str, Any]:
        """Generate the expected JSON output structure"""
        primitive_id = str(uuid.uuid4())
        
        # Create primitive structure
        primitive = GeometricPrimitive(
            id=primitive_id,
            type=geometry_type,
            dimensions=dimensions,
            material=material,
            color=color,
            description=original_prompt,
            metadata={
                'generated': True,
                'dataset_version': '1.0.0',
                'confidence': random.uniform(0.7, 1.0)
            }
        )
        
        return primitive.to_dict()
    
    def _generate_fallback_prompt(self, geometry_type: GeometryType) -> Tuple[str, Dict[str, Any]]:
        """Generate fallback prompt for unsupported geometry types"""
        action = random.choice(self.action_verbs)
        material = random.choice(self.materials)
        color = random.choice(self.colors)
        
        prompt = f"{action} a {material.value} {color} {geometry_type.value}"
        
        dimensions = Dimensions(length=10, width=10, height=10)
        expected_output = self._generate_expected_output(geometry_type, dimensions, 
                                                       material, color, prompt)
        
        return prompt, expected_output

class ComplexAssemblyGenerator:
    """Generates complex multi-part assembly prompts"""
    
    def __init__(self):
        self.assembly_templates = [
            "Create a {assembly_type} consisting of {part1} and {part2} connected together",
            "Design a {assembly_type} with {part1} mounted on {part2}",
            "Build a {assembly_type} assembly with {part1} and {part2} in {material}",
            "Generate a {assembly_type} made from {part1} and {part2} components"
        ]
        
        self.assembly_types = [
            "bracket assembly", "mounting plate", "mechanical joint", "support structure",
            "connector assembly", "housing unit", "frame structure", "compound part"
        ]
        
        self.part_relationships = [
            "bolted to", "welded to", "inserted into", "mounted on", "connected to", "joined with"
        ]
    
    def generate_assembly_prompt(self) -> Tuple[str, Dict[str, Any]]:
        """Generate complex assembly prompt"""
        assembly_type = random.choice(self.assembly_types)
        template = random.choice(self.assembly_templates)
        
        # Generate two parts
        geometry_generator = GeometricPromptGenerator()
        
        part1_type = random.choice(list(GeometryType)[:6])  # Basic shapes only
        part2_type = random.choice(list(GeometryType)[:6])
        
        part1_desc, part1_data = geometry_generator.generate_prompt(part1_type, "medium")
        part2_desc, part2_data = geometry_generator.generate_prompt(part2_type, "medium")
        
        # Extract part descriptions
        part1_name = f"{part1_type.value}"
        part2_name = f"{part2_type.value}"
        
        material = random.choice(list(Material))
        
        prompt = template.format(
            assembly_type=assembly_type,
            part1=part1_name,
            part2=part2_name,
            material=material.value
        )
        
        # Generate expected output for assembly
        assembly_id = str(uuid.uuid4())
        
        expected_output = {
            "id": assembly_id,
            "type": "assembly",
            "name": assembly_type,
            "description": prompt,
            "components": [
                {
                    "id": part1_data["id"],
                    "type": part1_data["type"],
                    "dimensions": part1_data["dimensions"],
                    "position": {"x": 0, "y": 0, "z": 0},
                    "parent_id": assembly_id
                },
                {
                    "id": part2_data["id"], 
                    "type": part2_data["type"],
                    "dimensions": part2_data["dimensions"],
                    "position": {"x": 10, "y": 0, "z": 0},
                    "parent_id": assembly_id
                }
            ],
            "material": material.value,
            "metadata": {
                "complexity": "high",
                "component_count": 2,
                "generated": True
            }
        }
        
        return prompt, expected_output

class ManufacturingContextGenerator:
    """Generates manufacturing-specific prompts"""
    
    def __init__(self):
        self.manufacturing_processes = [
            "CNC machining", "3D printing", "injection molding", "casting", 
            "forging", "laser cutting", "waterjet cutting", "turning", "milling"
        ]
        
        self.tolerances = ["±0.1mm", "±0.05mm", "±0.2mm", "±0.01mm"]
        self.surface_finishes = ["Ra 3.2", "Ra 1.6", "Ra 0.8", "polished", "as-machined"]
        
        self.manufacturing_templates = [
            "Design a {part} for {process} with {tolerance} tolerance and {finish} surface finish",
            "Create a {part} suitable for {process} manufacturing in {material}",
            "Generate a {part} optimized for {process} with {tolerance} dimensional accuracy",
            "Make a {part} for {process} production with {finish} finish requirements"
        ]
    
    def generate_manufacturing_prompt(self) -> Tuple[str, Dict[str, Any]]:
        """Generate manufacturing-context prompt"""
        geometry_generator = GeometricPromptGenerator()
        geometry_type = random.choice(list(GeometryType)[:6])
        
        base_prompt, base_output = geometry_generator.generate_prompt(geometry_type, "medium")
        
        # Add manufacturing context
        process = random.choice(self.manufacturing_processes)
        tolerance = random.choice(self.tolerances)
        finish = random.choice(self.surface_finishes)
        material = random.choice(list(Material))
        
        template = random.choice(self.manufacturing_templates)
        
        manufacturing_prompt = template.format(
            part=geometry_type.value,
            process=process,
            tolerance=tolerance,
            finish=finish,
            material=material.value
        )
        
        # Enhance output with manufacturing data
        enhanced_output = base_output.copy()
        enhanced_output["manufacturing"] = {
            "process": process,
            "tolerance": tolerance,
            "surface_finish": finish,
            "material": material.value,
            "optimized_for_manufacturing": True
        }
        
        enhanced_output["metadata"]["manufacturing_context"] = True
        enhanced_output["description"] = manufacturing_prompt
        
        return manufacturing_prompt, enhanced_output

# =====================================================
# DATASET COMPILATION AND EXPORT
# =====================================================

class DatasetCompiler:
    """Compiles and exports training datasets in various formats"""
    
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = output_dir
        self.geometry_generator = GeometricPromptGenerator()
        self.assembly_generator = ComplexAssemblyGenerator()
        self.manufacturing_generator = ManufacturingContextGenerator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_dataset(self, 
                                     total_examples: int = 10000,
                                     validation_split: float = 0.2) -> Dict[str, List[TrainingExample]]:
        """
        Generate comprehensive dataset for fine-tuning
        
        Args:
            total_examples: Total number of examples to generate
            validation_split: Fraction for validation set
            
        Returns:
            Dictionary with 'train' and 'validation' splits
        """
        print(f"Generating {total_examples} training examples...")
        
        all_examples = []
        
        # Basic geometry examples (40%)
        basic_count = int(total_examples * 0.4)
        basic_examples = self._generate_basic_geometry_examples(basic_count)
        all_examples.extend(basic_examples)
        
        # Complex assemblies (20%)
        assembly_count = int(total_examples * 0.2)
        assembly_examples = self._generate_assembly_examples(assembly_count)
        all_examples.extend(assembly_examples)
        
        # Manufacturing context (30%)
        manufacturing_count = int(total_examples * 0.3)
        manufacturing_examples = self._generate_manufacturing_examples(manufacturing_count)
        all_examples.extend(manufacturing_examples)
        
        # Validation examples (10%)
        validation_count = total_examples - len(all_examples)
        validation_examples = self._generate_validation_examples(validation_count)
        all_examples.extend(validation_examples)
        
        # Shuffle and split
        random.shuffle(all_examples)
        
        split_index = int(len(all_examples) * (1 - validation_split))
        train_examples = all_examples[:split_index]
        val_examples = all_examples[split_index:]
        
        print(f"Generated {len(train_examples)} training examples")
        print(f"Generated {len(val_examples)} validation examples")
        
        return {
            'train': train_examples,
            'validation': val_examples
        }
    
    def _generate_basic_geometry_examples(self, count: int) -> List[TrainingExample]:
        """Generate basic geometry examples"""
        examples = []
        
        geometry_types = list(GeometryType)[:6]  # Basic shapes
        difficulties = ['easy', 'medium', 'hard']
        
        for i in range(count):
            geometry_type = random.choice(geometry_types)
            difficulty = random.choice(difficulties)
            
            prompt, expected_output = self.geometry_generator.generate_prompt(geometry_type, difficulty)
            
            example = TrainingExample(
                id=str(uuid.uuid4()),
                input_text=prompt,
                expected_output=expected_output,
                category=DatasetType.BASIC_GEOMETRY,
                difficulty=difficulty,
                metadata={
                    "geometry_type": geometry_type.value,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            examples.append(example)
        
        return examples
    
    def _generate_assembly_examples(self, count: int) -> List[TrainingExample]:
        """Generate complex assembly examples"""
        examples = []
        
        for i in range(count):
            prompt, expected_output = self.assembly_generator.generate_assembly_prompt()
            
            example = TrainingExample(
                id=str(uuid.uuid4()),
                input_text=prompt,
                expected_output=expected_output,
                category=DatasetType.COMPLEX_ASSEMBLIES,
                difficulty='hard',
                metadata={
                    "assembly_type": True,
                    "component_count": expected_output.get("metadata", {}).get("component_count", 2),
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            examples.append(example)
        
        return examples
    
    def _generate_manufacturing_examples(self, count: int) -> List[TrainingExample]:
        """Generate manufacturing context examples"""
        examples = []
        
        for i in range(count):
            prompt, expected_output = self.manufacturing_generator.generate_manufacturing_prompt()
            
            example = TrainingExample(
                id=str(uuid.uuid4()),
                input_text=prompt,
                expected_output=expected_output,
                category=DatasetType.MANUFACTURING,
                difficulty='medium',
                metadata={
                    "manufacturing_context": True,
                    "process": expected_output.get("manufacturing", {}).get("process"),
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            examples.append(example)
        
        return examples
    
    def _generate_validation_examples(self, count: int) -> List[TrainingExample]:
        """Generate validation examples"""
        examples = []
        
        # Mix of all types for validation
        for i in range(count):
            example_type = random.choice([
                DatasetType.BASIC_GEOMETRY,
                DatasetType.COMPLEX_ASSEMBLIES,
                DatasetType.MANUFACTURING
            ])
            
            if example_type == DatasetType.BASIC_GEOMETRY:
                geometry_type = random.choice(list(GeometryType)[:6])
                prompt, expected_output = self.geometry_generator.generate_prompt(geometry_type, "medium")
            elif example_type == DatasetType.COMPLEX_ASSEMBLIES:
                prompt, expected_output = self.assembly_generator.generate_assembly_prompt()
            else:
                prompt, expected_output = self.manufacturing_generator.generate_manufacturing_prompt()
            
            example = TrainingExample(
                id=str(uuid.uuid4()),
                input_text=prompt,
                expected_output=expected_output,
                category=DatasetType.VALIDATION,
                difficulty='medium',
                metadata={
                    "validation_example": True,
                    "source_type": example_type.value,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            examples.append(example)
        
        return examples
    
    def export_datasets(self, datasets: Dict[str, List[TrainingExample]], 
                       formats: List[str] = ['jsonl', 'json', 'csv']) -> Dict[str, str]:
        """
        Export datasets in multiple formats
        
        Args:
            datasets: Dictionary with train/validation splits
            formats: List of export formats ('jsonl', 'json', 'csv')
            
        Returns:
            Dictionary mapping format to output file path
        """
        output_files = {}
        
        for split_name, examples in datasets.items():
            if 'jsonl' in formats:
                jsonl_path = self._export_jsonl(examples, f"{split_name}_dataset.jsonl")
                output_files[f'{split_name}_jsonl'] = jsonl_path
            
            if 'json' in formats:
                json_path = self._export_json(examples, f"{split_name}_dataset.json")
                output_files[f'{split_name}_json'] = json_path
            
            if 'csv' in formats:
                csv_path = self._export_csv(examples, f"{split_name}_dataset.csv")
                output_files[f'{split_name}_csv'] = csv_path
        
        # Export combined statistics
        self._export_statistics(datasets)
        
        return output_files
    
    def _export_jsonl(self, examples: List[TrainingExample], filename: str) -> str:
        """Export to JSONL format (ideal for fine-tuning)"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(example.to_jsonl_format() + '\n')
        
        print(f"Exported {len(examples)} examples to {output_path}")
        return output_path
    
    def _export_json(self, examples: List[TrainingExample], filename: str) -> str:
        """Export to JSON format"""
        output_path = os.path.join(self.output_dir, filename)
        
        data = {
            "metadata": {
                "total_examples": len(examples),
                "generated_at": datetime.now().isoformat(),
                "dataset_version": "1.0.0"
            },
            "examples": [example.to_dict() for example in examples]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(examples)} examples to {output_path}")
        return output_path
    
    def _export_csv(self, examples: List[TrainingExample], filename: str) -> str:
        """Export to CSV format"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['id', 'input_text', 'expected_output_json', 'category', 'difficulty', 'metadata'])
            
            # Data
            for example in examples:
                writer.writerow([
                    example.id,
                    example.input_text,
                    json.dumps(example.expected_output),
                    example.category.value,
                    example.difficulty,
                    json.dumps(example.metadata)
                ])
        
        print(f"Exported {len(examples)} examples to {output_path}")
        return output_path
    
    def _export_statistics(self, datasets: Dict[str, List[TrainingExample]]) -> str:
        """Export dataset statistics"""
        stats_path = os.path.join(self.output_dir, "dataset_statistics.json")
        
        stats = {
            "generated_at": datetime.now().isoformat(),
            "total_examples": sum(len(examples) for examples in datasets.values()),
            "splits": {}
        }
        
        for split_name, examples in datasets.items():
            category_counts = {}
            difficulty_counts = {}
            geometry_type_counts = {}
            
            for example in examples:
                # Category counts
                cat = example.category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
                # Difficulty counts
                diff = example.difficulty
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                
                # Geometry type counts
                geom_type = example.metadata.get('geometry_type', 'unknown')
                geometry_type_counts[geom_type] = geometry_type_counts.get(geom_type, 0) + 1
            
            stats["splits"][split_name] = {
                "total_examples": len(examples),
                "categories": category_counts,
                "difficulties": difficulty_counts,
                "geometry_types": geometry_type_counts
            }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Exported statistics to {stats_path}")
        return stats_path

# =====================================================
# MAIN GENERATION FUNCTIONS
# =====================================================

def generate_fine_tuning_dataset(output_dir: str = "fine_tuning_datasets",
                                total_examples: int = 10000,
                                validation_split: float = 0.2) -> Dict[str, str]:
    """
    Generate complete fine-tuning dataset for the Natural Language Geometry Engine
    
    Args:
        output_dir: Directory to save datasets
        total_examples: Total number of examples to generate
        validation_split: Fraction for validation set
        
    Returns:
        Dictionary mapping format to output file paths
    """
    print("🚀 Starting Natural Language Geometry Engine dataset generation...")
    
    compiler = DatasetCompiler(output_dir)
    
    # Generate datasets
    datasets = compiler.generate_comprehensive_dataset(total_examples, validation_split)
    
    # Export in multiple formats
    output_files = compiler.export_datasets(datasets, formats=['jsonl', 'json', 'csv'])
    
    print("\n✅ Dataset generation complete!")
    print("\nOutput files:")
    for format_name, file_path in output_files.items():
        print(f"  {format_name}: {file_path}")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total examples: {total_examples}")
    print(f"  Training examples: {len(datasets['train'])}")
    print(f"  Validation examples: {len(datasets['validation'])}")
    
    print("\n💡 When to upload for fine-tuning:")
    print("  1. Upload the JSONL files to your LLM fine-tuning platform")
    print("  2. Recommended: Start with 1,000 examples for initial training")
    print("  3. Scale up to 10,000+ examples for production models")
    print("  4. Monitor validation loss to prevent overfitting")
    print("  5. Use temperature=0.1-0.3 for consistent geometric outputs")
    
    return output_files

if __name__ == "__main__":
    # Generate sample dataset
    output_files = generate_fine_tuning_dataset(
        total_examples=1000,  # Start with smaller dataset for testing
        validation_split=0.2
    )
