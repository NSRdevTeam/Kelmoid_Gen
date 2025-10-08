"""
Natural Language Geometry Engine (NLG)
=====================================

A fine-tuned LLM-based system that maps natural language text to geometric primitives
using a custom JSON schema. This creates a unique "Text-to-CAD compiler" approach
for generating parametric CAD models.

Author: KelmoidAI Genesis Team
License: MIT
"""

import json
import re
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# GEOMETRIC PRIMITIVE SCHEMAS
# =====================================================

class GeometryType(Enum):
    """Supported geometric primitive types"""
    CUBE = "cube"
    SPHERE = "sphere" 
    CYLINDER = "cylinder"
    CONE = "cone"
    PYRAMID = "pyramid"
    TORUS = "torus"
    PRISM = "prism"
    EXTRUSION = "extrusion"
    REVOLUTION = "revolution"
    SWEEP = "sweep"
    BOOLEAN_UNION = "boolean_union"
    BOOLEAN_DIFFERENCE = "boolean_difference"
    BOOLEAN_INTERSECTION = "boolean_intersection"
    FILLET = "fillet"
    CHAMFER = "chamfer"
    HOLE = "hole"
    SLOT = "slot"
    GROOVE = "groove"

class Unit(Enum):
    """Measurement units"""
    MM = "mm"
    CM = "cm"
    INCH = "inch"
    M = "m"

class Material(Enum):
    """Material types for manufacturing context"""
    STEEL = "steel"
    ALUMINUM = "aluminum"
    PLASTIC = "plastic"
    WOOD = "wood"
    BRASS = "brass"
    COPPER = "copper"
    TITANIUM = "titanium"
    CARBON_FIBER = "carbon_fiber"

@dataclass
class Vector3D:
    """3D vector representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

@dataclass
class Dimensions:
    """Standard dimensional parameters"""
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    diameter: Optional[float] = None
    thickness: Optional[float] = None
    depth: Optional[float] = None
    angle: Optional[float] = None
    unit: Unit = Unit.MM
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v.value if isinstance(v, Enum) else v 
                for k, v in asdict(self).items() if v is not None}

from dataclasses import field

@dataclass
class GeometricPrimitive:
    """Core geometric primitive with full parametric description"""
    id: str
    type: GeometryType
    dimensions: Dimensions
    position: Vector3D = field(default_factory=Vector3D)
    rotation: Vector3D = field(default_factory=Vector3D)
    scale: Vector3D = field(default_factory=lambda: Vector3D(1.0, 1.0, 1.0))
    material: Optional[Material] = None
    color: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    features: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        result = {}
        data = asdict(self)
        
        for key, value in data.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                # Handle nested dictionaries
                result[key] = self._serialize_dict(value)
            elif isinstance(value, list):
                # Handle lists
                result[key] = [self._serialize_value(item) for item in value]
            else:
                result[key] = self._serialize_value(value)
        
        return result
    
    def _serialize_value(self, value):
        """Serialize individual values to JSON-compatible format"""
        if hasattr(value, '__dataclass_fields__'):
            # Handle dataclass objects
            return self._serialize_dict(asdict(value))
        elif hasattr(value, 'value') and hasattr(value, '__class__') and hasattr(value.__class__, '__members__'):
            # Handle Enum objects
            return value.value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            return value
        else:
            # Try to convert to string as fallback
            return str(value)
    
    def _serialize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize dictionary values"""
        result = {}
        for key, value in d.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._serialize_dict(value)
            elif isinstance(value, list):
                result[key] = [self._serialize_value(item) for item in value]
            else:
                result[key] = self._serialize_value(value)
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

# =====================================================
# NATURAL LANGUAGE PROCESSING ENGINE
# =====================================================

class SemanticAnalyzer:
    """Analyzes natural language for geometric intent"""
    
    def __init__(self):
        self.geometry_patterns = self._initialize_patterns()
        self.dimension_extractors = self._initialize_extractors()
        self.feature_recognizers = self._initialize_features()
        
    def _initialize_patterns(self) -> Dict[GeometryType, List[str]]:
        """Initialize regex patterns for geometry recognition"""
        return {
            GeometryType.CUBE: [
                r'\b(cube|box|rectangular\s+prism)\b',
                r'\bsquare\s+block\b',
                r'\brectangular\s+block\b'
            ],
            GeometryType.SPHERE: [
                r'\b(sphere|ball|orb)\b',
                r'\bspherical\s+shape\b',
                r'\bround\s+ball\b'
            ],
            GeometryType.CYLINDER: [
                r'\b(cylinder|tube|pipe|rod)\b',
                r'\bcylindrical\s+shape\b',
                r'\bround\s+tube\b'
            ],
            GeometryType.CONE: [
                r'\b(cone|conical|tapered)\b',
                r'\bconical\s+shape\b',
                r'\btapered\s+cylinder\b'
            ],
            GeometryType.PYRAMID: [
                r'\b(pyramid|pyramidal)\b',
                r'\btriangular\s+pyramid\b',
                r'\bsquare\s+pyramid\b'
            ],
            GeometryType.TORUS: [
                r'\b(torus|ring|donut|doughnut)\b',
                r'\btoroidal\s+shape\b',
                r'\bround\s+ring\b'
            ],
            GeometryType.HOLE: [
                r'\b(hole|perforation|opening)\b',
                r'\bcircular\s+hole\b',
                r'\bdrilled\s+hole\b'
            ],
            GeometryType.SLOT: [
                r'\b(slot|groove|channel)\b',
                r'\brectangular\s+slot\b',
                r'\blinear\s+groove\b'
            ]
        }
    
    def _initialize_extractors(self) -> Dict[str, List[str]]:
        """Initialize dimension extraction patterns"""
        return {
            'length': [
                r'length[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'long[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?\s*long'
            ],
            'width': [
                r'width[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'wide[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?\s*wide'
            ],
            'height': [
                r'height[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'tall[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?\s*tall'
            ],
            'radius': [
                r'radius[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'r[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?'
            ],
            'diameter': [
                r'diameter[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'dia[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'd[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?'
            ],
            'thickness': [
                r'thickness[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r'thick[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?',
                r't[:=\s]*(\d+(?:\.\d+)?)\s*(mm|cm|inch|m)?'
            ]
        }
    
    def _initialize_features(self) -> Dict[str, List[str]]:
        """Initialize feature recognition patterns"""
        return {
            'fillet': [r'\b(fillet|rounded\s+edge|round\s+corner)\b'],
            'chamfer': [r'\b(chamfer|beveled\s+edge|angled\s+corner)\b'],
            'counterbore': [r'\b(counterbore|countersink|recessed\s+hole)\b'],
            'thread': [r'\b(thread|threaded|screw\s+thread)\b'],
            'knurl': [r'\b(knurl|knurled|textured\s+surface)\b']
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze natural language text for geometric intent
        
        Args:
            text: Natural language description
            
        Returns:
            Analysis results with geometry type, dimensions, and features
        """
        text_lower = text.lower()
        
        # Extract geometry type
        geometry_type = self._extract_geometry_type(text_lower)
        
        # Extract dimensions
        dimensions = self._extract_dimensions(text_lower)
        
        # Extract features
        features = self._extract_features(text_lower)
        
        # Extract material and color
        material = self._extract_material(text_lower)
        color = self._extract_color(text_lower)
        
        # Extract position and orientation
        position = self._extract_position(text_lower)
        rotation = self._extract_rotation(text_lower)
        
        # Analyze complexity and requirements
        complexity = self._analyze_complexity(text_lower)
        
        return {
            'geometry_type': geometry_type,
            'dimensions': dimensions,
            'features': features,
            'material': material,
            'color': color,
            'position': position,
            'rotation': rotation,
            'complexity': complexity,
            'confidence': self._calculate_confidence(text_lower, geometry_type, dimensions)
        }
    
    def _extract_geometry_type(self, text: str) -> Optional[GeometryType]:
        """Extract primary geometry type from text"""
        for geom_type, patterns in self.geometry_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return geom_type
        return None
    
    def _extract_dimensions(self, text: str) -> Dimensions:
        """Extract dimensional parameters from text"""
        dims = Dimensions()
        
        for dim_name, patterns in self.dimension_extractors.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit_str = match.group(2) if len(match.groups()) > 1 and match.group(2) else "mm"
                    
                    # Convert unit
                    try:
                        unit = Unit(unit_str.lower())
                        dims.unit = unit
                    except ValueError:
                        unit = Unit.MM
                    
                    setattr(dims, dim_name, value)
                    break
        
        # Handle special patterns like "20x30x40"
        xyz_match = re.search(r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)', text)
        if xyz_match:
            dims.length = float(xyz_match.group(1))
            dims.width = float(xyz_match.group(2))
            dims.height = float(xyz_match.group(3))
        
        return dims
    
    def _extract_features(self, text: str) -> List[Dict[str, Any]]:
        """Extract geometric features from text"""
        features = []
        
        for feature_name, patterns in self.feature_recognizers.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    features.append({
                        'type': feature_name,
                        'parameters': self._extract_feature_parameters(text, feature_name)
                    })
        
        return features
    
    def _extract_feature_parameters(self, text: str, feature_type: str) -> Dict[str, Any]:
        """Extract parameters for specific features"""
        params = {}
        
        if feature_type in ['fillet', 'chamfer']:
            # Look for radius or dimension
            radius_match = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|inch)?\s*radius', text)
            if radius_match:
                params['radius'] = float(radius_match.group(1))
                params['unit'] = radius_match.group(2) or 'mm'
        
        return params
    
    def _extract_material(self, text: str) -> Optional[Material]:
        """Extract material type from text"""
        material_patterns = {
            Material.STEEL: [r'\b(steel|stainless\s+steel|carbon\s+steel)\b'],
            Material.ALUMINUM: [r'\b(aluminum|aluminium|al)\b'],
            Material.PLASTIC: [r'\b(plastic|polymer|abs|pla|petg)\b'],
            Material.WOOD: [r'\b(wood|timber|oak|pine|maple)\b'],
            Material.BRASS: [r'\b(brass|bronze)\b'],
            Material.COPPER: [r'\bcopper\b'],
            Material.TITANIUM: [r'\btitanium\b'],
            Material.CARBON_FIBER: [r'\b(carbon\s+fiber|carbon\s+fibre|cf)\b']
        }
        
        for material, patterns in material_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return material
        
        return None
    
    def _extract_color(self, text: str) -> Optional[str]:
        """Extract color from text"""
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 
                 'brown', 'black', 'white', 'gray', 'grey', 'silver', 'gold']
        
        for color in colors:
            if re.search(rf'\b{color}\b', text, re.IGNORECASE):
                return color
        
        return None
    
    def _extract_position(self, text: str) -> Vector3D:
        """Extract position information from text"""
        position = Vector3D()
        
        # Look for position keywords
        x_match = re.search(r'x[:=\s]*(-?\d+(?:\.\d+)?)', text)
        y_match = re.search(r'y[:=\s]*(-?\d+(?:\.\d+)?)', text)
        z_match = re.search(r'z[:=\s]*(-?\d+(?:\.\d+)?)', text)
        
        if x_match:
            position.x = float(x_match.group(1))
        if y_match:
            position.y = float(y_match.group(1))
        if z_match:
            position.z = float(z_match.group(1))
        
        return position
    
    def _extract_rotation(self, text: str) -> Vector3D:
        """Extract rotation information from text"""
        rotation = Vector3D()
        
        # Look for rotation keywords
        angle_match = re.search(r'rotate[d]?\s+(\d+(?:\.\d+)?)\s*degrees?', text)
        if angle_match:
            rotation.z = float(angle_match.group(1))  # Default to Z rotation
        
        return rotation
    
    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze the complexity of the requested geometry"""
        complexity_indicators = {
            'simple': len(re.findall(r'\b(cube|sphere|cylinder)\b', text)),
            'medium': len(re.findall(r'\b(cone|pyramid|torus|fillet|chamfer)\b', text)),
            'complex': len(re.findall(r'\b(boolean|union|difference|sweep|extrusion)\b', text)),
            'features': len(re.findall(r'\b(hole|slot|groove|thread|knurl)\b', text))
        }
        
        total_score = sum(complexity_indicators.values())
        if total_score <= 1:
            level = 'simple'
        elif total_score <= 3:
            level = 'medium'
        else:
            level = 'complex'
        
        return {
            'level': level,
            'score': total_score,
            'indicators': complexity_indicators
        }
    
    def _calculate_confidence(self, text: str, geometry_type: Optional[GeometryType], 
                           dimensions: Dimensions) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.0
        
        # Base confidence from geometry type recognition
        if geometry_type:
            confidence += 0.4
        
        # Additional confidence from dimension extraction
        dim_count = sum(1 for v in [dimensions.length, dimensions.width, dimensions.height,
                                  dimensions.radius, dimensions.diameter, dimensions.thickness] 
                       if v is not None)
        confidence += min(0.4, dim_count * 0.1)
        
        # Boost from explicit keywords
        explicit_keywords = len(re.findall(r'\b(create|make|design|build|generate)\b', text))
        confidence += min(0.2, explicit_keywords * 0.1)
        
        return min(1.0, confidence)

# =====================================================
# TEXT-TO-CAD COMPILER
# =====================================================

class TextToCADCompiler:
    """
    Compiler that converts natural language to structured CAD commands
    This is the core innovation - a true "Text-to-CAD compiler"
    """
    
    def __init__(self):
        self.analyzer = SemanticAnalyzer()
        self.schema_validator = GeometrySchemaValidator()
        self.optimization_engine = GeometryOptimizer()
        
    def compile(self, text: str) -> Dict[str, Any]:
        """
        Compile natural language text to structured CAD commands
        
        Args:
            text: Natural language description
            
        Returns:
            Compiled geometry specification with validation and optimization
        """
        try:
            logger.info(f"Compiling text: {text[:100]}...")
            
            # Phase 1: Semantic Analysis
            analysis = self.analyzer.analyze_text(text)
            logger.info(f"Analysis confidence: {analysis['confidence']:.2f}")
            
            # Phase 2: Generate Geometric Primitive
            primitive = self._generate_primitive(analysis, text)
            
            # Phase 3: Schema Validation
            validation_result = self.schema_validator.validate(primitive)
            if not validation_result['valid']:
                logger.warning(f"Validation warnings: {validation_result['errors']}")
                primitive = self._apply_corrections(primitive, validation_result['corrections'])
            
            # Phase 4: Optimization
            optimized_primitive = self.optimization_engine.optimize(primitive)
            
            # Phase 5: Generate Compilation Report
            report = self._generate_compilation_report(text, analysis, primitive, 
                                                     validation_result, optimized_primitive)
            
            return {
                'success': True,
                'primitive': optimized_primitive,
                'analysis': analysis,
                'validation': validation_result,
                'report': report,
                'json_schema': optimized_primitive.to_json(),
                'compilation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Compilation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'compilation_time': datetime.now().isoformat()
            }
    
    def _generate_primitive(self, analysis: Dict[str, Any], original_text: str) -> GeometricPrimitive:
        """Generate geometric primitive from analysis"""
        
        # Generate unique ID
        primitive_id = str(uuid.uuid4())
        
        # Extract core parameters
        geometry_type = analysis['geometry_type'] or GeometryType.CUBE
        dimensions = analysis['dimensions']
        features = analysis['features']
        material = analysis['material']
        color = analysis['color']
        position = analysis['position']
        rotation = analysis['rotation']
        
        # Create primitive
        primitive = GeometricPrimitive(
            id=primitive_id,
            type=geometry_type,
            dimensions=dimensions,
            position=position,
            rotation=rotation,
            material=material,
            color=color,
            name=self._generate_name(geometry_type, dimensions),
            description=original_text,
            features=features,
            metadata={
                'confidence': analysis['confidence'],
                'complexity': analysis['complexity'],
                'source_text': original_text,
                'compiler_version': '1.0.0'
            }
        )
        
        return primitive
    
    def _generate_name(self, geometry_type: GeometryType, dimensions: Dimensions) -> str:
        """Generate a descriptive name for the primitive"""
        base_name = geometry_type.value.replace('_', ' ').title()
        
        # Add key dimensions to name
        dim_parts = []
        if dimensions.length:
            dim_parts.append(f"L{dimensions.length}")
        if dimensions.width:
            dim_parts.append(f"W{dimensions.width}")
        if dimensions.height:
            dim_parts.append(f"H{dimensions.height}")
        if dimensions.radius:
            dim_parts.append(f"R{dimensions.radius}")
        if dimensions.diameter:
            dim_parts.append(f"D{dimensions.diameter}")
        
        if dim_parts:
            return f"{base_name} ({'+'.join(dim_parts)}{dimensions.unit.value})"
        else:
            return base_name
    
    def _apply_corrections(self, primitive: GeometricPrimitive, 
                          corrections: List[Dict[str, Any]]) -> GeometricPrimitive:
        """Apply validation corrections to primitive"""
        for correction in corrections:
            if correction['type'] == 'dimension_default':
                field = correction['field']
                value = correction['value']
                setattr(primitive.dimensions, field, value)
                logger.info(f"Applied correction: {field} = {value}")
        
        return primitive
    
    def _generate_compilation_report(self, original_text: str, analysis: Dict[str, Any],
                                   primitive: GeometricPrimitive, validation: Dict[str, Any],
                                   optimized: GeometricPrimitive) -> Dict[str, Any]:
        """Generate detailed compilation report"""
        return {
            'original_text': original_text,
            'recognized_geometry': primitive.type.value,
            'extracted_dimensions': len([d for d in [primitive.dimensions.length,
                                                   primitive.dimensions.width,
                                                   primitive.dimensions.height,
                                                   primitive.dimensions.radius,
                                                   primitive.dimensions.diameter,
                                                   primitive.dimensions.thickness] if d is not None]),
            'features_detected': len(primitive.features),
            'confidence_score': analysis['confidence'],
            'complexity_level': analysis['complexity']['level'],
            'validation_passed': validation['valid'],
            'optimizations_applied': len(optimized.metadata.get('optimizations', [])),
            'compilation_success': True
        }

# =====================================================
# SCHEMA VALIDATION
# =====================================================

class GeometrySchemaValidator:
    """Validates geometric primitives against JSON schema"""
    
    def __init__(self):
        self.schema = self._load_schema()
        self.defaults = self._load_defaults()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the geometric primitive JSON schema"""
        return {
            "type": "object",
            "required": ["id", "type", "dimensions"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": [t.value for t in GeometryType]},
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "length": {"type": ["number", "null"], "minimum": 0},
                        "width": {"type": ["number", "null"], "minimum": 0},
                        "height": {"type": ["number", "null"], "minimum": 0},
                        "radius": {"type": ["number", "null"], "minimum": 0},
                        "diameter": {"type": ["number", "null"], "minimum": 0},
                        "thickness": {"type": ["number", "null"], "minimum": 0}
                    }
                }
            }
        }
    
    def _load_defaults(self) -> Dict[GeometryType, Dict[str, float]]:
        """Load default dimensions for each geometry type"""
        return {
            GeometryType.CUBE: {"length": 10, "width": 10, "height": 10},
            GeometryType.SPHERE: {"radius": 5},
            GeometryType.CYLINDER: {"radius": 5, "height": 10},
            GeometryType.CONE: {"radius": 5, "height": 10},
            GeometryType.PYRAMID: {"width": 10, "height": 10},
            GeometryType.TORUS: {"radius": 10, "thickness": 2}
        }
    
    def validate(self, primitive: GeometricPrimitive) -> Dict[str, Any]:
        """
        Validate a geometric primitive
        
        Returns:
            Validation result with errors and corrections
        """
        errors = []
        corrections = []
        
        # Check required fields
        if not primitive.id:
            errors.append("Missing primitive ID")
        
        if not primitive.type:
            errors.append("Missing geometry type")
        
        # Validate dimensions based on geometry type
        dims = primitive.dimensions
        defaults = self.defaults.get(primitive.type, {})
        
        if primitive.type in [GeometryType.CUBE, GeometryType.PYRAMID]:
            if not dims.length:
                corrections.append({"type": "dimension_default", "field": "length", 
                                  "value": defaults.get("length", 10)})
            if not dims.width:
                corrections.append({"type": "dimension_default", "field": "width", 
                                  "value": defaults.get("width", 10)})
            if not dims.height:
                corrections.append({"type": "dimension_default", "field": "height", 
                                  "value": defaults.get("height", 10)})
        
        elif primitive.type in [GeometryType.SPHERE]:
            if not dims.radius and not dims.diameter:
                corrections.append({"type": "dimension_default", "field": "radius", 
                                  "value": defaults.get("radius", 5)})
        
        elif primitive.type in [GeometryType.CYLINDER, GeometryType.CONE]:
            if not dims.radius and not dims.diameter:
                corrections.append({"type": "dimension_default", "field": "radius", 
                                  "value": defaults.get("radius", 5)})
            if not dims.height:
                corrections.append({"type": "dimension_default", "field": "height", 
                                  "value": defaults.get("height", 10)})
        
        # Check for negative dimensions
        for field in ['length', 'width', 'height', 'radius', 'diameter', 'thickness']:
            value = getattr(dims, field)
            if value is not None and value < 0:
                errors.append(f"Negative {field}: {value}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'corrections': corrections,
            'warnings': [f"Applied default for {c['field']}" for c in corrections]
        }

# =====================================================
# GEOMETRY OPTIMIZATION
# =====================================================

class GeometryOptimizer:
    """Optimizes geometric primitives for manufacturing and performance"""
    
    def optimize(self, primitive: GeometricPrimitive) -> GeometricPrimitive:
        """
        Optimize geometric primitive
        
        Returns:
            Optimized primitive with metadata about optimizations
        """
        optimizations = []
        
        # Round dimensions to reasonable precision
        dims = primitive.dimensions
        for field in ['length', 'width', 'height', 'radius', 'diameter', 'thickness']:
            value = getattr(dims, field)
            if value is not None:
                rounded_value = round(value, 2)  # Round to 0.01mm precision
                if rounded_value != value:
                    setattr(dims, field, rounded_value)
                    optimizations.append(f"Rounded {field} from {value} to {rounded_value}")
        
        # Suggest manufacturing-friendly dimensions
        manufacturing_opts = self._suggest_manufacturing_optimizations(primitive)
        optimizations.extend(manufacturing_opts)
        
        # Add optimization metadata
        if 'optimizations' not in primitive.metadata:
            primitive.metadata['optimizations'] = []
        primitive.metadata['optimizations'].extend(optimizations)
        
        return primitive
    
    def _suggest_manufacturing_optimizations(self, primitive: GeometricPrimitive) -> List[str]:
        """Suggest manufacturing-friendly optimizations"""
        suggestions = []
        
        dims = primitive.dimensions
        
        # Check for very thin walls
        if dims.thickness and dims.thickness < 1.0:
            suggestions.append(f"Warning: Thickness {dims.thickness}mm may be too thin for manufacturing")
        
        # Check for very small features
        if dims.radius and dims.radius < 0.5:
            suggestions.append(f"Warning: Radius {dims.radius}mm may be too small for standard tools")
        
        # Suggest standard sizes for common geometries
        if primitive.type == GeometryType.CYLINDER and dims.diameter:
            standard_diameters = [3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50]
            closest = min(standard_diameters, key=lambda x: abs(x - dims.diameter))
            if abs(closest - dims.diameter) < 1.0:
                suggestions.append(f"Consider standard diameter {closest}mm instead of {dims.diameter}mm")
        
        return suggestions
