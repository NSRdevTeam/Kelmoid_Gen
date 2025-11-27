interface DesignIntent {
  primaryShape: string;
  dimensions: Map<string, number>;
  features: string[];
  constraints: string[];
  materialRequirements?: string;
  tolerance?: string;
  manufacturing?: string;
  confidence: number;
}

interface OptimizedPrompt {
  original: string;
  optimized: string;
  designIntent: DesignIntent;
  reasoning: string[];
  suggestedImprovements: string[];
}

/**
 * Prompt Optimization Layer
 * This is the patentable "Design Intention Interpreter"
 * It uses AI reasoning to understand and reformulate engineering prompts
 */
class PromptOptimizationLayer {
  private aiModel: any; // In production, this would be your 7B model
  private engineeringVocabulary: Map<string, string[]>;
  private geometricPatterns: Map<string, RegExp>;

  constructor() {
    this.engineeringVocabulary = this.loadEngineeringVocabulary();
    this.geometricPatterns = this.loadGeometricPatterns();
  }

  /**
   * Main method: Analyzes and optimizes the user's prompt
  */
  async optimizePrompt(userPrompt: string): Promise<OptimizedPrompt> {
    console.log('🔍 Analyzing design intent...');

    // Step 1: Parse the prompt to extract design intent
    const designIntent = await this.extractDesignIntent(userPrompt);

    // Step 2: Identify missing critical information
    const missingInfo = this.identifyMissingInformation(designIntent);

    // Step 3: Apply engineering best practices
    const reasoning = this.applyEngineeringReasoning(designIntent, missingInfo);

    // Step 4: Reformulate the prompt for better CAD generation
    const optimizedPrompt = this.reformulatePrompt(userPrompt, designIntent, missingInfo);

    // Step 5: Generate suggestions for the user
    const suggestedImprovements = this.generateSuggestions(designIntent, missingInfo);

    return {
      original: userPrompt,
      optimized: optimizedPrompt,
      designIntent,
      reasoning,
      suggestedImprovements,
    };
  }

  /**
   * PATENTABLE COMPONENT 1: Design Intent Extraction
   * Uses AI reasoning to understand what the user wants to build
   */
  private async extractDesignIntent(prompt: string): Promise<DesignIntent> {
    const intent: DesignIntent = {
      primaryShape: 'unknown',
      dimensions: new Map(),
      features: [],
      constraints: [],
      confidence: 0,
    };

    // Identify primary shape using pattern matching + AI
    intent.primaryShape = this.identifyPrimaryShape(prompt);

    // Extract dimensions with units
    intent.dimensions = this.extractDimensions(prompt);

    // Identify features (holes, threads, chamfers, etc.)
    intent.features = this.identifyFeatures(prompt);

    // Extract constraints (tolerances, fits, finishes)
    intent.constraints = this.extractConstraints(prompt);

    // Extract material requirements
    intent.materialRequirements = this.extractMaterial(prompt);

    // Extract manufacturing context
    intent.manufacturing = this.extractManufacturingContext(prompt);

    // Calculate confidence score
    intent.confidence = this.calculateConfidence(intent);

    return intent;
  }

  /**
   * PATENTABLE COMPONENT 2: Missing Information Inference
   * AI infers what critical information is missing and suggests defaults
   */
  private identifyMissingInformation(intent: DesignIntent): string[] {
    const missing: string[] = [];

    // Check for missing dimensions
    const expectedDimensions = this.getExpectedDimensions(intent.primaryShape);
    expectedDimensions.forEach((dim) => {
      if (!intent.dimensions.has(dim)) {
        missing.push(`Missing ${dim} dimension for ${intent.primaryShape}`);
      }
    });

    // Check for manufacturing feasibility
    if (!intent.manufacturing) {
      missing.push('No manufacturing method specified');
    }

    // Check for material properties
    if (!intent.materialRequirements && intent.constraints.length > 0) {
      missing.push('Material not specified but constraints are present');
    }

    // Check for tolerance specifications
    if (intent.features.length > 0 && !intent.tolerance) {
      missing.push('Features specified but no tolerance given');
    }

    return missing;
  }

  /**
   * PATENTABLE COMPONENT 3: Engineering Reasoning Engine
   * Applies domain-specific knowledge to improve the design specification
   */
  private applyEngineeringReasoning(
    intent: DesignIntent,
    missingInfo: string[]
  ): string[] {
    const reasoning: string[] = [];

    // Apply geometric reasoning
    if (intent.primaryShape === 'cylinder') {
      reasoning.push(
        'Cylinder detected: Ensuring diameter and height are specified'
      );
      
      // Check for reasonable aspect ratio
      const diameter = intent.dimensions.get('diameter') || 0;
      const height = intent.dimensions.get('height') || 0;
      
      if (diameter > 0 && height > 0) {
        const aspectRatio = height / diameter;
        if (aspectRatio > 10) {
          reasoning.push(
            `High aspect ratio (${aspectRatio.toFixed(1)}:1) detected - consider adding support features`
          );
        }
      }
    }

    // Apply manufacturing reasoning
    if (intent.manufacturing === '3d printing' || intent.manufacturing === 'additive') {
      reasoning.push(
        'Additive manufacturing: Checking for overhangs and support structures'
      );
      
      if (intent.features.includes('overhang')) {
        reasoning.push(
          'Overhang detected: Recommending support structures or design modification'
        );
      }
    }

    // Apply material reasoning
    if (intent.materialRequirements?.toLowerCase().includes('steel')) {
      reasoning.push(
        'Steel material: Considering thermal expansion and machining allowances'
      );
    }

    // Apply tolerance reasoning
    if (intent.features.includes('threaded hole')) {
      if (!intent.tolerance) {
        reasoning.push(
          'Threaded feature requires tolerance specification - suggesting ISO general tolerance'
        );
      }
    }

    return reasoning;
  }

  /**
   * PATENTABLE COMPONENT 4: Prompt Reformulation
   * Rewrites the prompt in a standardized, unambiguous format optimized for CAD generation
   */
  private reformulatePrompt(
    original: string,
    intent: DesignIntent,
    missingInfo: string[]
  ): string {
    let optimized = '';

    // Start with clear shape definition
    optimized += `Create a ${intent.primaryShape}`;

    // Add dimensions in a structured way
    if (intent.dimensions.size > 0) {
      optimized += ' with the following dimensions: ';
      const dims: string[] = [];
      intent.dimensions.forEach((value, key) => {
        dims.push(`${key} = ${value}`);
      });
      optimized += dims.join(', ');
    }

    // Add features
    if (intent.features.length > 0) {
      optimized += '. Include these features: ' + intent.features.join(', ');
    }

    // Add material if specified
    if (intent.materialRequirements) {
      optimized += `. Material: ${intent.materialRequirements}`;
    }

    // Add manufacturing context
    if (intent.manufacturing) {
      optimized += `. Manufacturing method: ${intent.manufacturing}`;
    }

    // Add constraints
    if (intent.constraints.length > 0) {
      optimized += '. Constraints: ' + intent.constraints.join(', ');
    }

    // Add inferred defaults for missing information
    if (missingInfo.length > 0) {
      optimized += '. [Auto-inferred] ';
      missingInfo.forEach((info) => {
        if (info.includes('tolerance')) {
          optimized += 'Apply ISO 2768-m general tolerance. ';
        }
        if (info.includes('manufacturing') && !intent.manufacturing) {
          optimized += 'Assume CNC machining. ';
        }
      });
    }

    // Add units clarification
    optimized += ' All dimensions in millimeters unless specified otherwise.';

    return optimized;
  }

  /**
   * Generate suggestions to help the user improve their prompt
   */
  private generateSuggestions(
    intent: DesignIntent,
    missingInfo: string[]
  ): string[] {
    const suggestions: string[] = [];

    // Suggest dimension improvements
    if (intent.dimensions.size < 2) {
      suggestions.push(
        '💡 Add more specific dimensions for better accuracy'
      );
    }

    // Suggest feature additions
    if (intent.features.length === 0) {
      suggestions.push(
        '💡 Consider specifying features like holes, fillets, or chamfers'
      );
    }

    // Suggest material specification
    if (!intent.materialRequirements) {
      suggestions.push(
        '💡 Specify material type for accurate property simulation'
      );
    }

    // Suggest tolerance specification
    if (!intent.tolerance && intent.features.length > 0) {
      suggestions.push(
        '💡 Add tolerance specification for manufacturing feasibility'
      );
    }

    // Context-specific suggestions
    if (intent.confidence < 0.7) {
      suggestions.push(
        '⚠️ Prompt clarity is low - consider being more specific about your design'
      );
    }

    return suggestions;
  }

  // Helper methods for extraction

  private identifyPrimaryShape(prompt: string): string {
    const shapes = [
      'cylinder',
      'cube',
      'box',
      'sphere',
      'cone',
      'bracket',
      'plate',
      'shaft',
      'gear',
      'bearing',
      'flange',
    ];

    const lowerPrompt = prompt.toLowerCase();
    for (const shape of shapes) {
      if (lowerPrompt.includes(shape)) {
        return shape;
      }
    }

    return 'unknown';
  }

  private extractDimensions(prompt: string): Map<string, number> {
    const dimensions = new Map<string, number>();

    // Pattern: "diameter of 50mm" or "50mm diameter"
    const patterns = [
      /(\w+)\s+(?:of\s+)?(\d+\.?\d*)\s*(mm|cm|m|in|ft)/gi,
      /(\d+\.?\d*)\s*(mm|cm|m|in|ft)\s+(\w+)/gi,
    ];

    patterns.forEach((pattern) => {
      let match;
      while ((match = pattern.exec(prompt)) !== null) {
        const dimName = match[1] || match[3];
        const value = parseFloat(match[2] || match[1]);
        const unit = match[3] || match[2];

        dimensions.set(dimName.toLowerCase(), value);
      }
    });

    return dimensions;
  }

  private identifyFeatures(prompt: string): string[] {
    const features = [
      'hole',
      'thread',
      'chamfer',
      'fillet',
      'groove',
      'slot',
      'keyway',
      'counterbore',
      'countersink',
      'taper',
      'knurl',
    ];

    const lowerPrompt = prompt.toLowerCase();
    return features.filter((feature) => lowerPrompt.includes(feature));
  }

  private extractConstraints(prompt: string): string[] {
    const constraints: string[] = [];
    const lowerPrompt = prompt.toLowerCase();

    if (lowerPrompt.includes('tolerance')) constraints.push('tolerance specified');
    if (lowerPrompt.includes('fit')) constraints.push('fit requirement');
    if (lowerPrompt.includes('clearance')) constraints.push('clearance specified');
    if (lowerPrompt.includes('interference')) constraints.push('interference fit');
    if (lowerPrompt.includes('surface finish')) constraints.push('surface finish');

    return constraints;
  }

  private extractMaterial(prompt: string): string | undefined {
    const materials = [
      'steel',
      'aluminum',
      'brass',
      'copper',
      'plastic',
      'titanium',
      'stainless',
      'carbon fiber',
    ];

    const lowerPrompt = prompt.toLowerCase();
    for (const material of materials) {
      if (lowerPrompt.includes(material)) {
        return material;
      }
    }

    return undefined;
  }

  private extractManufacturingContext(prompt: string): string | undefined {
    const methods = [
      '3d printing',
      'cnc',
      'milling',
      'turning',
      'casting',
      'forging',
      'additive',
      'subtractive',
    ];

    const lowerPrompt = prompt.toLowerCase();
    for (const method of methods) {
      if (lowerPrompt.includes(method)) {
        return method;
      }
    }

    return undefined;
  }

  private getExpectedDimensions(shape: string): string[] {
    const dimMap: Record<string, string[]> = {
      cylinder: ['diameter', 'height'],
      cube: ['side', 'length'],
      box: ['length', 'width', 'height'],
      sphere: ['diameter', 'radius'],
      cone: ['diameter', 'height'],
      bracket: ['length', 'width', 'thickness'],
      shaft: ['diameter', 'length'],
      gear: ['pitch_diameter', 'teeth', 'module'],
    };

    return dimMap[shape] || ['length', 'width', 'height'];
  }

  private calculateConfidence(intent: DesignIntent): number {
    let score = 0;

    // Primary shape identified
    if (intent.primaryShape !== 'unknown') score += 0.3;

    // Dimensions specified
    if (intent.dimensions.size > 0) score += 0.2;
    if (intent.dimensions.size >= 3) score += 0.1;

    // Features specified
    if (intent.features.length > 0) score += 0.15;

    // Material specified
    if (intent.materialRequirements) score += 0.1;

    // Manufacturing context
    if (intent.manufacturing) score += 0.1;

    // Constraints specified
    if (intent.constraints.length > 0) score += 0.05;

    return Math.min(score, 1.0);
  }

  private loadEngineeringVocabulary(): Map<string, string[]> {
    // In production, load from a database or config file
    return new Map([
      ['fastener', ['bolt', 'screw', 'nut', 'washer']],
      ['bearing', ['ball bearing', 'roller bearing', 'sleeve']],
      ['connection', ['flange', 'coupling', 'joint']],
    ]);
  }

  private loadGeometricPatterns(): Map<string, RegExp> {
    return new Map([
      ['dimension', /(\d+\.?\d*)\s*(mm|cm|m|in|ft)/gi],
      ['angle', /(\d+\.?\d*)\s*(?:degrees?|deg|°)/gi],
    ]);
  }
}

export { PromptOptimizationLayer, type OptimizedPrompt, type DesignIntent };