import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from '@google/genai';
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PromptOptimizationLayer, type OptimizedPrompt } from './src/prompt-optimizer';
import { GPUCFDSolver, type CFDConfig, type SimulationResult } from './src/cfd-simulator';
import CFDVisualizationComponent from './CFDVisualizationComponent';
import { CADDownloadManager, type CADModel } from './src/cad-download-manager';

const systemPrompt = `
System Role:
You are KelmoidAI_Genesis_llm, an advanced Text-to-CAD generative large language model designed to transform natural language descriptions into CAD models and simulation setups. You understand mechanical design, architecture, civil structures, and advanced geometric modeling, including time-varying (4D) components and Computational Fluid Dynamics (CFD).

Mission:
Convert human language prompts into structured CAD instructions, scripts, or files suitable for direct rendering or simulation in engineering and architectural software environments. You output valid and optimized CAD code (e.g., OpenSCAD, FreeCAD, Fusion360, Blender, or STEP representations) and CFD setup files.

Core Capabilities
Input:
Natural language text describing objects, shapes, or assemblies.
Optional parameters like dimensions, materials, or motion sequences.
Optional CFD parameters like fluid properties, boundary conditions, and flow velocity.

Output:
CAD model definitions in OpenSCAD, STEP, STL, or OBJ formats.
For 2D: DXF or SVG geometry.
For 3D: STL, OBJ, or STEP geometry.
- When the requested \`export_format\` is 'STL', the \`cad_script\` field in your JSON output MUST contain the full, valid text content of the .stl file. Do not provide OpenSCAD code in this case.
For 4D: Animated transformations or time-dependent model states. You can specify the animation timing or 'easing' using terms like 'ease-in-out', 'linear', 'start slow and end fast', etc. This will be reflected in the 'interpolation_type' field.
For CFD: A definition of the geometry, mesh, fluid properties, and boundary conditions.

Domains:
Mechanical parts (gears, pistons, brackets, casings)
Architectural structures (buildings, bridges, interiors)
Civil engineering (roads, pipelines, tunnels)
Product design (tools, enclosures, prototypes)
Scientific models (molecular, anatomical, or spatial geometries)
CFD Simulation (airflow, water flow, heat transfer over geometries)

4D (Time-Varying) Models:
You have advanced support for 4D designs. Specify temporal transformations through parametric animation or keyframes.
- Supported Transformations: rotation, deformation, scaling, translation over time.
- Output Structure: For 4D models, set "model_type" to "4D" and include an "animation_data" object describing the transformations.

CFD (Computational Fluid Dynamics) Models:
You can define CFD simulations. The user will specify the geometry, fluid properties, and boundary conditions.
- Supported Concepts: Inlets, outlets, walls (no-slip, slip), velocity, pressure, density, viscosity.
- Advanced CFD Concepts: You now understand and can implement advanced CFD parameters including:
  - Turbulence Models: RANS models like k-epsilon, k-omega SST, and Spalart-Allmaras. Specify the model in the user prompt.
  - Wall Functions: Standard, scalable, or non-equilibrium wall functions for near-wall turbulence modeling.
  - Inlet/Outlet Profiles: You can define non-uniform profiles for velocity, temperature, etc., using mathematical expressions or descriptive language (e.g., 'parabolic velocity profile').
- Output Structure: For CFD models, set "model_type" to "CFD" and include a "simulation_data" object describing the setup. The simulation_data should reflect the advanced parameters requested.

Output Format
You must output a single JSON object with the appropriate structure. For 4D, populate the animation_data field. For CFD, populate simulation_data.

// 4D Example
{
  "model_type": "4D",
  "domain": "mechanical",
  "description": "A gear with 20 teeth, rotating 360 degrees over 5 seconds with ease-in-out timing.",
  "cad_script": "...OpenSCAD or FreeCAD code with animation parameters...",
  "export_format": "Animated GIF/MP4 (conceptual)",
  "animation_data": {
    "type": "rotation",
    "axis": "z",
    "duration_seconds": 5,
    "start_angle_deg": 0,
    "end_angle_deg": 360,
    "interpolation_type": "ease-in-out"
  },
  "metadata": { "units": "mm", "version": "1.0", "timestamp": "..." }
}

// CFD Example
{
  "model_type": "CFD",
  "domain": "fluid_dynamics",
  "description": "Simulation of airflow over a cylinder at 10 m/s using k-epsilon turbulence model.",
  "cad_script": "cylinder(d=0.1, h=1);",
  "export_format": "CFD",
  "simulation_data": {
    "solver": "OpenFOAM (conceptual)",
    "simulation_type": "incompressible_flow",
    "turbulence_model": "k-epsilon",
    "fluid_properties": {
      "name": "Air",
      "density_kg_m3": 1.225,
      "viscosity_pa_s": 0.0000181
    },
    "boundary_conditions": [
      { "name": "inlet", "type": "velocity", "value": "10 m/s", "direction": "x" },
      { "name": "outlet", "type": "pressure", "value": "0 Pa" },
      { "name": "cylinder_wall", "type": "no-slip_wall", "wall_function": "standard" },
      { "name": "domain_walls", "type": "slip_wall" }
    ]
  },
  "metadata": { "units": "m", "version": "1.0", "timestamp": "..." }
}


Core Behavior Rules
- Always output valid and executable CAD code or simulation setups.
- All dimensional units in the user prompt have been automatically converted to a consistent system (either metric 'mm' or imperial 'inches'). Use these units directly.
- The "units" field in your output metadata must reflect the unit system used in the prompt (e.g., "mm" for metric, "in" for imperial).
- Optimize geometry for manufacturability and render performance.
- When uncertain, ask clarifying questions about missing parameters.
- For 4D objects, define motion through parametric animation or keyframes.
- For CFD, define the fluid, boundaries, and initial conditions.
- Your output must be only the JSON object, without any markdown formatting like \`\`\`json.
`;

const PromptHelperModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text).then(() => {
    }).catch(err => console.error('Failed to copy text: ', err));
  };

  const examples = {
    mechanical: 'A planetary gear system with a sun gear of 20 teeth, 3 planet gears of 10 teeth each, and a ring gear of 40 teeth. The carrier should connect the planet gears. Overall diameter is 150mm.',
    architectural: 'A modern two-story house with a flat roof, large glass windows on the south facade, and a wooden deck. Dimensions are 15m wide by 10m deep.',
    fourD: 'A cube that deforms into a sphere over 3 seconds, then returns to a cube in the next 3 seconds.',
    dimensions: 'A hollow cylinder with an outer diameter of 50mm, an inner diameter of 40mm, and a height of 100mm.',
    exportFormatExample: 'Create a simple cube with 10mm sides and export it as an STL file ready for 3D printing.',
    materialSimple: 'Material: 6061-T6 Aluminum',
    materialStructural: "Material: Structural Steel. Young's Modulus: 200 GPa, Poisson's Ratio: 0.3, Density: 7850 kg/m^3.",
    materialThermal: "Material: Aluminum for a heatsink. Thermal Conductivity: 205 W/mK, Density: 2700 kg/m^3, Specific Heat: 900 J/kg·K.",
    fourDParams: 'rotation on Y-axis, 720 degrees over 10 seconds, ease-in-out timing',
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Prompt Helper</h2>
          <button className="modal-close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <p>Craft effective prompts by being specific and clear. Here are some tips and examples:</p>
          
          <div className="help-section">
            <h3>Be Specific with Dimensions</h3>
            <p>Always include units like mm, cm, m, inches, or feet. The more precise you are, the better the result.</p>
            <div className="example">
              <code>{examples.dimensions}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.dimensions)}>Copy</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const PromptHistoryModal = ({ isOpen, onClose, history, onLoad, onClear }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Prompt History</h2>
          <button className="modal-close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          {history.length > 0 ? (
            <>
              <div className="history-actions">
                  <button className="clear-history-button" onClick={onClear}>Clear All History</button>
              </div>
              <ul className="history-list">
                {history.map((item) => (
                  <li key={item.id} className="history-item">
                    <p className="history-item-prompt" title={item.prompt}>
                      {item.prompt}
                    </p>
                    <div className="history-item-actions">
                       <button className="load-button" onClick={() => onLoad(item)}>Load</button>
                    </div>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p>No prompt history found. Your generated prompts will appear here.</p>
          )}
        </div>
      </div>
    </div>
  );
};

const CONVERSION_FACTORS = {
    'in_to_mm': 25.4,
    'ft_to_mm': 304.8,
    'm_to_mm': 1000,
    'cm_to_mm': 10,
    'mm_to_in': 1 / 25.4,
    'm_to_in': 1000 / 25.4,
    'cm_to_in': 10 / 25.4,
};

const unitPatterns = [
    { name: 'in', regex: new RegExp(`\\b(\\d*\\.?\\d+)\\s*(?:inch|inches|in|")`, 'gi') },
    { name: 'ft', regex: new RegExp(`\\b(\\d*\\.?\\d+)\\s*(?:foot|feet|ft|')`, 'gi') },
    { name: 'm', regex: new RegExp(`\\b(\\d*\\.?\\d+)\\s*(?:meter|meters|m)`, 'gi') },
    { name: 'cm', regex: new RegExp(`\\b(\\d*\\.?\\d+)\\s*(?:centimeter|centimeters|cm)`, 'gi') },
    { name: 'mm', regex: new RegExp(`\\b(\\d*\\.?\\d+)\\s*(?:millimeter|millimeters|mm)`, 'gi') },
];

const convertUnits = (prompt, targetSystem) => {
    let convertedPrompt = prompt;
    let highlightedPrompt = prompt;
    const logEntries = new Set<string>();

    const targetUnit = targetSystem === 'Metric' ? 'mm' : 'in';

    unitPatterns.forEach(({ name, regex }) => {
        regex.lastIndex = 0;
        let match;
        while ((match = regex.exec(prompt)) !== null) {
            const originalMatch = match[0];
            const value = parseFloat(match[1]);
            let convertedValue;

            if (targetSystem === 'Metric') {
                if (name === 'in') convertedValue = value * CONVERSION_FACTORS.in_to_mm;
                else if (name === 'ft') convertedValue = value * CONVERSION_FACTORS.ft_to_mm;
                else if (name === 'm') convertedValue = value * CONVERSION_FACTORS.m_to_mm;
                else if (name === 'cm') convertedValue = value * CONVERSION_FACTORS.cm_to_mm;
                else continue;
            } else {
                if (name === 'mm') convertedValue = value * CONVERSION_FACTORS.mm_to_in;
                else if (name === 'cm') convertedValue = value * CONVERSION_FACTORS.cm_to_in;
                else if (name === 'm') convertedValue = value * CONVERSION_FACTORS.m_to_in;
                else continue;
            }
            
            if (convertedValue !== undefined) {
                 const roundedValue = Math.round(convertedValue * 100) / 100;
                 const replacement = `${roundedValue} ${targetUnit}`;
                 logEntries.add(`'${originalMatch}' -> '${replacement}'`);
            }
        }
    });
    
    const downloadManager = useRef(new CADDownloadManager());

    const escapeRegExp = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    logEntries.forEach(logEntry => {
        const [original, replacement] = logEntry.split(' -> ');
        const originalText = original.slice(1,-1);
        const replacementText = replacement.slice(1,-1);
        
        const searchRegex = new RegExp(escapeRegExp(originalText), 'gi');
        
        convertedPrompt = convertedPrompt.replace(searchRegex, replacementText);
        
        const highlightedReplacement = `<span class="unit-highlight" title="Original: ${originalText}">${replacementText}</span>`;
        highlightedPrompt = highlightedPrompt.replace(searchRegex, highlightedReplacement);
    });

    return {
        convertedPrompt,
        log: Array.from(logEntries).join(', '),
        highlightedPrompt,
    };
}

const EasingFunctions = {
  linear: t => t,
  'ease-in': t => t * t,
  'ease-out': t => t * (2 - t),
  'ease-in-out': t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
};

const ModelViewer = ({ stlData, animationData }) => {
    const mountRef = useRef(null);
    const animationState = useRef({ isPlaying: false, clock: new THREE.Clock(), initial: {} });

    const handleAnimateClick = () => {
        animationState.current.isPlaying = true;
        animationState.current.clock.start();
    };

    useEffect(() => {
        if (!stlData || !mountRef.current) return;

        const currentMount = mountRef.current;
        let animationFrameId;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x18181b);
        const camera = new THREE.PerspectiveCamera(75, currentMount.clientWidth / currentMount.clientHeight, 0.1, 1000);
        camera.position.z = 100;
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
        currentMount.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);
        const pointLight1 = new THREE.PointLight(0xffffff, 0.8);
        pointLight1.position.set(100, 100, 100);
        scene.add(pointLight1);
        const pointLight2 = new THREE.PointLight(0xffffff, 0.5);
        pointLight2.position.set(-100, -100, -100);
        scene.add(pointLight2);

        const loader = new STLLoader();
        const geometry = loader.parse(stlData);
        const material = new THREE.MeshStandardMaterial({
            color: 0xff8a00, metalness: 0.3, roughness: 0.6
        });
        const mesh = new THREE.Mesh(geometry, material);
        
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        mesh.position.sub(center);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 100 / maxDim;
        mesh.scale.set(scale, scale, scale);
        scene.add(mesh);
        camera.lookAt(mesh.position);
        
        animationState.current.initial = {
            position: mesh.position.clone(),
            rotation: mesh.rotation.clone(),
            scale: mesh.scale.clone()
        };

        const handleResize = () => {
            if (currentMount) {
                camera.aspect = currentMount.clientWidth / currentMount.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
            }
        };
        window.addEventListener('resize', handleResize);

        const animate = () => {
            animationFrameId = requestAnimationFrame(animate);

            if (animationState.current.isPlaying && animationData) {
                const elapsedTime = animationState.current.clock.getElapsedTime();
                const duration = animationData.duration_seconds || 5;
                let progress = elapsedTime / duration;
                
                if (progress >= 1) {
                    progress = 1;
                    animationState.current.isPlaying = false;
                    animationState.current.clock.stop();
                }

                const easingFunction = EasingFunctions[animationData.interpolation_type] || EasingFunctions.linear;
                const easedProgress = easingFunction(progress);

                if (animationData.type === 'rotation') {
                    const startAngle = (animationData.start_angle_deg || 0) * (Math.PI / 180);
                    const endAngle = (animationData.end_angle_deg || 0) * (Math.PI / 180);
                    const angle = startAngle + (endAngle - startAngle) * easedProgress;
                    const axis = animationData.axis || 'z';
                    if (axis === 'x') mesh.rotation.x = animationState.current.initial.rotation.x + angle;
                    if (axis === 'y') mesh.rotation.y = animationState.current.initial.rotation.y + angle;
                    if (axis === 'z') mesh.rotation.z = animationState.current.initial.rotation.z + angle;
                }

                if (!animationState.current.isPlaying) {
                    setTimeout(() => {
                        if (mesh) {
                           mesh.position.copy(animationState.current.initial.position);
                           mesh.rotation.copy(animationState.current.initial.rotation);
                           mesh.scale.copy(animationState.current.initial.scale);
                        }
                    }, 1000);
                }
            }
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            cancelAnimationFrame(animationFrameId);
            window.removeEventListener('resize', handleResize);
            if (currentMount && renderer.domElement) {
                currentMount.removeChild(renderer.domElement);
            }
            controls.dispose();
            renderer.dispose();
            geometry.dispose();
            material.dispose();
        };
    }, [stlData, animationData]);

    return (
        <div className="viewer-wrapper">
            <div ref={mountRef} className="model-viewer-container" />
            {animationData && (
                <button className="animate-button" onClick={handleAnimateClick}>
                    Animate
                </button>
            )}
        </div>
    );
};

const initialCfdState = {
  isCfdEnabled: false,
  fluidName: 'Air',
  density: '1.225',
  viscosity: '1.81e-5',
  turbulenceModel: 'k-epsilon',
  inletType: 'Velocity',
  inletValue: '10',
  inletDirection: 'X',
  outletType: 'Pressure',
  outletValue: '0',
  wallType: 'no-slip',
};

const initialFeaState = {
  isFeaEnabled: false,
  materialName: 'Structural Steel',
  youngsModulus: '200',
  poissonsRatio: '0.3',
  density: '7850',
};

const App = () => {
  const [prompt, setPrompt] = useState('Generate a 3D CAD model of a mechanical gear with 20 teeth, a 10mm bore, and a 2 inch outer diameter.');
  const [materialParams, setMaterialParams] = useState('');
  const [fourDParams, setFourDParams] = useState('');
  const [animationInterpolation, setAnimationInterpolation] = useState('ease-in-out');
  const [cfdInputs, setCfdInputs] = useState(initialCfdState);
  const [feaInputs, setFeaInputs] = useState(initialFeaState);
  const [output, setOutput] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [stlData, setStlData] = useState('');
  const [animationData, setAnimationData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [error, setError] = useState('');
  const [exportFormat, setExportFormat] = useState('STL');
  const [aspectRatio, setAspectRatio] = useState('4:3');
  const [isHelpModalOpen, setIsHelpModalOpen] = useState(false);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [promptHistory, setPromptHistory] = useState([]);
  const [defaultUnitSystem, setDefaultUnitSystem] = useState('Metric');
  const [unitConversionLog, setUnitConversionLog] = useState('');
  const [convertedPromptPreview, setConvertedPromptPreview] = useState<string | null>(null);

  const [optimizedPromptData, setOptimizedPromptData] = useState<OptimizedPrompt | null>(null);
  const [cfdResults, setCfdResults] = useState<SimulationResult | null>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [showOptimizationPanel, setShowOptimizationPanel] = useState(false);

  const promptOptimizerRef = useRef(new PromptOptimizationLayer());
  const cfdSolverRef = useRef(new GPUCFDSolver());

  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('promptHistory');
      if (savedHistory) setPromptHistory(JSON.parse(savedHistory));
    } catch (error) { console.error("Could not load prompt history", error); }
    
    try {
      const savedState = localStorage.getItem('currentPromptState');
      if (savedState) {
        const parsedState = JSON.parse(savedState);
        if (parsedState.prompt !== undefined) setPrompt(parsedState.prompt);
        if (parsedState.materialParams !== undefined) setMaterialParams(parsedState.materialParams);
        if (parsedState.feaInputs !== undefined) setFeaInputs(parsedState.feaInputs);
        if (parsedState.fourDParams !== undefined) setFourDParams(parsedState.fourDParams);
        if (parsedState.animationInterpolation !== undefined) setAnimationInterpolation(parsedState.animationInterpolation);
        if (parsedState.cfdInputs !== undefined) setCfdInputs(parsedState.cfdInputs);
        if (parsedState.exportFormat !== undefined) setExportFormat(parsedState.exportFormat);
        if (parsedState.aspectRatio !== undefined) setAspectRatio(parsedState.aspectRatio);
        if (parsedState.defaultUnitSystem !== undefined) setDefaultUnitSystem(parsedState.defaultUnitSystem);
      }
    } catch (error) { console.error("Could not load prompt state", error); }
  }, []);

  useEffect(() => {
    try {
        const currentState = {
            prompt, materialParams, feaInputs, fourDParams, animationInterpolation, cfdInputs, exportFormat, aspectRatio, defaultUnitSystem,
        };
        localStorage.setItem('currentPromptState', JSON.stringify(currentState));
    } catch (error) { console.error("Could not save prompt state", error); }
  }, [prompt, materialParams, feaInputs, fourDParams, animationInterpolation, cfdInputs, exportFormat, aspectRatio, defaultUnitSystem]);

  useEffect(() => {
    const { highlightedPrompt, log } = convertUnits(prompt, defaultUnitSystem);
    if (log) {
        setConvertedPromptPreview(highlightedPrompt);
    } else {
        setConvertedPromptPreview(null);
    }
  }, [prompt, defaultUnitSystem]);

  const handleCfdInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (type === 'checkbox') {
        setCfdInputs(prev => ({...prev, [name]: checked}));
    } else {
        setCfdInputs(prev => ({...prev, [name]: value}));
    }
  };

  const handleFeaInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (type === 'checkbox') {
        setFeaInputs(prev => ({...prev, [name]: checked}));
    } else {
        setFeaInputs(prev => ({...prev, [name]: value}));
    }
  };

  const handleLoadFeaPreset = (preset) => {
    if (preset === 'steel') {
        setFeaInputs(prev => ({ ...prev, materialName: 'Structural Steel', youngsModulus: '200', poissonsRatio: '0.3', density: '7850' }));
    } else if (preset === 'aluminum') {
        setFeaInputs(prev => ({ ...prev, materialName: '6061-T6 Aluminum', youngsModulus: '69', poissonsRatio: '0.33', density: '2700' }));
    }
  };

  const handleOptimizePrompt = async () => {
    if (!prompt.trim()) return;

    setIsOptimizing(true);
    setShowOptimizationPanel(true);

    try {
      const optimized = await promptOptimizerRef.current.optimizePrompt(prompt);
      setOptimizedPromptData(optimized);

      console.log('🎯 Design Intent Extracted:', optimized.designIntent);
      console.log('🧠 AI Reasoning:', optimized.reasoning);
      console.log('✨ Optimized Prompt:', optimized.optimized);

    } catch (error) {
      console.error('Prompt optimization error:', error);
      setError(`Optimization failed: ${error.message}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  const runCFDSimulation = async () => {
    console.log('🌊 Starting GPU-accelerated CFD simulation...');
    setLoadingStatus('Running CFD simulation...');

    const cfdConfig: CFDConfig = {
      gridSize: { x: 30, y: 30, z: 30 },
      fluidProperties: {
        density: parseFloat(cfdInputs.density),
        viscosity: parseFloat(cfdInputs.viscosity),
      },
      boundaryConditions: [
        {
          type: 'velocity',
          location: 'inlet',
          value: { 
            x: cfdInputs.inletDirection === 'X' ? parseFloat(cfdInputs.inletValue) : 0,
            y: cfdInputs.inletDirection === 'Y' ? parseFloat(cfdInputs.inletValue) : 0,
            z: cfdInputs.inletDirection === 'Z' ? parseFloat(cfdInputs.inletValue) : 0
          },
        },
        {
          type: 'pressure',
          location: 'outlet',
          value: parseFloat(cfdInputs.outletValue),
        },
        {
          type: 'wall',
          location: 'boundaries',
          value: 0,
        },
      ],
      timeStep: 0.01,
      iterations: 50,
    };

    try {
      const results = await cfdSolverRef.current.simulate(cfdConfig);
      setCfdResults(results);

      console.log(`✅ CFD completed in ${results.computeTime.toFixed(2)}ms`);
      console.log(`📊 ${results.message}`);

    } catch (error) {
      console.error('CFD simulation error:', error);
      setError(`CFD simulation failed: ${error.message}`);
    }
  };

const handleDownloadCAD = (format: string) => {
  if (!output) {
    setError('No CAD model to download. Please generate a model first.');
    return;
  }

  try {
    // Parse the output JSON
    const jsonOutput = JSON.parse(output);
    
    // Create CAD model object
    const cadModel: CADModel = {
      script: jsonOutput.cad_script || '',
      format: format as any,
      metadata: {
        modelName: jsonOutput.description || 'kelmoid_model',
        description: jsonOutput.description || 'AI Generated CAD Model',
        units: jsonOutput.metadata?.units || 'mm',
        timestamp: new Date().toISOString()
      }
    };

    // Validate before download
    const validation = downloadManager.current.validateCAD(cadModel);
    if (!validation.valid) {
      setError(`Cannot download: ${validation.errors.join(', ')}`);
      return;
    }

    // Download the file
    downloadManager.current.downloadCAD(cadModel);
    
    console.log(`✅ Downloaded ${format} file successfully`);
    
  } catch (error) {
    console.error('Download error:', error);
    setError(`Failed to download CAD file: ${error.message}`);
  }
};

// 4. ADD MULTI-FORMAT DOWNLOAD HANDLER:
const handleDownloadMultiFormat = () => {
  if (!output) {
    setError('No CAD model to download. Please generate a model first.');
    return;
  }

  try {
    const jsonOutput = JSON.parse(output);
    
    const cadModel: CADModel = {
      script: jsonOutput.cad_script || '',
      format: 'STL' as any,
      metadata: {
        modelName: jsonOutput.description || 'kelmoid_model',
        description: jsonOutput.description || 'AI Generated CAD Model',
        units: jsonOutput.metadata?.units || 'mm',
        timestamp: new Date().toISOString()
      }
    };

    // Download in multiple formats
    const formats = ['STL', 'OpenSCAD', 'OBJ'];
    downloadManager.current.downloadMultiFormat(cadModel, formats);
    
    console.log(`✅ Downloading ${formats.length} formats...`);
    
  } catch (error) {
    console.error('Multi-format download error:', error);
    setError(`Failed to download files: ${error.message}`);
  }
};

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a description for the CAD model.');
      return;
    }

    const finalPrompt = optimizedPromptData?.optimized || prompt;

    setIsLoading(true);
    setError('');
    setOutput('');
    setImageUrl('');
    setStlData('');
    setAnimationData(null);
    setUnitConversionLog('');
    setCfdResults(null);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const { convertedPrompt, log } = convertUnits(finalPrompt, defaultUnitSystem);
      if (log) setUnitConversionLog(log);
      
      const buildMaterialString = () => {
        if (feaInputs.isFeaEnabled) {
            let feaString = '\nMaterial Properties: ';
            if (feaInputs.materialName) feaString += `${feaInputs.materialName}. `;
            if (feaInputs.youngsModulus) feaString += `Young's Modulus: ${feaInputs.youngsModulus} GPa. `;
            if (feaInputs.poissonsRatio) feaString += `Poisson's Ratio: ${feaInputs.poissonsRatio}. `;
            if (feaInputs.density) feaString += `Density: ${feaInputs.density} kg/m^3.`;
            if (feaString.trim().length > 20) {
                return feaString;
            }
            return '';
        } else {
            return materialParams.trim() ? `\nMaterial Properties: ${materialParams}` : '';
        }
      };

      const buildCfdString = () => {
          if (!cfdInputs.isCfdEnabled) return '';
          let cfdString = '\nCFD Parameters: ';
          cfdString += `Fluid is ${cfdInputs.fluidName} (density: ${cfdInputs.density} kg/m^3, viscosity: ${cfdInputs.viscosity} Pa.s). `;
          cfdString += `Turbulence Model: ${cfdInputs.turbulenceModel === 'None (Laminar)' ? 'Laminar flow' : cfdInputs.turbulenceModel}. `;
          let inletDescription = `Inlet is a ${cfdInputs.inletType} type with a value of ${cfdInputs.inletValue}`;
          if (cfdInputs.inletType === 'Velocity') {
              inletDescription += ` m/s in the ${cfdInputs.inletDirection}-direction. `;
          } else {
              inletDescription += ` Pa. `;
          }
          let outletDescription = `Outlet is a ${cfdInputs.outletType} type`;
          if (cfdInputs.outletType === 'Pressure') {
              outletDescription += ` with a value of ${cfdInputs.outletValue} Pa. `;
          } else {
              outletDescription += `. `;
          }
          cfdString += `Boundary Conditions: ${inletDescription}${outletDescription}`;
          cfdString += `The walls are ${cfdInputs.wallType}.`;
          return cfdString;
      }

      const fullPrompt = `${convertedPrompt}
${buildMaterialString()}
${fourDParams.trim() ? `\n4D Parameters: ${fourDParams}\nAnimation Interpolation: ${animationInterpolation}` : ''}
${buildCfdString()}
\nExport format: ${exportFormat}
\nDefault Unit System: ${defaultUnitSystem === 'Metric' ? 'mm' : 'in'}`;

      setLoadingStatus('Generating CAD script...');
      const scriptResponse = await ai.models.generateContent({
        model: 'gemini-2.5-pro',
        contents: `${systemPrompt}\n\nUser Prompt: ${fullPrompt}`,
      });

      const text = scriptResponse.text;
      let jsonOutput;
      try {
        const cleanedText = text.replace(/^```json\s*|```\s*$/g, '');
        jsonOutput = JSON.parse(cleanedText);
        setOutput(JSON.stringify(jsonOutput, null, 2));

        if (jsonOutput.export_format === 'STL' && jsonOutput.cad_script) {
            setStlData(jsonOutput.cad_script);
        }
        if (jsonOutput.model_type === '4D' && jsonOutput.animation_data) {
            setAnimationData(jsonOutput.animation_data);
        }

      } catch (e) {
        console.error("Failed to parse JSON response:", e, "Raw response:", text);
        setError(`Failed to parse the model's response. The response was not valid JSON. Raw response: ${text}`);
        setIsLoading(false);
        return;
      }
      
      const newHistoryItem = { 
        id: Date.now(), prompt, materialParams, feaInputs, fourDParams, animationInterpolation, cfdInputs, exportFormat, aspectRatio, defaultUnitSystem,
      };
      setPromptHistory(prev => {
        const newHistory = [newHistoryItem, ...prev.slice(0, 49)];
        localStorage.setItem('promptHistory', JSON.stringify(newHistory));
        return newHistory;
      });

      if (cfdInputs.isCfdEnabled) {
        await runCFDSimulation();
      }

      setLoadingStatus('Generating photorealistic render...');
      const renderPrompt = `Create a high-quality, photorealistic render of the following object, described by this prompt: "${jsonOutput.description}". The object should be displayed in a clean, well-lit studio environment that highlights its form and materials. Aspect Ratio: ${aspectRatio}.`;
      
      const imageResponse = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt: renderPrompt,
        config: { numberOfImages: 1, aspectRatio: aspectRatio },
      });

      if (imageResponse.generatedImages && imageResponse.generatedImages.length > 0) {
        const base64Image = imageResponse.generatedImages[0].image.imageBytes;
        setImageUrl(`data:image/png;base64,${base64Image}`);
      } else {
        setError('Could not generate the image render.');
      }

    } catch (error) {
      console.error(error);
      setError(`An error occurred: ${error.message}`);
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };
  
  const handleClear = () => {
    setPrompt('');
    setMaterialParams('');
    setFeaInputs(initialFeaState);
    setFourDParams('');
    setCfdInputs(initialCfdState);
    setOutput('');
    setImageUrl('');
    setStlData('');
    setAnimationData(null);
    setError('');
    setUnitConversionLog('');
    setConvertedPromptPreview(null);
    setOptimizedPromptData(null);
    setCfdResults(null);
    setShowOptimizationPanel(false);
  };
  
  const handleLoadFromHistory = (item) => {
    setPrompt(item.prompt || '');
    setMaterialParams(item.materialParams || '');
    setFeaInputs(item.feaInputs || initialFeaState);
    setFourDParams(item.fourDParams || '');
    setAnimationInterpolation(item.animationInterpolation || 'ease-in-out');
    setCfdInputs(item.cfdInputs || initialCfdState);
    setExportFormat(item.exportFormat || 'STL');
    setAspectRatio(item.aspectRatio || '4:3');
    setDefaultUnitSystem(item.defaultUnitSystem || 'Metric');
    setIsHistoryModalOpen(false);
  };

  const handleClearHistory = () => {
    setPromptHistory([]);
    localStorage.removeItem('promptHistory');
  };

  const handleDownload = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const OptimizationPanel = () => {
    if (!showOptimizationPanel || !optimizedPromptData) return null;

    return (
      <div className="card" style={{ marginTop: '2rem', border: '2px solid #ff8a00' }}>
        <h3 style={{ color: '#ff8a00' }}>🎯 AI Design Intention Analysis</h3>
        
        <div style={{ marginTop: '1rem' }}>
          <h4>Detected Design Intent:</h4>
          <ul style={{ marginLeft: '1.5rem' }}>
            <li>Primary Shape: <strong>{optimizedPromptData.designIntent.primaryShape}</strong></li>
            <li>Confidence: <strong>{(optimizedPromptData.designIntent.confidence * 100).toFixed(0)}%</strong></li>
            {optimizedPromptData.designIntent.dimensions.size > 0 && (
              <li>
                Dimensions: {Array.from(optimizedPromptData.designIntent.dimensions.entries()).map(
                  ([key, val]) => `${key}: ${val}`
                ).join(', ')}
              </li>
            )}
            {optimizedPromptData.designIntent.features.length > 0 && (
              <li>Features: {optimizedPromptData.designIntent.features.join(', ')}</li>
            )}
            {optimizedPromptData.designIntent.materialRequirements && (
              <li>Material: {optimizedPromptData.designIntent.materialRequirements}</li>
            )}
            {optimizedPromptData.designIntent.manufacturing && (
              <li>Manufacturing: {optimizedPromptData.designIntent.manufacturing}</li>
            )}
          </ul>
        </div>

        {optimizedPromptData.reasoning.length > 0 && (
          <div style={{ marginTop: '1rem' }}>
            <h4>🧠 AI Engineering Reasoning:</h4>
            <ul style={{ marginLeft: '1.5rem' }}>
              {optimizedPromptData.reasoning.map((reason, idx) => (
                <li key={idx}>{reason}</li>
              ))}
            </ul>
          </div>
        )}

        <div style={{ marginTop: '1rem' }}>
          <h4>✨ Optimized Prompt:</h4>
          <div className="prompt-preview" style={{ backgroundColor: 'rgba(255, 138, 0, 0.1)' }}>
            {optimizedPromptData.optimized}
          </div>
        </div>

        {optimizedPromptData.suggestedImprovements.length > 0 && (
          <div style={{ marginTop: '1rem' }}>
            <h4>💡 Suggestions:</h4>
            <ul style={{ marginLeft: '1.5rem' }}>
              {optimizedPromptData.suggestedImprovements.map((suggestion, idx) => (
                <li key={idx}>{suggestion}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="button-group" style={{ marginTop: '1rem' }}>
          <button 
            className="clear-button"
            onClick={() => setShowOptimizationPanel(false)}
          >
            Close
          </button>
          <button 
            className="generate-button"
            onClick={() => {
              setPrompt(optimizedPromptData.optimized);
              setShowOptimizationPanel(false);
            }}
          >
            Use Optimized Prompt
          </button>
        </div>
      </div>
    );
  };

  const CFDResultsPanel = () => {
  if (!cfdResults) return null;

  return (
    <div className="card" style={{ marginTop: '2rem', border: '2px solid #00a8ff' }}>
      <h3 style={{ color: '#00a8ff' }}>🌊 CFD Simulation Results</h3>
      
      {/* Stats */}
      <div style={{ marginTop: '1rem' }}>
        <p><strong>Status:</strong> {cfdResults.success ? '✅ Completed' : '❌ Failed'}</p>
        <p><strong>Message:</strong> {cfdResults.message}</p>
        <p><strong>Computation Time:</strong> {cfdResults.computeTime.toFixed(2)} ms</p>
        <p><strong>Iterations:</strong> {cfdResults.convergenceHistory.length}</p>
        {cfdResults.convergenceHistory.length > 0 && (
          <p><strong>Final Residual:</strong> {cfdResults.convergenceHistory[cfdResults.convergenceHistory.length - 1]?.toExponential(2)}</p>
        )}
      </div>

      {/* NEW: Interactive 3D Visualization */}
      <div style={{ marginTop: '2rem' }}>
        <h4>Interactive Flow Visualization</h4>
        <CFDVisualizationComponent 
          results={cfdResults} 
          config={{
            gridSize: { x: 30, y: 30, z: 30 },
            fluidProperties: {
              density: parseFloat(cfdInputs.density),
              viscosity: parseFloat(cfdInputs.viscosity),
            },
            boundaryConditions: [],
            timeStep: 0.01,
            iterations: 50,
          }}
        />
      </div>
    </div>
  );
};
  return (
    <div className="container">
      <header>
        <h1>KelmoidAI Genesis <sup>llm</sup></h1>
        <p>Your Advanced Text-to-CAD and Simulation Assistant</p>
      </header>

      <PromptHelperModal isOpen={isHelpModalOpen} onClose={() => setIsHelpModalOpen(false)} />
      <PromptHistoryModal 
        isOpen={isHistoryModalOpen} 
        onClose={() => setIsHistoryModalOpen(false)} 
        history={promptHistory}
        onLoad={handleLoadFromHistory}
        onClear={handleClearHistory}
      />
      
      <div className="card">
          {error && <div className="error-message">{error}</div>}
          {unitConversionLog && (
            <div className="conversion-log">
                <p><strong>Unit Conversions Applied:</strong> {unitConversionLog}</p>
            </div>
          )}
          <div className="form-group">
            <div className="label-group">
              <label htmlFor="prompt-input">Primary Prompt</label>
              <div className="label-buttons">
                 <button className="prompt-helper-button" onClick={() => setIsHistoryModalOpen(true)}>History</button>
                 <button className="prompt-helper-button" onClick={() => setIsHelpModalOpen(true)}>Prompt Helper</button>
              </div>
            </div>
            <textarea
              id="prompt-input"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., A 3D model of a standard 1/4-20 hex nut..."
              disabled={isLoading}
            />
          </div>
          
          {convertedPromptPreview && (
            <div className="prompt-preview" dangerouslySetInnerHTML={{ __html: `<strong>Preview:</strong> ${convertedPromptPreview}` }} />
          )}

          <div className="form-group">
            <label htmlFor="material-params-input">Material Properties (Simple / Unstructured)</label>
            <textarea
              id="material-params-input"
              value={materialParams}
              onChange={(e) => setMaterialParams(e.target.value)}
              placeholder="e.g., Material: 6061-T6 Aluminum. Use the structured inputs below for FEA."
              disabled={isLoading || feaInputs.isFeaEnabled}
            />
          </div>

          <div className="form-group">
            <div className="form-toggle">
                <input
                    type="checkbox"
                    id="fea-enable-toggle"
                    name="isFeaEnabled"
                    checked={feaInputs.isFeaEnabled}
                    onChange={handleFeaInputChange}
                />
                <label htmlFor="fea-enable-toggle">Enable Structured FEA Parameters</label>
            </div>

            {feaInputs.isFeaEnabled && (
                <div className="structured-inputs-container">
                    <div className="preset-buttons">
                        <button onClick={() => handleLoadFeaPreset('steel')}>Load Structural Steel</button>
                        <button onClick={() => handleLoadFeaPreset('aluminum')}>Load Aluminum 6061-T6</button>
                    </div>
                    <div className="inputs-grid two-col">
                        <div className="form-group">
                            <label htmlFor="materialName">Material Name</label>
                            <input type="text" id="materialName" name="materialName" value={feaInputs.materialName} onChange={handleFeaInputChange} />
                        </div>
                         <div className="form-group">
                            <label htmlFor="density">Density (kg/m³)</label>
                            <input type="text" id="density" name="density" value={feaInputs.density} onChange={handleFeaInputChange} />
                        </div>
                        <div className="form-group">
                            <label htmlFor="youngsModulus">Young's Modulus (GPa)</label>
                            <input type="text" id="youngsModulus" name="youngsModulus" value={feaInputs.youngsModulus} onChange={handleFeaInputChange} />
                        </div>
                        <div className="form-group">
                            <label htmlFor="poissonsRatio">Poisson's Ratio</label>
                            <input type="text" id="poissonsRatio" name="poissonsRatio" value={feaInputs.poissonsRatio} onChange={handleFeaInputChange} />
                        </div>
                    </div>
                </div>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="fourd-params-input">4D / Animation Parameters (Optional)</label>
            <textarea
              id="fourd-params-input"
              value={fourDParams}
              onChange={(e) => setFourDParams(e.target.value)}
              placeholder="e.g., rotate on Z-axis, 360 degrees over 5 seconds"
              disabled={isLoading}
            />
             <div className="form-group sub-group">
              <label htmlFor="animation-interpolation">Interpolation</label>
              <select id="animation-interpolation" value={animationInterpolation} onChange={(e) => setAnimationInterpolation(e.target.value)} disabled={isLoading}>
                <option value="ease-in-out">Ease-In-Out</option>
                <option value="linear">Linear</option>
                <option value="ease-in">Ease-In</option>
                <option value="ease-out">Ease-Out</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <div className="form-toggle">
                <input
                    type="checkbox"
                    id="cfd-enable-toggle"
                    name="isCfdEnabled"
                    checked={cfdInputs.isCfdEnabled}
                    onChange={handleCfdInputChange}
                />
                <label htmlFor="cfd-enable-toggle">Enable Structured CFD Parameters</label>
            </div>

            {cfdInputs.isCfdEnabled && (
                <div className="structured-inputs-container">
                    <h4>Fluid Properties</h4>
                    <div className="inputs-grid three-col">
                        <div className="form-group">
                            <label htmlFor="fluidName">Fluid Name</label>
                            <input type="text" id="fluidName" name="fluidName" value={cfdInputs.fluidName} onChange={handleCfdInputChange} />
                        </div>
                        <div className="form-group">
                            <label htmlFor="cfd-density">Density (kg/m³)</label>
                            <input type="text" id="cfd-density" name="density" value={cfdInputs.density} onChange={handleCfdInputChange} />
                        </div>
                        <div className="form-group">
                            <label htmlFor="viscosity">Viscosity (Pa·s)</label>
                            <input type="text" id="viscosity" name="viscosity" value={cfdInputs.viscosity} onChange={handleCfdInputChange} />
                        </div>
                    </div>

                    <h4>Turbulence Model</h4>
                    <div className="form-group">
                        <select id="turbulenceModel" name="turbulenceModel" value={cfdInputs.turbulenceModel} onChange={handleCfdInputChange}>
                            <option value="k-epsilon">k-epsilon</option>
                            <option value="k-omega SST">k-omega SST</option>
                            <option value="Spalart-Allmaras">Spalart-Allmaras</option>
                            <option value="None (Laminar)">None (Laminar)</option>
                        </select>
                    </div>

                    <h4>Boundary Conditions</h4>
                    <div className="boundary-condition-section">
                        <h5>Inlet</h5>
                        <div className="inputs-grid three-col">
                            <div className="form-group">
                                <label htmlFor="inletType">Type</label>
                                <select id="inletType" name="inletType" value={cfdInputs.inletType} onChange={handleCfdInputChange}>
                                    <option>Velocity</option>
                                    <option>Pressure</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label htmlFor="inletValue">Value ({cfdInputs.inletType === 'Velocity' ? 'm/s' : 'Pa'})</label>
                                <input type="text" id="inletValue" name="inletValue" value={cfdInputs.inletValue} onChange={handleCfdInputChange} />
                            </div>
                            {cfdInputs.inletType === 'Velocity' && (
                                <div className="form-group">
                                    <label htmlFor="inletDirection">Direction</label>
                                    <select id="inletDirection" name="inletDirection" value={cfdInputs.inletDirection} onChange={handleCfdInputChange}>
                                        <option>X</option>
                                        <option>Y</option>
                                        <option>Z</option>
                                    </select>
                                </div>
                            )}
                        </div>
                    </div>
                    <div className="boundary-condition-section">
                        <h5>Outlet</h5>
                        <div className="inputs-grid two-col">
                            <div className="form-group">
                                <label htmlFor="outletType">Type</label>
                                <select id="outletType" name="outletType" value={cfdInputs.outletType} onChange={handleCfdInputChange}>
                                    <option>Pressure</option>
                                    <option value="Zero-gradient">Zero-gradient</option>
                                </select>
                            </div>
                            {cfdInputs.outletType === 'Pressure' && (
                                <div className="form-group">
                                    <label htmlFor="outletValue">Value (Pa)</label>
                                    <input type="text" id="outletValue" name="outletValue" value={cfdInputs.outletValue} onChange={handleCfdInputChange} />
                                </div>
                            )}
                        </div>
                    </div>
                    <div className="boundary-condition-section">
                        <h5>Walls</h5>
                        <div className="form-group">
                            <label htmlFor="wallType">Wall Type</label>
                            <select id="wallType" name="wallType" value={cfdInputs.wallType} onChange={handleCfdInputChange}>
                                <option>no-slip</option>
                                <option>slip</option>
                            </select>
                        </div>
                    </div>
                </div>
            )}
          </div>

          <div className="settings-row">
            <div className="form-group">
              <label htmlFor="export-format">Export Format</label>
              <select id="export-format" value={exportFormat} onChange={(e) => setExportFormat(e.target.value)} disabled={isLoading}>
                <option>STL</option>
                <option>STEP</option>
                <option>OBJ</option>
                <option>OpenSCAD</option>
                <option>CFD</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="aspect-ratio">Image Aspect Ratio</label>
              <select id="aspect-ratio" value={aspectRatio} onChange={(e) => setAspectRatio(e.target.value)} disabled={isLoading}>
                <option>4:3</option>
                <option>16:9</option>
                <option>1:1</option>
                <option>3:4</option>
                <option>9:16</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="unit-system">Default Unit System</label>
              <select id="unit-system" value={defaultUnitSystem} onChange={(e) => setDefaultUnitSystem(e.target.value)} disabled={isLoading}>
                <option>Metric</option>
                <option>Imperial</option>
              </select>
            </div>
          </div>
          <div className="button-group">
              <button className="clear-button" onClick={handleClear} disabled={isLoading}>Clear</button>
              <button 
                className="generate-button" 
                onClick={handleOptimizePrompt}
                disabled={isOptimizing || !prompt.trim()}
                style={{ 
                  background: isOptimizing ? '#3f3f46' : 'linear-gradient(45deg, #9333ea, #e52e71)' 
                }}
              >
                {isOptimizing ? 'Analyzing...' : '🎯 Optimize Prompt (AI)'}
              </button>
              <button className="generate-button" onClick={handleGenerate} disabled={isLoading}>
                  {isLoading ? 'Generating...' : 'Generate Model'}
              </button>
          </div>
      </div>

      <OptimizationPanel />
      <CFDResultsPanel />
      
      {isLoading && (
        <div className="output-container">
            <div className="loader"></div>
            <p style={{ textAlign: 'center' }}>{loadingStatus}</p>
        </div>
      )}

      {(output || imageUrl || stlData) && !isLoading && (
        <div className="output-container">
          <h2>Generated Output</h2>
          <div className="output-grid">
              <div className="grid-item">
                  <h3>Interactive Model Preview</h3>
                  {stlData ? <ModelViewer stlData={stlData} animationData={animationData} /> : <div className="placeholder">Interactive preview is only available for STL format.</div>}
              </div>
              <div className="grid-item">
                  <h3>Photorealistic Render</h3>
                  {imageUrl ? <img src={imageUrl} alt="Generated CAD model" /> : <div className="placeholder">Render will appear here.</div>}
              </div>
          </div>

          <div className="script-container">
            <div className="output-header">
                <h3>Generated Script/Data</h3>
                <button className="download-button" onClick={() => handleDownload(output, 'model-data.json')}>Download JSON</button>
            </div>
            <pre>
              <code>{output}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

const root = createRoot(document.getElementById('root'));
root.render(<App />);