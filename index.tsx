import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from '@google/genai';
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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
      // Maybe show a small "Copied!" message in the future
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
    fourDCombined: 'A gear rotates 720 degrees on its Z-axis while simultaneously scaling up to 150% of its original size over 8 seconds.',
    fourDSequential: 'First, a robotic arm extends 50mm along the X-axis over 2 seconds. Then, its claw rotates 90 degrees clockwise over 1 second.',
    fourDBending: 'A flat metal sheet (100x100x2mm) is bent 90 degrees along its central X-axis to create an L-shape over 1.5 seconds.',
    fourDTwisting: 'A square bar (10x10x100mm) is twisted 180 degrees along its longest axis (the Z-axis) over 3 seconds. The base of the bar at Z=0 remains fixed.',
    cfd: 'Simulate airflow over a cylinder with a diameter of 0.1m. The fluid is air (density 1.225 kg/m^3, viscosity 1.81e-5 Pa.s). The inlet velocity is 10 m/s from the left, and the outlet is at atmospheric pressure on the right. The top and bottom walls are slip walls.',
    cfdWater: 'Simulate water flowing through a pipe with a 90-degree bend. Pipe diameter is 5cm. Fluid is water at 20C (density 998.2 kg/m^3, viscosity 1.002e-3 Pa.s). Inlet velocity is 1.5 m/s. Outlet is pressure-based.',
    cfdMultiphase: 'Simulate a 2D water droplet (diameter 1cm) falling under gravity into a pool of still air. The domain is 10cm wide by 15cm high. The bottom 5cm is water, the rest is air. Use a VOF (Volume of Fluid) multiphase model to track the interface.',
    cfdMultiphaseDropletImpact: 'Simulate a 5mm water droplet impacting a dry surface at 2 m/s. Track the splashing and spreading of the droplet using a Volume of Fluid (VOF) model.',
    cfdMultiphaseBubbleColumn: 'Model a bubble column where air is injected from the bottom of a water tank at 0.1 m/s through a 1cm orifice. Observe bubble rise and coalescence using a VOF multiphase model.',
    cfdHeatTransfer: 'Simulate conjugate heat transfer for an aluminum heatsink in a channel. The heatsink base is at a constant 373K (100C). Air at 298K (25C) flows into the channel at 2 m/s. Include heat conduction within the solid heatsink and convection to the surrounding air.',
    cfdTurbulenceKEpsilon: 'Turbulence Model: Standard k-epsilon, a robust model for general industrial flows.',
    cfdTurbulenceKOmega: 'Turbulence Model: k-omega SST, suitable for aerodynamic flows with boundary layer separation.',
    cfdWallFunction: 'Wall Treatment: Apply standard wall functions on the airfoil surface.',
    cfdInletProfileTurbulent: 'Inlet Velocity Profile: A fully developed turbulent profile following the 1/7th power law.',
    cfdInletProfileFormula: 'Inlet Velocity Profile: Parabolic, with max velocity of 2 m/s at the center, via U(y) = 2 * (1 - (y/0.05)^2), where y is distance from centerline.',
    cfdInletProfileAngled: 'Inlet Velocity: 10 m/s at a 30-degree angle upwards from the main X-axis. Vector: (8.66, 5.0, 0) m/s.',
    cfdInletProfileSwirl: 'Inlet Condition: A swirling flow with a constant axial velocity of 5 m/s and tangential velocity varying linearly from 0 at the center to 2 m/s at the pipe wall (radius = 0.1m).',
    cfdMeshGlobal: 'Use a global cell size of 0.01m for the entire domain.',
    cfdMeshFine: 'Use a fine mesh, especially around the cylinder, to capture the wake accurately.',
    cfdMeshBoundaryLayer: 'Apply boundary layer refinement with 5 inflation layers and a growth rate of 1.2 on all no-slip walls.',
    cfdBoundaryInletUniform: 'Boundary Condition: Inlet with a uniform velocity of 10 m/s on the left face.',
    cfdBoundaryInletParabolic: 'Boundary Condition: Inlet with a parabolic velocity profile, max speed of 5 m/s on the left face.',
    cfdBoundaryOutletZeroGradient: 'Boundary Condition: The outlet on the right face is a free outlet (zero-gradient) for all flow variables.',
    cfdBoundaryOutletFarField: 'Simulate external airflow over a sports car at 60 m/s. The car surfaces are no-slip walls. All far-field boundaries, including the outlet, should be set as a pressure_far_field with 0 Pa gauge pressure.',
    cfdBoundaryWallNoSlip: 'Boundary Condition: The surface of the cylinder is a no-slip wall, where the fluid velocity is zero.',
    cfdBoundaryWallSlip: 'Boundary Condition: The top and bottom domain boundaries are slip walls, simulating an infinitely large domain.',
    cfdBoundarySymmetry: 'Boundary Condition: The top face is a symmetry plane to reduce computational cost.',
    cfdBoundaryPeriodic: 'Boundary Condition: The left and right faces are periodic, simulating flow over an infinite array of cylinders.',
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

          <div className="help-section">
            <h3>Specifying Export Formats</h3>
            <p>You can request a specific file format using the dropdown menu or by stating it in your prompt. The AI will generate a script or data structure suitable for that format.</p>
            <ul>
                <li><strong>Common CAD Formats:</strong> OpenSCAD, STEP, STL, OBJ</li>
                <li><strong>Conceptual CFD Formats:</strong> OpenFOAM_mesh, Fluent_mesh, VTK</li>
            </ul>
            <div className="example">
                <code>{examples.exportFormatExample}</code>
                <button className="copy-button" onClick={() => handleCopy(examples.exportFormatExample)}>Copy</button>
            </div>
          </div>

          <div className="help-section">
            <h3>Specifying Material Properties</h3>
            <p>Define the material for your model. This is crucial for simulations (CFD/FEA) but also useful for specifying the type of material for a CAD model.</p>
            
            <h5>For Simple CAD Models</h5>
            <p>Just stating the material name is often enough.</p>
            <div className="example">
              <code>{examples.materialSimple}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.materialSimple)}>Copy</button>
            </div>

            <h5>For Simulation (CFD / FEA)</h5>
            <p>Provide specific physical properties needed for the analysis. You can use the structured inputs for this.</p>
            <p><strong>Structural Analysis Example:</strong></p>
            <div className="example">
              <code>{examples.materialStructural}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.materialStructural)}>Copy</button>
            </div>

            <p><strong>Thermal Analysis Example:</strong></p>
            <div className="example">
              <code>{examples.materialThermal}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.materialThermal)}>Copy</button>
            </div>
          </div>
          
          <div className="help-section">
            <h3>Mechanical Domain</h3>
            <p>Use keywords like <strong>gear, bearing, piston, shaft, bracket, enclosure, threads (e.g., M5), tolerance, assembly</strong>.</p>
            <div className="example">
              <code>{examples.mechanical}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.mechanical)}>Copy</button>
            </div>
          </div>

          <div className="help-section">
            <h3>Architectural Domain</h3>
            <p>Use keywords like <strong>building, facade, floor plan, beam, column, truss, window, door, roof type (e.g., gable, flat)</strong>.</p>
            <div className="example">
              <code>{examples.architectural}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.architectural)}>Copy</button>
            </div>
          </div>

          <div className="help-section">
            <h3>CFD Simulation</h3>
            <p>Describe the fluid, flow conditions, and boundaries. Use keywords like <strong>inlet, outlet, wall, velocity, pressure, density, viscosity, laminar, turbulent</strong>.</p>
            <div className="example">
              <code>{examples.cfd}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfd)}>Copy</button>
            </div>

            <h4>Fluid Properties</h4>
            <p>Specify the fluid being simulated. For common fluids, you can state the name and temperature. For others, provide density and viscosity.</p>
            <div className="example">
              <code>{examples.cfdWater}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdWater)}>Copy</button>
            </div>
            
            <h4>Multiphase Flow</h4>
            <p>
              Simulate systems with two or more distinct fluids, like water and air. These models are essential for analyzing phenomena such as droplet formation, bubble dynamics, and liquid-air interfaces. The <strong>Volume of Fluid (VOF)</strong> model is a common and effective method for tracking the interface between the fluids.
            </p>
            <p><strong>Droplet Falling in Air (Liquid-Air Interface):</strong> A classic example to model interface tracking and gravity effects.</p>
            <div className="example">
              <code>{examples.cfdMultiphase}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdMultiphase)}>Copy</button>
            </div>
            <p><strong>Droplet Impact:</strong> Useful for analyzing spray cooling, inkjet printing, or surface coating processes.</p>
            <div className="example">
              <code>{examples.cfdMultiphaseDropletImpact}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdMultiphaseDropletImpact)}>Copy</button>
            </div>
            <p><strong>Bubble Column:</strong> Common in chemical reactors and aeration systems to study bubble dynamics.</p>
            <div className="example">
              <code>{examples.cfdMultiphaseBubbleColumn}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdMultiphaseBubbleColumn)}>Copy</button>
            </div>

            <h4>Heat Transfer</h4>
            <p>Include temperatures for fluids and surfaces. For conjugate heat transfer (CHT), specify solid materials and their thermal properties.</p>
            <div className="example">
              <code>{examples.cfdHeatTransfer}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdHeatTransfer)}>Copy</button>
            </div>

            <h4>Advanced CFD Parameters</h4>
            <p>For more complex simulations, you can specify turbulence models, wall treatments, and custom boundary profiles to achieve higher fidelity results.</p>
            
            <h5>Turbulence Models</h5>
            <p>Required for non-laminar (turbulent) flows. The choice of model impacts accuracy and computational cost.</p>
            <div className="example">
              <code>{examples.cfdTurbulenceKOmega}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdTurbulenceKOmega)}>Copy</button>
            </div>
            <div className="example">
              <code>{examples.cfdTurbulenceKEpsilon}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdTurbulenceKEpsilon)}>Copy</button>
            </div>

            <h5>Wall Functions</h5>
            <p>Define how the flow is modeled near solid surfaces in turbulent simulations. This is crucial for accurately capturing boundary layer effects.</p>
            <div className="example">
              <code>{examples.cfdWallFunction}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdWallFunction)}>Copy</button>
            </div>

            <h5>Custom Inlet Profiles</h5>
            <p>Instead of a uniform value, you can define a velocity or temperature profile across an inlet to better represent real-world conditions. This allows for more realistic simulations of phenomena like fully developed pipe flow, angled jets, or swirling vortices.</p>

            <p><strong>Using a Formula:</strong> Define the profile using a mathematical expression for precise control.</p>
            <div className="example">
              <code>{examples.cfdInletProfileFormula}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdInletProfileFormula)}>Copy</button>
            </div>
            
            <p><strong>Angled Flow:</strong> Specify the velocity direction using an angle or a vector for non-axial flow.</p>
            <div className="example">
              <code>{examples.cfdInletProfileAngled}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdInletProfileAngled)}>Copy</button>
            </div>

            <p><strong>Descriptive Swirl:</strong> Describe complex rotational flow patterns for things like cyclones or vortex tubes.</p>
            <div className="example">
              <code>{examples.cfdInletProfileSwirl}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdInletProfileSwirl)}>Copy</button>
            </div>
            
            <p><strong>Turbulent Profile:</strong> For turbulent flows, you can specify common engineering profiles like the power law.</p>
            <div className="example">
              <code>{examples.cfdInletProfileTurbulent}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdInletProfileTurbulent)}>Copy</button>
            </div>
            
            <h4>Mesh Properties (Optional)</h4>
            <p>Control the resolution of your simulation grid. A finer mesh gives more accurate results but requires more computation. You can specify global settings or local refinements.</p>
            
            <p><strong>Global Cell Size:</strong> Sets a uniform size for mesh cells everywhere. Good for simple geometries.</p>
            <div className="example">
              <code>{examples.cfdMeshGlobal}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdMeshGlobal)}>Copy</button>
            </div>

            <p><strong>Qualitative Refinement:</strong> Ask for a finer mesh in critical areas without specifying exact numbers.</p>
            <div className="example">
              <code>{examples.cfdMeshFine}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdMeshFine)}>Copy</button>
            </div>

            <p><strong>Boundary Layer Refinement:</strong> Crucial for accurately modeling flow near walls (e.g., on an airfoil or car body).</p>
            <div className="example">
              <code>{examples.cfdMeshBoundaryLayer}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdMeshBoundaryLayer)}>Copy</button>
            </div>


            <h4>Boundary Conditions</h4>
            <p>Clearly define how the fluid interacts with the boundaries of your domain. Being explicit is key.</p>
            
            <h5>Inlet Conditions (Where fluid enters)</h5>
            <p><strong>Uniform:</strong> The simplest inlet, where velocity is constant across the entire face. Good for general cases.</p>
            <div className="example">
              <code>{examples.cfdBoundaryInletUniform}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundaryInletUniform)}>Copy</button>
            </div>
            <p><strong>Parabolic:</strong> Simulates a fully developed flow profile, common in pipes or channels.</p>
            <div className="example">
              <code>{examples.cfdBoundaryInletParabolic}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundaryInletParabolic)}>Copy</button>
            </div>

            <h5>Outlet Conditions (Where fluid exits)</h5>
            <p><strong>Zero-Gradient / Free Outlet:</strong> A simple outlet that assumes the flow properties are no longer changing. It lets the flow exit naturally without imposing a fixed pressure. Best for internal flows where the outlet is far from disturbances.</p>
            <div className="example">
              <code>{examples.cfdBoundaryOutletZeroGradient}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundaryOutletZeroGradient)}>Copy</button>
            </div>
            <p><strong>Pressure Far-Field:</strong> Ideal for external aerodynamic simulations (e.g., airflow over a car, airplane wing, or a building). This condition is applied to boundaries of the simulation domain that are far away from the object, representing the undisturbed atmospheric pressure of the open air.</p>
            <div className="example">
              <code>{examples.cfdBoundaryOutletFarField}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundaryOutletFarField)}>Copy</button>
            </div>

            <h5>Wall Conditions (Solid surfaces)</h5>
            <p><strong>No-Slip:</strong> The default for most physical walls. The fluid velocity at the wall surface is zero.</p>
            <div className="example">
              <code>{examples.cfdBoundaryWallNoSlip}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundaryWallNoSlip)}>Copy</button>
            </div>
            <p><strong>Slip:</strong> The fluid can flow along the wall without friction. Often used for far-field boundaries or symmetry planes.</p>
            <div className="example">
              <code>{examples.cfdBoundaryWallSlip}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundaryWallSlip)}>Copy</button>
            </div>

            <h5>Other Conditions</h5>
            <p><strong>Symmetry:</strong> Used to model only a fraction of a symmetric problem, saving computational resources.</p>
            <div className="example">
              <code>{examples.cfdBoundarySymmetry}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundarySymmetry)}>Copy</button>
            </div>
            <p><strong>Periodic:</strong> What exits one boundary enters the opposite one, useful for repeating patterns (e.g., heat exchanger fins, turbine blades).</p>
             <div className="example">
              <code>{examples.cfdBoundaryPeriodic}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfdBoundaryPeriodic)}>Copy</button>
            </div>
          </div>

          <div className="help-section">
            <h3>4D / Animation Parameters</h3>
            <p>Use the dedicated input field for animations. Clearly describe the transformation, axis, magnitude, and duration. Be as descriptive as possible.</p>
            
            <h4>Basic Transformations</h4>
            <ul>
              <li><strong>Rotation:</strong> 'rotation on Z-axis, 360 degrees over 5 seconds'</li>
              <li><strong>Scaling:</strong> 'scale from 1x to 1.5x on all axes over 2 seconds'</li>
              <li><strong>Translation:</strong> 'move 100mm along the X-axis over 4 seconds'</li>
            </ul>

            <h4>Animation Timing (Interpolation)</h4>
            <p>You can control the timing of the animation using the "Interpolation" dropdown or by describing it in the prompt (e.g., "with ease-in-out timing").</p>
            <ul>
              <li><strong>Linear:</strong> Constant speed.</li>
              <li><strong>Ease-In-Out:</strong> Starts slow, speeds up, then ends slow.</li>
              <li><strong>Ease-In:</strong> Starts slow and accelerates.</li>
              <li><strong>Ease-Out:</strong> Starts fast and decelerates.</li>
            </ul>


            <h4>Advanced Examples</h4>
            <p><strong>Combined Animations:</strong> Describe multiple transformations happening at the same time.</p>
            <div className="example">
              <code>{examples.fourDCombined}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.fourDCombined)}>Copy</button>
            </div>

            <p><strong>Sequential Animations:</strong> Use words like "first," "then," and "after that" to define a sequence of events.</p>
             <div className="example">
              <code>{examples.fourDSequential}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.fourDSequential)}>Copy</button>
            </div>

            <h5>Complex Deformations (Bending & Twisting)</h5>
            <p>For non-rigid transformations, be explicit about the axis or plane of deformation. Specify the angle of the bend or twist and over what duration it should occur. Mentioning fixed points can help define the transformation more clearly.</p>

            <p><strong>Bending Example:</strong></p>
            <div className="example">
                <code>{examples.fourDBending}</code>
                <button className="copy-button" onClick={() => handleCopy(examples.fourDBending)}>Copy</button>
            </div>

            <p><strong>Twisting Example:</strong></p>
            <div className="example">
                <code>{examples.fourDTwisting}</code>
                <button className="copy-button" onClick={() => handleCopy(examples.fourDTwisting)}>Copy</button>
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
        // Always match against the original, unmodified prompt for consistency.
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
            } else { // Imperial
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
    
    // Function to escape special characters for use in a RegExp
    const escapeRegExp = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    // Perform the replacements after finding all matches to avoid conflicts
    logEntries.forEach(logEntry => {
        const [original, replacement] = logEntry.split(' -> ');
        const originalText = original.slice(1,-1); // remove quotes
        const replacementText = replacement.slice(1,-1);
        
        // Use a case-insensitive, global regex for replacement. This is more robust
        // than the previous implementation as it handles special characters.
        const searchRegex = new RegExp(escapeRegExp(originalText), 'gi');
        
        // Replace in the plain text version for the AI
        convertedPrompt = convertedPrompt.replace(searchRegex, replacementText);
        
        // Replace in the HTML version for the UI preview
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

        // Scene, Camera, Renderer
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x18181b);
        const camera = new THREE.PerspectiveCamera(75, currentMount.clientWidth / currentMount.clientHeight, 0.1, 1000);
        camera.position.z = 100;
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
        currentMount.appendChild(renderer.domElement);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);
        const pointLight1 = new THREE.PointLight(0xffffff, 0.8);
        pointLight1.position.set(100, 100, 100);
        scene.add(pointLight1);
        const pointLight2 = new THREE.PointLight(0xffffff, 0.5);
        pointLight2.position.set(-100, -100, -100);
        scene.add(pointLight2);

        // Load STL
        const loader = new STLLoader();
        const geometry = loader.parse(stlData);
        const material = new THREE.MeshStandardMaterial({
            color: 0xff8a00, metalness: 0.3, roughness: 0.6
        });
        const mesh = new THREE.Mesh(geometry, material);
        
        // Center and scale model
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        mesh.position.sub(center);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 100 / maxDim;
        mesh.scale.set(scale, scale, scale);
        scene.add(mesh);
        camera.lookAt(mesh.position);
        
        // Store initial state for animation reset
        animationState.current.initial = {
            position: mesh.position.clone(),
            rotation: mesh.rotation.clone(),
            scale: mesh.scale.clone()
        };

        // Handle resize
        const handleResize = () => {
            if (currentMount) {
                camera.aspect = currentMount.clientWidth / currentMount.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
            }
        };
        window.addEventListener('resize', handleResize);

        // Animation loop
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
                    // Optional: reset to end state explicitly
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
                
                // Add other animation types (scaling, translation) here...

                if (!animationState.current.isPlaying) {
                    // Reset to initial state after a delay to allow viewing the final state
                    setTimeout(() => {
                        if (mesh) { // check if mesh still exists
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

        // Cleanup
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

  // Load state from localStorage on initial render
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

  // Save state to localStorage whenever it changes
  useEffect(() => {
    try {
        const currentState = {
            prompt, materialParams, feaInputs, fourDParams, animationInterpolation, cfdInputs, exportFormat, aspectRatio, defaultUnitSystem,
        };
        localStorage.setItem('currentPromptState', JSON.stringify(currentState));
    } catch (error) { console.error("Could not save prompt state", error); }
  }, [prompt, materialParams, feaInputs, fourDParams, animationInterpolation, cfdInputs, exportFormat, aspectRatio, defaultUnitSystem]);

  // Update unit conversion preview on prompt change
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

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a description for the CAD model.');
      return;
    }
    setIsLoading(true);
    setError('');
    setOutput('');
    setImageUrl('');
    setStlData('');
    setAnimationData(null);
    setUnitConversionLog('');

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const { convertedPrompt, log } = convertUnits(prompt, defaultUnitSystem);
      if (log) setUnitConversionLog(log);
      
      const buildMaterialString = () => {
        if (feaInputs.isFeaEnabled) {
            let feaString = '\nMaterial Properties: ';
            if (feaInputs.materialName) feaString += `${feaInputs.materialName}. `;
            if (feaInputs.youngsModulus) feaString += `Young's Modulus: ${feaInputs.youngsModulus} GPa. `;
            if (feaInputs.poissonsRatio) feaString += `Poisson's Ratio: ${feaInputs.poissonsRatio}. `;
            if (feaInputs.density) feaString += `Density: ${feaInputs.density} kg/m^3.`;
            // Check if anything was actually added besides the label
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
                            <label htmlFor="density">Density (kg/m³)</label>
                            <input type="text" id="density" name="density" value={cfdInputs.density} onChange={handleCfdInputChange} />
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
              <button className="generate-button" onClick={handleGenerate} disabled={isLoading}>
                  {isLoading ? 'Generating...' : 'Generate Model'}
              </button>
          </div>
      </div>
      
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