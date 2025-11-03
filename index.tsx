
import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from '@google/genai';

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
For 4D: Animated transformations or time-dependent model states.
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
  "description": "A gear with 20 teeth, rotating 360 degrees",
  "cad_script": "...OpenSCAD or FreeCAD code with animation parameters...",
  "export_format": "Animated GIF/MP4 (conceptual)",
  "animation_data": {
    "type": "rotation",
    "axis": "z",
    "duration_seconds": 5,
    "start_angle_deg": 0,
    "end_angle_deg": 360
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
            <p>Provide specific physical properties needed for the analysis.</p>
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
    // Fix: Explicitly type `logEntries` as a Set of strings to avoid type errors.
    const logEntries = new Set<string>();

    const targetUnit = targetSystem === 'Metric' ? 'mm' : 'in';

    unitPatterns.forEach(({ name, regex }) => {
        // We need to reset lastIndex for global regex in a loop
        regex.lastIndex = 0;
        let match;
        while ((match = regex.exec(convertedPrompt)) !== null) {
            const originalMatch = match[0];
            const value = parseFloat(match[1]);
            let convertedValue;

            if (targetSystem === 'Metric') {
                if (name === 'in') convertedValue = value * CONVERSION_FACTORS.in_to_mm;
                else if (name === 'ft') convertedValue = value * CONVERSION_FACTORS.ft_to_mm;
                else if (name === 'm') convertedValue = value * CONVERSION_FACTORS.m_to_mm;
                else if (name === 'cm') convertedValue = value * CONVERSION_FACTORS.cm_to_mm;
                else continue; // Already metric mm, or not convertible
            } else { // Imperial
                if (name === 'mm') convertedValue = value * CONVERSION_FACTORS.mm_to_in;
                else if (name === 'cm') convertedValue = value * CONVERSION_FACTORS.cm_to_in;
                else if (name === 'm') convertedValue = value * CONVERSION_FACTORS.m_to_in;
                else continue; // Already imperial in/ft, or not convertible
            }
            
            if (convertedValue !== undefined) {
                 const roundedValue = Math.round(convertedValue * 100) / 100;
                 const replacement = `${roundedValue} ${targetUnit}`;
                 logEntries.add(`'${originalMatch}' -> '${replacement}'`);
            }
        }
    });
    
    // Perform the replacements after finding all matches to avoid conflicts
    logEntries.forEach(logEntry => {
        const [original, replacement] = logEntry.split(' -> ');
        const originalText = original.slice(1,-1); // remove quotes
        const replacementText = replacement.slice(1,-1);
        // Use a regex to replace to ensure we replace whole words
        convertedPrompt = convertedPrompt.replace(new RegExp(`\\b${originalText}\\b`, 'gi'), replacementText);
    });

    return {
        convertedPrompt,
        log: Array.from(logEntries).join(', ')
    };
}


const App = () => {
  const [prompt, setPrompt] = useState('Generate a 3D CAD model of a mechanical gear with 20 teeth, a 10mm bore, and a 2 inch outer diameter.');
  const [materialParams, setMaterialParams] = useState('');
  const [fourDParams, setFourDParams] = useState('');
  const [cfdParams, setCfdParams] = useState('');
  const [output, setOutput] = useState('');
  const [imageUrl, setImageUrl] = useState('');
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

  // Load state from localStorage on initial render
  useEffect(() => {
    // Load prompt history
    try {
      const savedHistory = localStorage.getItem('promptHistory');
      if (savedHistory) {
        setPromptHistory(JSON.parse(savedHistory));
      }
    } catch (error) {
      console.error("Could not load prompt history from localStorage", error);
    }
    
    // Load current prompt state
    try {
      const savedState = localStorage.getItem('currentPromptState');
      if (savedState) {
        const parsedState = JSON.parse(savedState);
        if (parsedState.prompt !== undefined) setPrompt(parsedState.prompt);
        if (parsedState.materialParams !== undefined) setMaterialParams(parsedState.materialParams);
        if (parsedState.fourDParams !== undefined) setFourDParams(parsedState.fourDParams);
        if (parsedState.cfdParams !== undefined) setCfdParams(parsedState.cfdParams);
        if (parsedState.exportFormat !== undefined) setExportFormat(parsedState.exportFormat);
        if (parsedState.aspectRatio !== undefined) setAspectRatio(parsedState.aspectRatio);
        if (parsedState.defaultUnitSystem !== undefined) setDefaultUnitSystem(parsedState.defaultUnitSystem);
      }
    } catch (error) {
        console.error("Could not load current prompt state from localStorage", error);
    }
  }, []);

  // Save state to localStorage whenever it changes
  useEffect(() => {
    try {
        const currentState = {
            prompt,
            materialParams,
            fourDParams,
            cfdParams,
            exportFormat,
            aspectRatio,
            defaultUnitSystem,
        };
        localStorage.setItem('currentPromptState', JSON.stringify(currentState));
    } catch (error) {
        console.error("Could not save current prompt state to localStorage", error);
    }
  }, [prompt, materialParams, fourDParams, cfdParams, exportFormat, aspectRatio, defaultUnitSystem]);


  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a description for the CAD model.');
      return;
    }
    setIsLoading(true);
    setError('');
    setOutput('');
    setImageUrl('');
    setUnitConversionLog('');

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const { convertedPrompt, log } = convertUnits(prompt, defaultUnitSystem);
      if (log) {
          setUnitConversionLog(log);
      }

      // Combine main prompt with parameters and export format
      const fullPrompt = `${convertedPrompt}
${materialParams.trim() ? `\nMaterial Properties: ${materialParams}` : ''}
${fourDParams.trim() ? `\n4D Parameters: ${fourDParams}` : ''}
${cfdParams.trim() ? `\nCFD Parameters: ${cfdParams}` : ''}
\nExport format: ${exportFormat}
\nDefault Unit System: ${defaultUnitSystem === 'Metric' ? 'mm' : 'in'}`;


      // Step 1: Generate CAD Script
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
      } catch (parseError) {
        console.error('Failed to parse JSON from response:', text, parseError);
        setError("The AI's response was not in a valid format. This can happen with complex or ambiguous prompts. Please try simplifying your request or rephrasing. The raw response is shown below for debugging.");
        setOutput(text);
        // Stop execution if the primary output is invalid
        setIsLoading(false);
        setLoadingStatus('');
        return;
      }

      // Step 2: Generate Image
      setLoadingStatus('Creating photorealistic render...');
      let imagePrompt;

      if (jsonOutput && jsonOutput.model_type === 'CFD') {
        const cfdDescription = jsonOutput.description || convertedPrompt; // Fallback to user prompt
        imagePrompt = `A scientific visualization of a CFD simulation showing ${cfdDescription}. Display the flow field using colored streamlines to show the fluid's path and velocity around the object. The geometry should be clearly rendered. Use a color spectrum from blue (low velocity) to red (high velocity). The overall image should be a professional, high-fidelity rendering suitable for an engineering report.`;
      } else {
        imagePrompt = `Photorealistic 3D CAD render of: ${convertedPrompt}. ${materialParams.trim() ? `Material: ${materialParams}.` : ''} ${fourDParams.trim() ? `Animation details: ${fourDParams}.` : ''} ${cfdParams.trim() ? `CFD simulation visualization: ${cfdParams}.` : ''} Professional studio lighting, detailed, high-resolution, on a neutral background.`;
      }
      
      const imageResponse = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt: imagePrompt,
        config: {
          numberOfImages: 1,
          outputMimeType: 'image/png',
          aspectRatio: aspectRatio,
        },
      });

      const base64ImageBytes = imageResponse.generatedImages[0].image.imageBytes;
      const generatedImageUrl = `data:image/png;base64,${base64ImageBytes}`;
      setImageUrl(generatedImageUrl);

       // Save to history on success
       const newHistoryEntry = {
        id: Date.now(),
        prompt,
        materialParams,
        fourDParams,
        cfdParams,
        exportFormat,
        aspectRatio,
        defaultUnitSystem,
      };

      setPromptHistory(prevHistory => {
        const updatedHistory = [newHistoryEntry, ...prevHistory].slice(0, 20); // Keep latest 20
        try {
          localStorage.setItem('promptHistory', JSON.stringify(updatedHistory));
        } catch (e) {
          console.error("Failed to save prompt history to localStorage", e);
        }
        return updatedHistory;
      });

    } catch (e) {
      console.error(e);
      let friendlyError = 'An unexpected error occurred while generating the CAD model. Please check the console for more details.';
      if (e instanceof Error) {
        const errorMessage = e.message.toLowerCase();
        if (errorMessage.includes('api key not valid')) {
            friendlyError = 'The API key is invalid or missing. Please ensure it is correctly configured in your environment settings.';
        } else if (errorMessage.includes('quota') || errorMessage.includes('rate limit') || errorMessage.includes('resource exhausted')) {
            friendlyError = 'You have exceeded your API usage quota or the service is temporarily overloaded. Please check your Google AI Studio account or try again in a few moments.';
        } else if (errorMessage.includes('fetch') || errorMessage.includes('network')) {
            friendlyError = 'A network error occurred. Please check your internet connection and try again.';
        } else if (errorMessage.includes('safety')) {
            friendlyError = 'The prompt was blocked due to safety settings. Please modify your prompt and try again.';
        }
      }
      setError(friendlyError);
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  const handleDownloadScript = () => {
    const blob = new Blob([output], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'kelmoidai-genesis-llm-cad-script.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadImage = () => {
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = 'kelmoidai-genesis-llm-render.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleClear = () => {
    const isConfirmed = window.confirm('Are you sure you want to clear the prompt and all generated output?');
    if (isConfirmed) {
      setPrompt('');
      setMaterialParams('');
      setFourDParams('');
      setCfdParams('');
      setOutput('');
      setImageUrl('');
      setError('');
      setUnitConversionLog('');
    }
  };

  const handleLoadPrompt = (historyItem) => {
    setPrompt(historyItem.prompt);
    setMaterialParams(historyItem.materialParams);
    setFourDParams(historyItem.fourDParams);
    setCfdParams(historyItem.cfdParams);
    setExportFormat(historyItem.exportFormat);
    setAspectRatio(historyItem.aspectRatio);
    setDefaultUnitSystem(historyItem.defaultUnitSystem);
    setIsHistoryModalOpen(false); // Close modal after loading
  };

  const handleClearPromptHistory = () => {
    if (window.confirm('Are you sure you want to clear your entire prompt history? This cannot be undone.')) {
        setPromptHistory([]);
        localStorage.removeItem('promptHistory');
    }
  };

  return (
    <div className="container">
      <header>
        <h1>KelmoidAI <span>Genesis llm</span></h1>
        <p>Your personal Text-to-CAD generative model</p>
      </header>
      
      <div className="card">
        <div className="form-group">
          <div className="label-group">
            <label htmlFor="prompt-input">Describe the model you want to create:</label>
            <div className="label-buttons">
              <button className="prompt-helper-button" onClick={() => setIsHistoryModalOpen(true)}>History</button>
              <button className="prompt-helper-button" onClick={() => setIsHelpModalOpen(true)}>Prompt Helper</button>
            </div>
          </div>
          <textarea
            id="prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., A 3D model of a coffee mug..."
            disabled={isLoading}
            aria-label="CAD model description"
          />
        </div>
        <div className="form-group">
          <label htmlFor="material-params-input">Material Properties (Optional):</label>
          <textarea
            id="material-params-input"
            value={materialParams}
            onChange={(e) => setMaterialParams(e.target.value)}
            placeholder="e.g., Material: Aluminum, Density: 2700 kg/m^3..."
            disabled={isLoading}
            aria-label="Material properties"
          />
        </div>
        <div className="form-group">
          <label htmlFor="fourd-params-input">4D / Animation Parameters (Optional):</label>
          <textarea
            id="fourd-params-input"
            value={fourDParams}
            onChange={(e) => setFourDParams(e.target.value)}
            placeholder="e.g., rotation on Z-axis, 360 degrees over 5 seconds..."
            disabled={isLoading}
            aria-label="4D or animation parameters"
          />
        </div>
        <div className="form-group">
          <label htmlFor="cfd-params-input">CFD / Simulation Parameters (Optional):</label>
          <textarea
            id="cfd-params-input"
            value={cfdParams}
            onChange={(e) => setCfdParams(e.target.value)}
            placeholder="e.g., fluid is air, inlet velocity 10 m/s, turbulence model k-epsilon..."
            disabled={isLoading}
            aria-label="CFD or simulation parameters"
          />
        </div>

        <div className="settings-row">
            <div className="form-group">
              <label htmlFor="unit-system-select">Default Unit System</label>
                <select
                  id="unit-system-select"
                  value={defaultUnitSystem}
                  onChange={(e) => setDefaultUnitSystem(e.target.value)}
                  disabled={isLoading}
                  aria-label="Select default unit system"
                >
                  <option value="Metric">Metric (mm)</option>
                  <option value="Imperial">Imperial (inches)</option>
                </select>
            </div>
            <div className="form-group">
              <label htmlFor="export-format-select">Export Format</label>
                <select
                  id="export-format-select"
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value)}
                  disabled={isLoading}
                  aria-label="Select CAD export format"
                >
                  <option value="OpenSCAD">OpenSCAD</option>
                  <option value="STEP">STEP</option>
                  <option value="STL">STL</option>
                  <option value="OBJ">OBJ</option>
                  <option value="CFD">CFD</option>
                </select>
            </div>
            <div className="form-group">
              <label htmlFor="aspect-ratio-select">Aspect Ratio</label>
                <select
                  id="aspect-ratio-select"
                  value={aspectRatio}
                  onChange={(e) => setAspectRatio(e.target.value)}
                  disabled={isLoading}
                  aria-label="Select image aspect ratio"
                >
                  <option value="1:1">1:1 (Square)</option>
                  <option value="4:3">4:3 (Landscape)</option>
                  <option value="3:4">3:4 (Portrait)</option>
                  <option value="16:9">16:9 (Widescreen)</option>
                  <option value="9:16">9:16 (Tall)</option>
                </select>
            </div>
        </div>

        <div className="button-group">
          <button className="clear-button" onClick={handleClear} disabled={isLoading}>
            Clear All
          </button>
          <button className="generate-button" onClick={handleGenerate} disabled={isLoading}>
            {isLoading ? (loadingStatus || 'Generating...') : 'Generate CAD Model'}
          </button>
        </div>
      </div>

      {(isLoading || output || error || imageUrl) && (
        <div className="card output-container" aria-live="polite">
          <h2>Output</h2>
          {isLoading && <div className="loader" role="status" aria-label="Loading"></div>}
          {unitConversionLog && (
            <div className="conversion-log">
                <p><strong>Unit Conversions:</strong> {unitConversionLog}</p>
            </div>
          )}
          {error && <div className="error-message">{error}</div>}
          
          {(imageUrl || (isLoading && !imageUrl)) && (
            <div className="image-container">
              {imageUrl ? (
                <>
                  <div className="output-header">
                    <h3>Visualization</h3>
                    <button className="download-button" onClick={handleDownloadImage}>Download Image</button>
                  </div>
                  <img src={imageUrl} alt="Generated CAD model visualization" />
                </>
              ) : (
                <div className="image-placeholder">
                  <p>{loadingStatus}</p>
                </div>
              )}
            </div>
          )}

          {output && (
            <div className="script-container">
              <div className="output-header">
                <h3>CAD Script</h3>
                <button className="download-button" onClick={handleDownloadScript}>Download Script</button>
              </div>
              <pre>
                <code>{output}</code>
              </pre>
            </div>
          )}
        </div>
      )}
      <PromptHelperModal isOpen={isHelpModalOpen} onClose={() => setIsHelpModalOpen(false)} />
      <PromptHistoryModal isOpen={isHistoryModalOpen} onClose={() => setIsHistoryModalOpen(false)} history={promptHistory} onLoad={handleLoadPrompt} onClear={handleClearPromptHistory} />
    </div>
  );
};

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(<App />);