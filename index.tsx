import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from '@google/genai';
import RootRouter from "./RootRouter";

const systemPrompt = `
System Role:
You are KelmoidAI_Genesis_llm, a World-Class Senior CAD Engineer and Computational Geometer. You specialize in translating natural language descriptions into precise, executable 3D model code with 100% geometric validity.

Your Goal:
Generate syntactically correct, parametrically sound, and geometrically accurate OpenSCAD code.

Critical Methodology (Chain of Thought):
1.  **Deconstruct**: Break the object down into primitive shapes (cubes, cylinders, spheres, polygons).
2.  **Parameterize**: ALWAYS define variables at the top of the script for key dimensions (e.g., \`length = 50;\`, \`radius = 10;\`). This makes the model adjustable.
3.  **Plan Operations**: Determine the Constructive Solid Geometry (CSG) tree (Unions, Differences, Intersections).
4.  **Prevent Z-Fighting (CRITICAL)**: When performing a \`difference()\` or subtraction, the negative shape MUST be slightly larger than the material it is removing.
    *   Define an epsilon variable: \`eps = 0.01;\`
    *   Extend cuts: \`translate([0,0,-eps]) cylinder(h=height+2*eps, ...)\`
5.  **Refine**: Ensure \`$fn=100\` (or higher) is used for all circular features to ensure smooth curves.

Output Format:
You must output a single JSON object.

JSON Schema:
{
  "model_type": "CAD" | "4D" | "CFD",
  "domain": "mechanical" | "architectural" | "civil" | "other",
  "description": "Short summary of the generated model",
  "design_logic": "Step-by-step engineering reasoning. Explain how you calculated dimensions, how you handled constraints, and specifically how you prevented Z-fighting.",
  "cad_script": "The full, executable OpenSCAD script.",
  "export_format": "SCAD" | "STL" | "DXF" | "SVG",
  "animation_data": { ... }, 
  "simulation_data": { ... },
  "metadata": { "units": "mm" | "in", "version": "2.1" }
}

Strategy for Common Requests:
-   **Gears**: Use parametric involute gear formulas if possible, or approximate with cylinders and teeth.
-   **Enclosures**: Use \`minkowski()\` sparingly (it is slow) or \`hull()\` with circles/spheres for rounded corners.
-   **Pipes/Tubes**: Difference of outer cylinder and inner cylinder (inner must be taller/longer).
-   **Brackets**: Union of cubes/prisms, then subtract holes.
`;

// --- CONFIGURATION ---
// Replace this with your actual Stripe Payment Link from the Stripe Dashboard
const STRIPE_PAYMENT_LINK = "https://buy.stripe.com/test_5kA7sI..."; 

const PricingModal = ({ isOpen, onClose, userPlan, onSimulateUpgrade }) => {
  if (!isOpen) return null;

  const handleSubscribe = () => {
    // In a real app, this redirects to Stripe
    if (STRIPE_PAYMENT_LINK && STRIPE_PAYMENT_LINK.startsWith('http')) {
        window.location.href = STRIPE_PAYMENT_LINK;
    } else {
        alert("Stripe Integration Demo:\n\nTo make this work, replace the 'STRIPE_PAYMENT_LINK' constant in the code with your actual Stripe Payment Link URL.\n\nFor now, you can click 'Simulate Success' to see the Pro UI.");
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content pricing-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Upgrade Plan</h2>
          <button className="modal-close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="pricing-grid">
            {/* Free Plan */}
            <div className={`pricing-card ${userPlan === 'free' ? 'current-plan' : ''}`}>
              <div className="plan-header">
                <h3>Starter</h3>
                <div className="price">$0<span>/mo</span></div>
              </div>
              <ul className="features-list">
                <li>✓ Standard Generation Speed</li>
                <li>✓ Basic CAD Scripts</li>
                <li>✓ 720p Render Previews</li>
                <li>✓ Local History (10 items)</li>
              </ul>
              <button className="plan-button" disabled={userPlan === 'free'}>
                {userPlan === 'free' ? 'Current Plan' : 'Downgrade'}
              </button>
            </div>

            {/* Pro Plan */}
            <div className={`pricing-card pro-card ${userPlan === 'pro' ? 'current-plan' : ''}`}>
              <div className="popular-badge">Most Popular</div>
              <div className="plan-header">
                <h3>Pro Engineer</h3>
                <div className="price">$29<span>/mo</span></div>
              </div>
              <ul className="features-list">
                <li>✓ <strong>Priority GPU Access</strong></li>
                <li>✓ <strong>4K Photorealistic Renders</strong></li>
                <li>✓ Advanced CFD & 4D Models</li>
                <li>✓ Unlimited Cloud History</li>
                <li>✓ Commercial License</li>
              </ul>
              {userPlan === 'pro' ? (
                 <button className="plan-button active">Plan Active</button>
              ) : (
                 <div className="button-stack">
                    <button className="plan-button pro-button" onClick={handleSubscribe}>
                        Subscribe with Stripe
                    </button>
                    {/* Demo Only Button */}
                    <button className="demo-link" onClick={onSimulateUpgrade}>
                        (Demo: Simulate Success)
                    </button>
                 </div>
              )}
            </div>
          </div>
          <p className="pricing-note">
            Secure payments processed by <strong>Stripe</strong>. You can cancel at any time.
          </p>
        </div>
      </div>
    </div>
  );
};

const AnalyticsModal = ({ isOpen, onClose, analytics }) => {
  if (!isOpen) return null;

  const totalAttempts = analytics.totalPrompts + analytics.totalErrors;
  const successRate = totalAttempts > 0 
    ? Math.round((analytics.totalPrompts / totalAttempts) * 100) 
    : 100;

  const successColor = successRate >= 90 ? '#4ade80' : successRate >= 70 ? '#fbbf24' : '#ef4444';

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  // --- Chart Logic ---
  const getLast7Days = () => {
    const days = [];
    const dateLabels = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      days.push(d.toISOString().split('T')[0]); // YYYY-MM-DD
      dateLabels.push(d.toLocaleDateString(undefined, { weekday: 'short' }));
    }
    return { keys: days, labels: dateLabels };
  };

  const { keys: dayKeys, labels: dayLabels } = getLast7Days();
  
  const chartData = dayKeys.map(key => analytics.dailyCounts?.[key] || 0);
  const maxVal = Math.max(...chartData, 1); // Ensure no division by zero

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content analytics-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Usage Analytics</h2>
          <button className="modal-close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="analytics-grid">
            <div className="stat-card">
              <span className="stat-label">Prompts Processed</span>
              <span className="stat-value">{analytics.totalPrompts}</span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Success Rate</span>
              <span className="stat-value" style={{ color: successColor }}>
                {successRate}%
              </span>
              <div className="stat-bar-bg">
                <div 
                  className="stat-bar-fill" 
                  style={{ width: `${successRate}%`, backgroundColor: successColor }}
                ></div>
              </div>
            </div>

            <div className="stat-card">
              <span className="stat-label">Total Errors</span>
              <span className="stat-value error-text">{analytics.totalErrors}</span>
            </div>

            <div className="stat-card">
              <span className="stat-label">Total Sessions</span>
              <span className="stat-value">{analytics.sessions}</span>
            </div>
          </div>

          <div className="chart-container">
            <h3>Activity (Last 7 Days)</h3>
            <div className="bar-chart">
                {chartData.map((count, idx) => (
                    <div key={idx} className="chart-column">
                        <div 
                            className="chart-bar" 
                            style={{ 
                                height: `${(count / maxVal) * 100}%`,
                                opacity: count > 0 ? 1 : 0.3 
                            }}
                            title={`${count} generations on ${dayKeys[idx]}`}
                        ></div>
                        <span className="chart-label">{dayLabels[idx]}</span>
                    </div>
                ))}
            </div>
          </div>

          <div className="retention-info">
             <div className="retention-row">
                <span className="label">First Active:</span>
                <span className="value">{formatDate(analytics.firstSeen)}</span>
             </div>
             <div className="retention-row">
                <span className="label">Last Active:</span>
                <span className="value">{formatDate(analytics.lastSeen)}</span>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

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
    cfdMultiphase: 'Simulate a 2D water droplet (diameter 1cm) falling under gravity into a pool of still air.'
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Prompt Engineering Guide</h2>
          <button className="modal-close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <p>KelmoidAI Genesis responds best to structured engineering requests. Below are high-fidelity templates.</p>

          <div className="help-section">
            <h3>Standard CAD</h3>
            <div className="example">
              <code>{examples.mechanical}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.mechanical)}>Copy</button>
            </div>
            <div className="example">
              <code>{examples.architectural}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.architectural)}>Copy</button>
            </div>
          </div>

          <div className="help-section">
            <h3>4D & Animation</h3>
            <div className="example">
              <code>{examples.fourD}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.fourD)}>Copy</button>
            </div>
             <div className="example">
              <code>{examples.fourDCombined}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.fourDCombined)}>Copy</button>
            </div>
          </div>

          <div className="help-section">
            <h3>CFD Simulation</h3>
            <div className="example">
              <code>{examples.cfd}</code>
              <button className="copy-button" onClick={() => handleCopy(examples.cfd)}>Copy</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const App = () => {
  const [userPrompt, setUserPrompt] = useState('');
  const [materialParams, setMaterialParams] = useState('');
  const [fourDParams, setFourDParams] = useState('');
  const [cfdParams, setCfdParams] = useState('');
  const [generatedCode, setGeneratedCode] = useState('');
  const [designLogic, setDesignLogic] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [showPromptHelper, setShowPromptHelper] = useState(false);
  
  // New State for Stripe & Analytics
  const [showPricing, setShowPricing] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [userPlan, setUserPlan] = useState('free'); // 'free' | 'pro'
  const [history, setHistory] = useState([]);
  const [analytics, setAnalytics] = useState({
    totalPrompts: 0,
    totalErrors: 0,
    firstSeen: Date.now(),
    lastSeen: Date.now(),
    sessions: 0,
    dailyCounts: {} // YYYY-MM-DD -> count
  });

  // Initialize LocalStorage Data
  useEffect(() => {
    // History
    const storedHistory = localStorage.getItem('kelmoid_history');
    if (storedHistory) {
      try { setHistory(JSON.parse(storedHistory)); } catch (e) {}
    }

    // Analytics
    const storedAnalytics = localStorage.getItem('kelmoid_analytics');
    if (storedAnalytics) {
       try { 
           const parsed = JSON.parse(storedAnalytics);
           // Update session
           parsed.sessions = (parsed.sessions || 0) + 1;
           parsed.lastSeen = Date.now();
           // Ensure dailyCounts exists (for backward compatibility)
           if (!parsed.dailyCounts) parsed.dailyCounts = {};
           
           setAnalytics(parsed);
           localStorage.setItem('kelmoid_analytics', JSON.stringify(parsed));
       } catch (e) {}
    } else {
        const initialAnalytics = { 
            totalPrompts: 0, 
            totalErrors: 0, 
            firstSeen: Date.now(), 
            lastSeen: Date.now(), 
            sessions: 1,
            dailyCounts: {} 
        };
        setAnalytics(initialAnalytics);
        localStorage.setItem('kelmoid_analytics', JSON.stringify(initialAnalytics));
    }

    // Subscription
    const storedPlan = localStorage.getItem('kelmoid_plan');
    if (storedPlan) setUserPlan(storedPlan);

    // Check URL for Payment Success (Stripe Redirect)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('payment_success') === 'true') {
        handlePaymentSuccess();
        // Clean URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
  }, []);

  const handlePaymentSuccess = () => {
    setUserPlan('pro');
    localStorage.setItem('kelmoid_plan', 'pro');
    alert("Welcome to KelmoidAI Pro! All advanced features are now unlocked.");
    setShowPricing(false);
  };

  const handleSimulateUpgrade = () => {
      handlePaymentSuccess();
  };

  const handleHistorySelect = (prompt) => {
    setUserPrompt(prompt);
  };

  const handleDeleteHistory = (e, index) => {
    e.stopPropagation();
    const newHistory = [...history];
    newHistory.splice(index, 1);
    setHistory(newHistory);
    localStorage.setItem('kelmoid_history', JSON.stringify(newHistory));
  }

  const handleGenerate = async () => {
    if (!userPrompt.trim()) return;

    setLoading(true);
    setErrorMessage('');
    setGeneratedCode('');
    setDesignLogic('');

    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
    let fullPrompt = `User Request: ${userPrompt}\n`;
    if (materialParams.trim()) fullPrompt += `Material Constraints: ${materialParams}\n`;
    if (fourDParams.trim()) fullPrompt += `4D Behavior: ${fourDParams}\n`;
    if (cfdParams.trim()) fullPrompt += `CFD Parameters: ${cfdParams}\n`;
    
    const requestConfig = {
      contents: fullPrompt,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
      }
    };

    try {
      let result;
      // Internal function to attempt generation with a specific model
      const generateWithModel = async (modelName) => {
        return await ai.models.generateContent({
            model: modelName,
            ...requestConfig
        });
      };

      try {
        // Attempt 1: High-Reasoning Model (Pro)
        result = await generateWithModel('gemini-3-pro-preview');
      } catch (err) {
        // Check for Overloaded (503) errors
        if (err.status === 503 || err.message?.includes('503') || err.message?.includes('overloaded')) {
            console.warn('Gemini 3 Pro overloaded. Initiating retry/fallback sequence...');
            
            // Wait 1.5s for transient issues to clear
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            try {
                // Attempt 2: Retry Pro once more
                result = await generateWithModel('gemini-3-pro-preview');
            } catch (retryErr) {
                console.warn('Retry failed. Falling back to Gemini 2.5 Flash for reliability.');
                // Attempt 3: Fallback to Flash (Fast, High Availability)
                result = await generateWithModel('gemini-2.5-flash');
            }
        } else {
            // Re-throw other errors (authentication, bad request, etc.)
            throw err;
        }
      }
      
      const responseText = result.text;
      const data = JSON.parse(responseText);

      // --- Success Logic ---
      setGeneratedCode(data.cad_script);
      setDesignLogic(data.design_logic);

      // Update History
      const newHistoryItem = { prompt: userPrompt, timestamp: Date.now() };
      const updatedHistory = [newHistoryItem, ...history].slice(0, userPlan === 'pro' ? 100 : 10);
      setHistory(updatedHistory);
      localStorage.setItem('kelmoid_history', JSON.stringify(updatedHistory));

      // Update Analytics (Functional update for thread safety)
      setAnalytics(prev => {
          const todayKey = new Date().toISOString().split('T')[0];
          const updatedCounts = { ...prev.dailyCounts };
          updatedCounts[todayKey] = (updatedCounts[todayKey] || 0) + 1;

          const newAnalytics = { 
              ...prev, 
              totalPrompts: prev.totalPrompts + 1,
              dailyCounts: updatedCounts
          };
          localStorage.setItem('kelmoid_analytics', JSON.stringify(newAnalytics));
          return newAnalytics;
      });

    } catch (error) {
      console.error(error);
      setErrorMessage(`Generation Failed: ${error.message || 'Unknown error'}`);
      
      // Update Analytics (Error)
      setAnalytics(prev => {
          const newAnalytics = { ...prev, totalErrors: prev.totalErrors + 1 };
          localStorage.setItem('kelmoid_analytics', JSON.stringify(newAnalytics));
          return newAnalytics;
      });

    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!generatedCode) return;
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'kelmoid_model.scad';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container">
      <header>
        <h1>KelmoidAI <span>Genesis</span></h1>
        <p>Text-to-CAD Geometric Engine</p>
        
        <div style={{ position: 'absolute', top: '50%', right: '0', transform: 'translateY(-50%)', display: 'flex', gap: '0.5rem' }}>
            <button className="stats-button" onClick={() => setShowAnalytics(true)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 3v18h18" />
                  <path d="M18 17V9" />
                  <path d="M13 17V5" />
                  <path d="M8 17v-3" />
                </svg>
                Stats
            </button>
            <button 
                className={`stats-button ${userPlan === 'pro' ? 'pro-badge' : ''}`} 
                onClick={() => setShowPricing(true)}
                style={userPlan === 'pro' ? { borderColor: '#ffd700', color: '#ffd700' } : {}}
            >
                {userPlan === 'pro' ? '★ Pro Member' : 'Upgrade'}
            </button>
        </div>
      </header>

      {/* History Section (Visible if history exists) */}
      {history.length > 0 && (
        <div className="history-container">
          <label>Recent Designs</label>
          <div className="history-list">
             {history.map((item, idx) => (
               <div key={idx} className="history-item" onClick={() => handleHistorySelect(item.prompt)}>
                  <span className="history-text">{item.prompt}</span>
                  <button className="history-delete" onClick={(e) => handleDeleteHistory(e, idx)}>&times;</button>
               </div>
             ))}
          </div>
        </div>
      )}

      <div className="card api-key-prompt">
        <h2>Enterprise-Grade CAD Generation</h2>
        <p>KelmoidAI requires a paid Google Cloud API key to access high-compute geometric models.</p>
      </div>

      <div className="card input-section">
        <div className="form-group">
          <div className="label-group">
            <label htmlFor="user-prompt">Geometric Description</label>
            <button className="prompt-helper-button" onClick={() => setShowPromptHelper(true)}>Prompt Guide</button>
          </div>
          <textarea
            id="user-prompt"
            value={userPrompt}
            onChange={(e) => setUserPrompt(e.target.value)}
            placeholder="Describe the 3D object, assembly, or simulation scenario..."
            disabled={loading}
          />
        </div>

        <div className="settings-row">
            <div className="form-group">
                <label htmlFor="material-params-input">Material Physics</label>
                <textarea
                    id="material-params-input"
                    value={materialParams}
                    onChange={(e) => setMaterialParams(e.target.value)}
                    placeholder="e.g. Aluminum 6061, Elastic Modulus..."
                    disabled={loading}
                />
            </div>
            <div className="form-group">
                <label htmlFor="fourd-params-input">4D / Animation</label>
                <textarea
                    id="fourd-params-input"
                    value={fourDParams}
                    onChange={(e) => setFourDParams(e.target.value)}
                    placeholder="e.g. Rotate 360deg over 5s..."
                    disabled={loading}
                />
            </div>
             <div className="form-group">
                <label htmlFor="cfd-params-input">CFD Boundary Conditions</label>
                <textarea
                    id="cfd-params-input"
                    value={cfdParams}
                    onChange={(e) => setCfdParams(e.target.value)}
                    placeholder="e.g. Inlet velocity 10m/s..."
                    disabled={loading}
                />
            </div>
        </div>

        <div className="button-group">
          <button 
            className="clear-button" 
            onClick={() => {
                setUserPrompt('');
                setMaterialParams('');
                setFourDParams('');
                setCfdParams('');
            }}
            disabled={loading}
          >
            Clear Inputs
          </button>
          <button 
            className="generate-button" 
            onClick={handleGenerate} 
            disabled={loading || !userPrompt.trim()}
          >
            {loading ? 'Processing Geometry...' : 'Generate Model'}
          </button>
        </div>
      </div>

      {loading && <div className="loader"></div>}

      {errorMessage && (
        <div className="error-message">
          <strong>Error:</strong> {errorMessage}
        </div>
      )}

      {(generatedCode || designLogic) && (
        <div className="output-container">
          <h2>Constructed Model</h2>

          {designLogic && (
             <div className="logic-container">
                <div className="output-header">
                    <h3>Design Logic & Chain of Thought</h3>
                </div>
                <div className="design-logic-content">
                    {designLogic}
                </div>
             </div>
          )}

          <div className="image-container">
             <div className="image-placeholder">
                [ 3D Render Preview - WebGL Engine Initializing... ]
             </div>
             <div className="rotation-hint">
                <p>Click & Drag to Rotate</p>
                <p>Scroll to Zoom</p>
             </div>
          </div>

          <div className="script-container">
            <div className="output-header">
              <h3>OpenSCAD Script</h3>
              <button className="download-button" onClick={handleDownload}>Download .SCAD</button>
            </div>
            <pre><code>{generatedCode}</code></pre>
          </div>
        </div>
      )}

      {/* Modals */}
      <PromptHelperModal isOpen={showPromptHelper} onClose={() => setShowPromptHelper(false)} />
      <PricingModal 
        isOpen={showPricing} 
        onClose={() => setShowPricing(false)} 
        userPlan={userPlan}
        onSimulateUpgrade={handleSimulateUpgrade}
      />
      <AnalyticsModal 
        isOpen={showAnalytics} 
        onClose={() => setShowAnalytics(false)} 
        analytics={analytics}
      />

    </div>
  );
};

const root = createRoot(document.getElementById('root'));
root.render(<App />);