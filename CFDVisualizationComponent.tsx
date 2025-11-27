import React, { useEffect, useRef, useState } from 'react';
import { CFDVisualizer, type VisualizationConfig } from './src/cfd-visualizer';
import type { SimulationResult, CFDConfig } from './src/cfd-simulator';

interface CFDVisualizationProps {
  results: SimulationResult;
  config: CFDConfig;
}

const CFDVisualizationComponent: React.FC<CFDVisualizationProps> = ({ results, config }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const visualizerRef = useRef<CFDVisualizer | null>(null);
  
  const [vizConfig, setVizConfig] = useState<VisualizationConfig>({
    showVelocityVectors: true,
    showPressureContours: true,
    showStreamlines: false,
    showParticles: true,
    particleCount: 1000,
    vectorScale: 2.0,
    colorScheme: 'jet',
  });

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize visualizer
    visualizerRef.current = new CFDVisualizer(containerRef.current);
    visualizerRef.current.visualize(results, config, vizConfig);

    // Cleanup
    return () => {
      if (visualizerRef.current) {
        visualizerRef.current.dispose();
      }
    };
  }, [results, config]);

  useEffect(() => {
    if (visualizerRef.current) {
      visualizerRef.current.updateConfig(vizConfig);
    }
  }, [vizConfig]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      {/* Visualization container */}
      <div 
        ref={containerRef} 
        style={{ 
          width: '100%', 
          height: '600px',
          border: '1px solid #3f3f46',
          borderRadius: '8px',
          backgroundColor: '#0c0a09'
        }} 
      />
      
      {/* Controls */}
      <div style={{ 
        marginTop: '1rem', 
        padding: '1rem',
        backgroundColor: '#18181b',
        borderRadius: '8px',
        border: '1px solid #3f3f46'
      }}>
        <h4 style={{ marginBottom: '1rem', color: '#e5e5e5' }}>Visualization Controls</h4>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
          {/* Toggle switches */}
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#d4d4d8' }}>
            <input
              type="checkbox"
              checked={vizConfig.showPressureContours}
              onChange={(e) => setVizConfig({ ...vizConfig, showPressureContours: e.target.checked })}
              style={{ accentColor: '#ff8a00' }}
            />
            Pressure Contours
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#d4d4d8' }}>
            <input
              type="checkbox"
              checked={vizConfig.showVelocityVectors}
              onChange={(e) => setVizConfig({ ...vizConfig, showVelocityVectors: e.target.checked })}
              style={{ accentColor: '#ff8a00' }}
            />
            Velocity Vectors
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#d4d4d8' }}>
            <input
              type="checkbox"
              checked={vizConfig.showStreamlines}
              onChange={(e) => setVizConfig({ ...vizConfig, showStreamlines: e.target.checked })}
              style={{ accentColor: '#ff8a00' }}
            />
            Streamlines
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#d4d4d8' }}>
            <input
              type="checkbox"
              checked={vizConfig.showParticles}
              onChange={(e) => setVizConfig({ ...vizConfig, showParticles: e.target.checked })}
              style={{ accentColor: '#ff8a00' }}
            />
            Animated Particles
          </label>
        </div>

        <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
          {/* Color scheme selector */}
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#d4d4d8' }}>
              Color Scheme
            </label>
            <select
              value={vizConfig.colorScheme}
              onChange={(e) => setVizConfig({ ...vizConfig, colorScheme: e.target.value as any })}
              style={{
                width: '100%',
                padding: '0.5rem',
                backgroundColor: '#0c0a09',
                border: '1px solid #3f3f46',
                borderRadius: '4px',
                color: '#e5e5e5'
              }}
            >
              <option value="jet">Jet (Blue → Red)</option>
              <option value="viridis">Viridis</option>
              <option value="rainbow">Rainbow</option>
            </select>
          </div>

          {/* Particle count slider */}
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#d4d4d8' }}>
              Particle Count: {vizConfig.particleCount}
            </label>
            <input
              type="range"
              min="100"
              max="2000"
              step="100"
              value={vizConfig.particleCount}
              onChange={(e) => setVizConfig({ ...vizConfig, particleCount: parseInt(e.target.value) })}
              style={{ width: '100%', accentColor: '#ff8a00' }}
            />
          </div>
        </div>

        <div style={{ marginTop: '1rem' }}>
          <p style={{ fontSize: '0.875rem', color: '#a1a1aa' }}>
            💡 <strong>Tip:</strong> Pressure contours show high (red) and low (blue) pressure regions. 
            Particles trace the flow path in real-time. Use checkboxes to toggle different visualization layers.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CFDVisualizationComponent;