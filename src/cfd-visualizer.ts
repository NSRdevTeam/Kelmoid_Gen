import * as THREE from 'three';
import type { SimulationResult, CFDConfig } from './cfd-simulator';

interface VisualizationConfig {
  showVelocityVectors: boolean;
  showPressureContours: boolean;
  showStreamlines: boolean;
  showParticles: boolean;
  particleCount: number;
  vectorScale: number;
  colorScheme: 'jet' | 'viridis' | 'rainbow';
}

interface ParticleData {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  age: number;
  maxAge: number;
}

class CFDVisualizer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private container: HTMLElement;
  private animationId: number | null = null;
  
  private velocityVectorGroup: THREE.Group;
  private pressureContoursGroup: THREE.Group;
  private streamlinesGroup: THREE.Group;
  private particleSystem: THREE.Points | null = null;
  private particles: ParticleData[] = [];
  
  private gridSize: { x: number; y: number; z: number };
  private velocityField: Float32Array;
  private pressureField: Float32Array;
  
  private config: VisualizationConfig = {
    showVelocityVectors: true,
    showPressureContours: true,
    showStreamlines: false,
    showParticles: true,
    particleCount: 1000,
    vectorScale: 2.0,
    colorScheme: 'jet',
  };

  constructor(containerElement: HTMLElement) {
    this.container = containerElement;
    this.velocityVectorGroup = new THREE.Group();
    this.pressureContoursGroup = new THREE.Group();
    this.streamlinesGroup = new THREE.Group();
    
    this.initializeScene();
  }

  private initializeScene(): void {
    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1a);
    
    // Camera setup
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    this.camera.position.set(50, 50, 50);
    this.camera.lookAt(0, 0, 0);
    
    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true 
    });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.container.appendChild(this.renderer.domElement);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(50, 50, 50);
    this.scene.add(directionalLight);
    
    // Add axes helper
    const axesHelper = new THREE.AxesHelper(20);
    this.scene.add(axesHelper);
    
    // Add grid helper
    const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x222222);
    this.scene.add(gridHelper);
    
    // Handle window resize
    window.addEventListener('resize', () => this.handleResize());
  }

  /**
   * Main visualization method - call this with CFD results
   */
  visualize(
    results: SimulationResult,
    cfdConfig: CFDConfig,
    config?: Partial<VisualizationConfig>
  ): void {
    // Update config
    if (config) {
      this.config = { ...this.config, ...config };
    }
    
    // Store data
    this.gridSize = cfdConfig.gridSize;
    this.velocityField = results.velocityField;
    this.pressureField = results.pressureField;
    
    // Clear previous visualizations
    this.clearVisualizations();
    
    // Generate visualizations
    if (this.config.showPressureContours) {
      this.createPressureContours();
    }
    
    if (this.config.showVelocityVectors) {
      this.createVelocityVectors();
    }
    
    if (this.config.showStreamlines) {
      this.createStreamlines();
    }
    
    if (this.config.showParticles) {
      this.createParticleSystem();
    }
    
    // Start animation
    this.startAnimation();
  }

  private clearVisualizations(): void {
    this.scene.remove(this.velocityVectorGroup);
    this.scene.remove(this.pressureContoursGroup);
    this.scene.remove(this.streamlinesGroup);
    
    if (this.particleSystem) {
      this.scene.remove(this.particleSystem);
      this.particleSystem = null;
    }
    
    this.velocityVectorGroup = new THREE.Group();
    this.pressureContoursGroup = new THREE.Group();
    this.streamlinesGroup = new THREE.Group();
    this.particles = [];
  }

  /**
   * Create pressure contours (color-coded planes)
   */
  private createPressureContours(): void {
    const { x: nx, y: ny, z: nz } = this.gridSize;
    
    // Find min and max pressure for normalization
    let minPressure = Infinity;
    let maxPressure = -Infinity;
    
    for (let i = 0; i < this.pressureField.length; i++) {
      minPressure = Math.min(minPressure, this.pressureField[i]);
      maxPressure = Math.max(maxPressure, this.pressureField[i]);
    }
    
    const pressureRange = maxPressure - minPressure;
    
    // Create slice planes at different Z positions
    const numSlices = Math.min(5, nz);
    const sliceSpacing = Math.floor(nz / numSlices);
    
    for (let slice = 0; slice < numSlices; slice++) {
      const k = slice * sliceSpacing;
      if (k >= nz) continue;
      
      // Create geometry for this slice
      const geometry = new THREE.PlaneGeometry(nx, ny, nx - 1, ny - 1);
      const positions = geometry.attributes.position.array as Float32Array;
      const colors = new Float32Array(positions.length);
      
      // Color each vertex based on pressure
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const idx = k * ny * nx + j * nx + i;
          const pressure = this.pressureField[idx];
          const normalized = (pressure - minPressure) / pressureRange;
          
          const vertexIdx = (j * nx + i) * 3;
          const color = this.getColorForValue(normalized);
          
          colors[vertexIdx] = color.r;
          colors[vertexIdx + 1] = color.g;
          colors[vertexIdx + 2] = color.b;
        }
      }
      
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      
      const material = new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7,
      });
      
      const plane = new THREE.Mesh(geometry, material);
      plane.position.set(0, 0, k - nz / 2);
      plane.rotation.x = -Math.PI / 2;
      
      this.pressureContoursGroup.add(plane);
    }
    
    this.scene.add(this.pressureContoursGroup);
  }

  /**
   * Create velocity vector arrows
   */
  private createVelocityVectors(): void {
    const { x: nx, y: ny, z: nz } = this.gridSize;
    
    // Sample every N points to avoid clutter
    const skip = Math.max(2, Math.floor(Math.min(nx, ny, nz) / 10));
    
    for (let k = skip; k < nz; k += skip) {
      for (let j = skip; j < ny; j += skip) {
        for (let i = skip; i < nx; i += skip) {
          const idx = k * ny * nx + j * nx + i;
          const vIdx = idx * 3;
          
          const u = this.velocityField[vIdx];
          const v = this.velocityField[vIdx + 1];
          const w = this.velocityField[vIdx + 2];
          
          const magnitude = Math.sqrt(u * u + v * v + w * w);
          
          if (magnitude > 0.001) {
            // Create arrow
            const origin = new THREE.Vector3(
              i - nx / 2,
              j - ny / 2,
              k - nz / 2
            );
            
            const direction = new THREE.Vector3(u, v, w).normalize();
            const length = magnitude * this.config.vectorScale;
            const color = this.getColorForValue(magnitude / 10); // Normalize velocity
            
            const arrow = new THREE.ArrowHelper(
              direction,
              origin,
              length,
              color.getHex(),
              length * 0.2,
              length * 0.15
            );
            
            this.velocityVectorGroup.add(arrow);
          }
        }
      }
    }
    
    this.scene.add(this.velocityVectorGroup);
  }

  /**
   * Create streamlines (flow paths)
   */
  private createStreamlines(): void {
    const { x: nx, y: ny, z: nz } = this.gridSize;
    const numStreamlines = 20;
    
    for (let s = 0; s < numStreamlines; s++) {
      // Start streamline at inlet
      let x = 0;
      let y = Math.random() * ny;
      let z = Math.random() * nz;
      
      const points: THREE.Vector3[] = [];
      const maxSteps = 100;
      
      for (let step = 0; step < maxSteps; step++) {
        // Get velocity at current position
        const i = Math.floor(x);
        const j = Math.floor(y);
        const k = Math.floor(z);
        
        if (i < 0 || i >= nx - 1 || j < 0 || j >= ny - 1 || k < 0 || k >= nz - 1) {
          break;
        }
        
        const idx = k * ny * nx + j * nx + i;
        const vIdx = idx * 3;
        
        const u = this.velocityField[vIdx];
        const v = this.velocityField[vIdx + 1];
        const w = this.velocityField[vIdx + 2];
        
        points.push(new THREE.Vector3(x - nx / 2, y - ny / 2, z - nz / 2));
        
        // Integrate forward
        const dt = 0.1;
        x += u * dt;
        y += v * dt;
        z += w * dt;
      }
      
      if (points.length > 2) {
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
          color: 0x00ff00,
          linewidth: 2,
        });
        const line = new THREE.Line(geometry, material);
        this.streamlinesGroup.add(line);
      }
    }
    
    this.scene.add(this.streamlinesGroup);
  }

  /**
   * Create animated particle system
   */
  private createParticleSystem(): void {
    const { x: nx, y: ny, z: nz } = this.gridSize;
    
    // Initialize particles
    for (let i = 0; i < this.config.particleCount; i++) {
      this.particles.push({
        position: new THREE.Vector3(
          Math.random() * nx - nx / 2,
          Math.random() * ny - ny / 2,
          Math.random() * nz - nz / 2
        ),
        velocity: new THREE.Vector3(0, 0, 0),
        age: Math.random() * 10,
        maxAge: 10,
      });
    }
    
    // Create particle geometry
    const positions = new Float32Array(this.particles.length * 3);
    const colors = new Float32Array(this.particles.length * 3);
    
    for (let i = 0; i < this.particles.length; i++) {
      positions[i * 3] = this.particles[i].position.x;
      positions[i * 3 + 1] = this.particles[i].position.y;
      positions[i * 3 + 2] = this.particles[i].position.z;
      
      colors[i * 3] = 1.0;
      colors[i * 3 + 1] = 1.0;
      colors[i * 3 + 2] = 1.0;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });
    
    this.particleSystem = new THREE.Points(geometry, material);
    this.scene.add(this.particleSystem);
  }

  /**
   * Update particle positions based on velocity field
   */
  private updateParticles(deltaTime: number): void {
    if (!this.particleSystem) return;
    
    const { x: nx, y: ny, z: nz } = this.gridSize;
    const positions = this.particleSystem.geometry.attributes.position.array as Float32Array;
    const colors = this.particleSystem.geometry.attributes.color.array as Float32Array;
    
    for (let i = 0; i < this.particles.length; i++) {
      const particle = this.particles[i];
      
      // Get velocity at particle position
      const gridX = Math.floor(particle.position.x + nx / 2);
      const gridY = Math.floor(particle.position.y + ny / 2);
      const gridZ = Math.floor(particle.position.z + nz / 2);
      
      if (gridX >= 0 && gridX < nx - 1 && 
          gridY >= 0 && gridY < ny - 1 && 
          gridZ >= 0 && gridZ < nz - 1) {
        
        const idx = gridZ * ny * nx + gridY * nx + gridX;
        const vIdx = idx * 3;
        
        particle.velocity.set(
          this.velocityField[vIdx],
          this.velocityField[vIdx + 1],
          this.velocityField[vIdx + 2]
        );
        
        // Update position
        particle.position.add(particle.velocity.clone().multiplyScalar(deltaTime));
        
        // Update age
        particle.age += deltaTime;
        
        // Color based on velocity magnitude
        const speed = particle.velocity.length();
        const color = this.getColorForValue(speed / 10);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      }
      
      // Reset particle if too old or out of bounds
      if (particle.age > particle.maxAge || 
          Math.abs(particle.position.x) > nx / 2 ||
          Math.abs(particle.position.y) > ny / 2 ||
          Math.abs(particle.position.z) > nz / 2) {
        
        particle.position.set(
          -nx / 2 + Math.random() * 2,
          Math.random() * ny - ny / 2,
          Math.random() * nz - nz / 2
        );
        particle.age = 0;
      }
      
      // Update buffer
      positions[i * 3] = particle.position.x;
      positions[i * 3 + 1] = particle.position.y;
      positions[i * 3 + 2] = particle.position.z;
    }
    
    this.particleSystem.geometry.attributes.position.needsUpdate = true;
    this.particleSystem.geometry.attributes.color.needsUpdate = true;
  }

  /**
   * Get color for a normalized value (0-1)
   */
  private getColorForValue(value: number): THREE.Color {
    // Clamp value
    value = Math.max(0, Math.min(1, value));
    
    switch (this.config.colorScheme) {
      case 'jet':
        return this.jetColormap(value);
      case 'viridis':
        return this.viridisColormap(value);
      case 'rainbow':
        return this.rainbowColormap(value);
      default:
        return this.jetColormap(value);
    }
  }

  private jetColormap(value: number): THREE.Color {
    const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * value - 3)));
    const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * value - 2)));
    const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * value - 1)));
    return new THREE.Color(r, g, b);
  }

  private viridisColormap(value: number): THREE.Color {
    // Simplified viridis approximation
    const r = 0.282 + value * 0.436;
    const g = value * 0.718;
    const b = 0.520 - value * 0.272;
    return new THREE.Color(r, g, b);
  }

  private rainbowColormap(value: number): THREE.Color {
    const hue = (1 - value) * 0.7; // Blue to red
    return new THREE.Color().setHSL(hue, 1.0, 0.5);
  }

  private startAnimation(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }
    
    let lastTime = performance.now();
    
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      
      const currentTime = performance.now();
      const deltaTime = (currentTime - lastTime) / 1000;
      lastTime = currentTime;
      
      // Update particles
      if (this.config.showParticles) {
        this.updateParticles(deltaTime * 0.1);
      }
      
      // Rotate camera slightly for better view
      this.camera.position.x = 50 * Math.cos(currentTime * 0.0001);
      this.camera.position.z = 50 * Math.sin(currentTime * 0.0001);
      this.camera.lookAt(0, 0, 0);
      
      this.renderer.render(this.scene, this.camera);
    };
    
    animate();
  }

  private handleResize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    
    this.renderer.setSize(width, height);
  }

  updateConfig(config: Partial<VisualizationConfig>): void {
    this.config = { ...this.config, ...config };
    
    // Re-visualize if data exists
    if (this.velocityField && this.pressureField) {
      this.clearVisualizations();
      
      if (this.config.showPressureContours) {
        this.createPressureContours();
      }
      
      if (this.config.showVelocityVectors) {
        this.createVelocityVectors();
      }
      
      if (this.config.showStreamlines) {
        this.createStreamlines();
      }
      
      if (this.config.showParticles) {
        this.createParticleSystem();
      }
    }
  }

  dispose(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }
    
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }
    
    this.renderer.dispose();
    this.scene.clear();
  }
}

export { CFDVisualizer, type VisualizationConfig };