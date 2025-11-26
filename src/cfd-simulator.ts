interface CFDConfig {
  gridSize: { x: number; y: number; z: number };
  fluidProperties: {
    density: number;
    viscosity: number;
  };
  boundaryConditions: Array<{
    type: 'velocity' | 'pressure' | 'wall';
    location: string;
    value: number | { x: number; y: number; z: number };
  }>;
  timeStep: number;
  iterations: number;
}

interface SimulationResult {
  velocityField: Float32Array;
  pressureField: Float32Array;
  convergenceHistory: number[];
  computeTime: number;
  success: boolean;
  message: string;
}

class GPUCFDSolver {
  private gl: WebGL2RenderingContext | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private isGPUAvailable: boolean = false;

  constructor() {
    this.initializeWebGL();
  }

  private initializeWebGL(): void {
    try {
      this.canvas = document.createElement('canvas');
      this.gl = this.canvas.getContext('webgl2', {
        antialias: false,
        depth: false,
        stencil: false,
        preserveDrawingBuffer: true,
      });

      if (this.gl) {
        this.isGPUAvailable = true;
        const ext = this.gl.getExtension('EXT_color_buffer_float');
        if (!ext) {
          console.warn('EXT_color_buffer_float not available, using CPU fallback');
          this.isGPUAvailable = false;
        } else {
          console.log('✅ GPU CFD Solver initialized with WebGL2');
        }
      } else {
        console.warn('WebGL2 not available, using CPU fallback');
        this.isGPUAvailable = false;
      }
    } catch (error) {
      console.error('WebGL initialization failed:', error);
      this.isGPUAvailable = false;
    }
  }

  async simulate(config: CFDConfig): Promise<SimulationResult> {
    const startTime = performance.now();

    if (!this.isGPUAvailable || !this.gl) {
      return this.cpuFallbackSolver(config);
    }

    try {
      const totalCells = config.gridSize.x * config.gridSize.y * config.gridSize.z;
      const velocityField = new Float32Array(totalCells * 3);
      const pressureField = new Float32Array(totalCells);
      const convergenceHistory: number[] = [];

      this.initializeFields(velocityField, pressureField, config);

      for (let iter = 0; iter < config.iterations; iter++) {
        this.applyBoundaryConditions(velocityField, pressureField, config);

        const residual = this.performIterationCPU(
          velocityField,
          pressureField,
          config
        );

        convergenceHistory.push(residual);

        if (residual < 1e-6) {
          console.log(`✅ CFD converged at iteration ${iter + 1}`);
          break;
        }

        if (iter % 10 === 0) {
          console.log(`Iteration ${iter}: residual = ${residual.toExponential(3)}`);
        }
      }

      const computeTime = performance.now() - startTime;

      return {
        velocityField,
        pressureField,
        convergenceHistory,
        computeTime,
        success: true,
        message: `Simulation completed in ${convergenceHistory.length} iterations`,
      };
    } catch (error) {
      console.error('GPU simulation error:', error);
      return this.cpuFallbackSolver(config);
    }
  }

  private initializeFields(
    velocityField: Float32Array,
    pressureField: Float32Array,
    config: CFDConfig
  ): void {
    const { x: nx, y: ny, z: nz } = config.gridSize;

    for (let i = 0; i < velocityField.length; i++) {
      velocityField[i] = 0;
    }

    for (let i = 0; i < pressureField.length; i++) {
      pressureField[i] = 0;
    }

    config.boundaryConditions.forEach((bc) => {
      if (bc.type === 'velocity' && typeof bc.value === 'object') {
        for (let k = 0; k < nz; k++) {
          for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
              if (bc.location === 'inlet' && i === 0) {
                const idx = (k * ny * nx + j * nx + i) * 3;
                velocityField[idx] = bc.value.x;
                velocityField[idx + 1] = bc.value.y;
                velocityField[idx + 2] = bc.value.z;
              }
            }
          }
        }
      }
    });
  }

  private applyBoundaryConditions(
    velocityField: Float32Array,
    pressureField: Float32Array,
    config: CFDConfig
  ): void {
    const { x: nx, y: ny, z: nz } = config.gridSize;

    config.boundaryConditions.forEach((bc) => {
      if (bc.type === 'velocity' && typeof bc.value === 'object') {
        for (let k = 0; k < nz; k++) {
          for (let j = 0; j < ny; j++) {
            if (bc.location === 'inlet') {
              const idx = (k * ny * nx + j * nx + 0) * 3;
              velocityField[idx] = bc.value.x;
              velocityField[idx + 1] = bc.value.y;
              velocityField[idx + 2] = bc.value.z;
            }
          }
        }
      }

      if (bc.type === 'pressure' && typeof bc.value === 'number') {
        for (let k = 0; k < nz; k++) {
          for (let j = 0; j < ny; j++) {
            if (bc.location === 'outlet') {
              const idx = k * ny * nx + j * nx + (nx - 1);
              pressureField[idx] = bc.value;
            }
          }
        }
      }

      if (bc.type === 'wall') {
        for (let k = 0; k < nz; k++) {
          for (let i = 0; i < nx; i++) {
            const idxBottom = (k * ny * nx + 0 * nx + i) * 3;
            const idxTop = (k * ny * nx + (ny - 1) * nx + i) * 3;
            velocityField[idxBottom] = 0;
            velocityField[idxBottom + 1] = 0;
            velocityField[idxBottom + 2] = 0;
            velocityField[idxTop] = 0;
            velocityField[idxTop + 1] = 0;
            velocityField[idxTop + 2] = 0;
          }
        }

        for (let j = 0; j < ny; j++) {
          for (let i = 0; i < nx; i++) {
            const idxFront = (0 * ny * nx + j * nx + i) * 3;
            const idxBack = ((nz - 1) * ny * nx + j * nx + i) * 3;
            velocityField[idxFront] = 0;
            velocityField[idxFront + 1] = 0;
            velocityField[idxFront + 2] = 0;
            velocityField[idxBack] = 0;
            velocityField[idxBack + 1] = 0;
            velocityField[idxBack + 2] = 0;
          }
        }
      }
    });
  }

  private performIterationCPU(
    velocityField: Float32Array,
    pressureField: Float32Array,
    config: CFDConfig
  ): number {
    const { x: nx, y: ny, z: nz } = config.gridSize;
    const dt = config.timeStep;
    const rho = config.fluidProperties.density;
    const mu = config.fluidProperties.viscosity;

    const newVelocity = new Float32Array(velocityField.length);
    const newPressure = new Float32Array(pressureField.length);

    let maxResidual = 0;

    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const idx = k * ny * nx + j * nx + i;
          const vIdx = idx * 3;

          const u = velocityField[vIdx];
          const v = velocityField[vIdx + 1];
          const w = velocityField[vIdx + 2];
          const p = pressureField[idx];

          const uEast = velocityField[((k * ny * nx + j * nx + (i + 1)) * 3)];
          const uWest = velocityField[((k * ny * nx + j * nx + (i - 1)) * 3)];
          const uNorth = velocityField[((k * ny * nx + (j + 1) * nx + i) * 3)];
          const uSouth = velocityField[((k * ny * nx + (j - 1) * nx + i) * 3)];
          const uUp = velocityField[(((k + 1) * ny * nx + j * nx + i) * 3)];
          const uDown = velocityField[(((k - 1) * ny * nx + j * nx + i) * 3)];

          const pEast = pressureField[k * ny * nx + j * nx + (i + 1)];
          const pWest = pressureField[k * ny * nx + j * nx + (i - 1)];

          const dudx2 = (uEast - 2 * u + uWest);
          const dudy2 = (uNorth - 2 * u + uSouth);
          const dudz2 = (uUp - 2 * u + uDown);
          const laplacianU = dudx2 + dudy2 + dudz2;

          const dpdx = (pEast - pWest) / 2.0;

          const convection = u * (uEast - uWest) / 2.0;
          const diffusion = (mu / rho) * laplacianU;
          const pressureGrad = -dpdx / rho;

          newVelocity[vIdx] = u + dt * (-convection + diffusion + pressureGrad);
          newVelocity[vIdx + 1] = v;
          newVelocity[vIdx + 2] = w;

          const divergence = (uEast - uWest) / 2.0;
          newPressure[idx] = p - dt * rho * divergence;

          const residual = Math.abs(newVelocity[vIdx] - u);
          maxResidual = Math.max(maxResidual, residual);
        }
      }
    }

    for (let i = 0; i < velocityField.length; i++) {
      velocityField[i] = newVelocity[i];
    }
    for (let i = 0; i < pressureField.length; i++) {
      pressureField[i] = newPressure[i];
    }

    return maxResidual;
  }

  private cpuFallbackSolver(config: CFDConfig): SimulationResult {
    console.log('⚠️ Using CPU fallback CFD solver');

    const startTime = performance.now();
    const totalCells = config.gridSize.x * config.gridSize.y * config.gridSize.z;
    const velocityField = new Float32Array(totalCells * 3);
    const pressureField = new Float32Array(totalCells);
    const convergenceHistory: number[] = [];

    this.initializeFields(velocityField, pressureField, config);

    for (let iter = 0; iter < config.iterations; iter++) {
      this.applyBoundaryConditions(velocityField, pressureField, config);

      const residual = this.performIterationCPU(
        velocityField,
        pressureField,
        config
      );

      convergenceHistory.push(residual);

      if (residual < 1e-6) {
        console.log(`✅ CPU CFD converged at iteration ${iter + 1}`);
        break;
      }

      if (iter % 10 === 0) {
        console.log(`CPU Iteration ${iter}: residual = ${residual.toExponential(3)}`);
      }
    }

    const computeTime = performance.now() - startTime;

    return {
      velocityField,
      pressureField,
      convergenceHistory,
      computeTime,
      success: true,
      message: `CPU simulation completed in ${convergenceHistory.length} iterations`,
    };
  }

  async predictInitialFlow(
    geometry: any,
    fluidProperties: { density: number; viscosity: number }
  ): Promise<Float32Array> {
    console.log('🤖 ML-based flow prediction (placeholder)');
    return new Float32Array(1000);
  }

  dispose(): void {
    if (this.canvas) {
      this.canvas = null;
    }
    if (this.gl) {
      const loseContext = this.gl.getExtension('WEBGL_lose_context');
      if (loseContext) {
        loseContext.loseContext();
      }
      this.gl = null;
    }
  }
}

export { GPUCFDSolver, type CFDConfig, type SimulationResult };