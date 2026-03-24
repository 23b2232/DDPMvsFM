# Results

## 1. Generated Samples

### Flow Matching
![Flow Samples](flow_samples.png)

### Diffusion (DDPM)
![Diffusion Samples](diffusion_samples.png)


## 2. Generation Dynamics

### Flow
![Flow Dynamics](dynamics_flow.png)

### Diffusion
![Diffusion Dynamics](dynamics_diffusion.png)


## 3. Latent Space Interpolation

### Flow
![Flow Interpolation](latent_flow.png)

### Diffusion
![Diffusion Interpolation](latent_diffusion.png)


## 4. Transport Efficiency

### Flow
![Flow Efficiency](efficiency_flow.png)

### Diffusion
![Diffusion Efficiency](efficiency_diffusion.png)


## 5. Curvature Analysis

![Curvature Histogram](curvature_histogram.png)


## 6. Vector Field Visualization

### Flow
![Flow Field](vector_field_flow.png)

### Diffusion
![Diffusion Field](vector_field_diffusion.png)


## 7. Solver Comparison (Flow)

![Solver Comparison](solver_comparison.png)


## Key Observations

- Flow Matching produces smoother and more structured samples
- Diffusion introduces higher stochasticity
- Flow trajectories are closer to optimal transport (lower curvature)
- Solver choice significantly impacts Flow-based generation

## Stats

Flow Mean Curvature: ~1.02
Diffusion Mean Curvature: ~1.05
**Observation:** Flow is closer to optimal transport (1.0)
