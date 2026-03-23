# DDPM vs Flow Matching: A Comparative Study (Research Implementation)

**Associated Research Paper:**  
[Efficiency vs. Fidelity: A Comparative Analysis of Diffusion Probabilistic Models and Flow Matching on Low-Resource Hardware](https://arxiv.org/pdf/2511.19379)\\
This repository contains the implementation, experiments, and additional analysis from our research paper.

## Paper Overview

This work investigates the trade-off between: Efficiency (computation) and Fidelity (sample quality)

We compare:
- Diffusion Models (DDPM)
- Flow Matching

## My Contribution

- Implemented DDPM and Flow Matching under a unified framework  
- Designed experiments to compare efficiency and sample quality  
- Performed trajectory curvature and solver sensitivity analysis  
- Generated visualizations to study transport dynamics and sampling behavior  

## Motivation
Diffusion models have become dominant in generative AI, but Flow Matching offers:
- Faster sampling
- More stable trajectories
- ODE-based formulation

## Methodology

### Shared Architecture
- U-Net backbone
- Sinusoidal time embeddings

### Two Modes
1. **Diffusion (DDPM)**
   - Learns to predict noise
   - Reverse stochastic process

2. **Flow Matching**
   - Learns velocity field
   - Deterministic ODE transport

## Key Insights

- Flow Matching produces straighter transport paths
- Diffusion introduces more stochasticity
- Flow models can achieve similar quality with fewer steps
- Solver choice significantly affects output quality
