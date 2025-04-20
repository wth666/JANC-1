# JANC: A cost-effective, differentiable compressible reacting flow solver featured with JAX-based adaptive mesh refinement

JANC, as the abbreviation for “JAX-AMR & Combustion”, is a fully-differentiable compressible reacting flow solver based on [JAX-AMR](https://github.com/JA4S/JAX-AMR).

Authors:
- [Haocheng Wen](https://github.com/thuwen)
- [Faxuan Luo](https://github.com/luofx23)

## Basic features of JANC
- Conjunction with JAX-AMR, allowing cost-effective large-scale simulations.
- Adoption of structured Cartesian grid, dimensionless equations,  high-order finite difference method, point-implicit chemical source advancing in the solver.
- Inheriting the basic features of JAX, including fully differentiable, compatible with CPUs/GPUs/TPUs computation, and convenient parallel management.
- Programmed by Python, allowing rapid and efficient prototyping of projects.

## Physical models and numerical methods
- Adaptive mesh refinenment (JAX-AMR)
- Dimensionless computation
- Explicit time advancing (RK3)
- High-order spatial reconstruction (WENO-5)
- Riemann solvers (Lax-Friedrichs)
- Point-implicit chemical source advancing
- CPU/GPU/TPU capability
- Parallel computation on GPU/TPU (only for the core solver in current version)

For the details, please refer to our [paper](xxx).

## Quick Installation
 The python files related to AMR should be cloned from [JAX-AMR](https://github.com/JA4S/JAX-AMR) repository, including jaxamr.py and amraux.py.

## Example

Rotating detonation combustor (RDC) simulation on 1,600,000 grids with 9sp-19r-H2-Air detailed reaction achieved within 45 minutes on single A100 GPU:
![image](https://github.com/JA4S/JANC/blob/main/RDC_example.gif)
Detonation tube simulation on 4,000,000 grids with 9sp-19r-H2-Air detailed reaction achieved within 1 hour on single A100 GPU:
![image](https://github.com/JA4S/JANC/blob/main/detonation_tube_example.gif)

## State of the Project

- [x] 2D solver for Euler equations  ✅
- [x] conjuction with the CFD solver ✅
- [x] Parallel compuation for the core solver ✅
- [ ] 3D solver for Navier-Stocks equations (soon)
- [ ] Implicit time advancing (soon)
- [ ] Turbulence model
- [ ] DPM model based on Euler-lagrange method
- [ ] Mixing-precision computation
- [ ] Parallel computation with JAX-AMR


## License
This project is licensed under the MIT License - see 
the [LICENSE](LICENSE) file or for details https://en.wikipedia.org/wiki/MIT_License.
