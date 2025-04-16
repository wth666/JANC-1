# JANC: A cost-effective, differentiable compressible reacting flow solver featured with JAX-based adaptive mesh refinement

JANC, as the abbreviation for “JAX-AMR & Combustion”, is a fully-differentiable compressible reacting flow solver based on [JAX-AMR](https://github.com/JA4S/JAX-AMR)

Authors:
- [Haocheng Wen](https://github.com/thuwen)
- [Faxuan Luo](https://github.com/luofx23)

## Basic features of JANC
- Implementation of adaptive mesh refinement (AMR) based on JAX, namely JAX-AMR, providing a feasible AMR framework for large-scale fully-differentiable computation.
- Adoption of structured Cartesian grid, high-order finite difference method, point-implicit chemical source advancing in the solver.
- Inheriting the basic features of JAX, including fully differentiable, compatible with CPUs/GPUs/TPUs computation, and convenient parallel management.
- Programmed by Python, allowing rapid and efficient prototyping of projects.

## Physical models and numerical methods
- Adaptive mesh refinenment (JAX-AMR)
- Explicit time advancing (RK3)
- High-order adaptive spatial reconstruction (WENO-5)
- Riemann solvers (Lax-Friedrichs)
- Point-implicit chemical source advancing
- CPU/GPU/TPU capability
- Parallel computation on GPU/TPU (only for the core solver in current version)

For the details, please refer to our [paper](xxx).


## License
This project is licensed under the MIT License - see 
the [LICENSE](LICENSE) file or for details https://en.wikipedia.org/wiki/MIT_License.
