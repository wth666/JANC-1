# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

Rg = 8.314463
P0 = 10 * 101325
T0 = 300
R0 = 369
x0 = 0.0125
rho0 = P0/(R0*T0)
M0 = Rg/R0
e0 = P0/rho0
u0 = jnp.sqrt(P0/rho0)
t0 = x0/u0
