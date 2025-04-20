# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
from janc.nondim import x0

Lx = 0.05/x0
Ly = 0.0125/x0

Nx = 2000
Ny = 500

n_block = [
    [1, 1],  # Level 0
    [100, 25], # Level 1
    [2, 2],  # Level 2
    [2, 2],  # Level 3
    [2, 2]   # Level 4
] # x-direction, y-direction

template_node_num = 3 

buffer_num = 5 

refinement_tolerance = { 
    'density': 10.0,
    'velocity': 0.5
}


'''AUTO Computation'''
n_grid = [[Nx // n_block[0][0], Ny // n_block[0][1]]] 
dx = [Lx/Nx] 
dy = [Ly/Ny]
for i, (bx, by) in enumerate(n_block[1:], 1):
    px, py = n_grid[-1]
    mult = 1 if i == 1 else 2
    if (px * mult) % bx != 0 or (py * mult) % by != 0:
        raise ValueError(f"Initial grid not divisible: {(px * mult)}%{bx}={(py * mult)%bx}, {(py * mult)}%{by}={(py * mult)%by}")
        break

    n_grid.append([(px * mult // bx) , (py * mult// by) ])
    dx.append(Lx/Nx / (2.0**i))
    dy.append(Ly/Ny / (2.0**i))


grid_mask_buffer_kernel = ( 
    jnp.zeros((2 * buffer_num + 1, 2 * buffer_num + 1))
        .at[buffer_num, :].set(1)
        .at[:, buffer_num].set(1)
        .at[buffer_num, buffer_num].set(0)
)

'''JANC config'''
def boundary_conditions(U,aux):
    U = jnp.concatenate([U,aux],axis=0)

    left_ghost = jnp.flip(U[:, 0:3, :], axis=1)
    left_ghost = left_ghost.at[1:3].multiply(-1)

    right_ghost = jnp.flip(U[:, -3:, :], axis=1)
    right_ghost = right_ghost.at[1:3].multiply(-1)

    U_with_ghost = jnp.concatenate([left_ghost, U, right_ghost], axis=1)

    lower_ghost = jnp.flip(U_with_ghost[:, :, 0:3], axis=2)
    lower_ghost = lower_ghost.at[2].multiply(-1) 

    upper_ghost = jnp.flip(U_with_ghost[:, :, -3:], axis=2)
    upper_ghost = upper_ghost.at[2].multiply(-1)

    U_with_ghost = jnp.concatenate([lower_ghost, U_with_ghost, upper_ghost], axis=2)

    U_with_ghost = U_with_ghost.at[1:3, :3, :].set(0.0)
    U_with_ghost = U_with_ghost.at[1:3, -3:, :].set(0.0)
    U_with_ghost = U_with_ghost.at[1:3, :, :3].set(0.0)
    U_with_ghost = U_with_ghost.at[1:3, :, -3:].set(0.0)

    return U_with_ghost[0:-2],U_with_ghost[-2:]#.astype(jnp.float32)


grid_set = {'Lx':Lx,'Ly':Ly,'nx':Nx,'ny':Ny}

thermo_set = {'is_detailed_chemistry':True,
          'chemistry_mechanism_diretory':'chem.txt',
          'thermo_model':'nasa7',
          'nasa7_mech':'gri30.yaml'}

boundary_set = {'boundary_conditions':boundary_conditions}
source_set = {'self_defined_source_terms':None}




