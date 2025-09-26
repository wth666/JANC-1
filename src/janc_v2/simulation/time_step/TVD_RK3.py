from jax import vmap
import jax.numpy as jnp
from jaxamr import amr

def advance_one_step(U,aux,dx,dy,dt,theta,rhs_func,update_func):
    U1 = U + rhs_func(U,aux,dx,dy,dt,theta)
    aux1 = update_func(U1, aux)
    U2 = 3/4*U + 1/4 * (U1 + rhs_func(U1,aux1,dx,dy,dt,theta))
    aux2 = update_func(U2, aux1)
    U3 = 1/3*U + 2/3 * (U2 + rhs_func(U2,aux2,dx,dy,dt,theta))
    aux3 = update_func(U3, aux2)
    return U3,aux3


def advance_one_step_amr(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, rhs_func, update_func):
    ghost_blk_data = amr.get_ghost_block_data(ref_blk_data, ref_blk_info)
    U,aux = ghost_blk_data[:,0:-2],ghost_blk_data[:,-2:]
    U1 = U + rhs_func(U, aux, dx, dy, dt, theta)
    aux1 = update_func(U1, aux)
    blk_data1 = jnp.concatenate([U1,aux1],axis=1)
    blk_data1 = amr.update_external_boundary(level, blk_data, blk_data1[..., 3:-3, 3:-3], ref_blk_info)

    ghost_blk_data1 = amr.get_ghost_block_data(blk_data1, ref_blk_info)
    U1,aux1 = ghost_blk_data1[:,0:-2],ghost_blk_data1[:,-2:]
    U2 = 3/4*U + 1/4*(U1 + rhs_func(U1, aux1, dx, dy, dt, theta))
    aux2 = update_func(U2, aux1)
    blk_data2 = jnp.concatenate([U2,aux2],axis=1)
    blk_data2 = amr.update_external_boundary(level, blk_data, blk_data2[..., 3:-3, 3:-3], ref_blk_info)

    ghost_blk_data2 = amr.get_ghost_block_data(blk_data2, ref_blk_info)
    U2,aux2 = ghost_blk_data2[:,0:-2],ghost_blk_data2[:,-2:]
    U3 = 1/3*U + 2/3*(U2 + rhs_func(U2, aux2, dx, dy, dt, theta))
    aux3 = update_func(U3, aux2)
    blk_data3 = jnp.concatenate([U3,aux3],axis=1)
    blk_data3 = amr.update_external_boundary(level, blk_data, blk_data3[..., 3:-3, 3:-3], ref_blk_info)
    return blk_data3

