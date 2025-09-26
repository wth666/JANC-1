import jax
import jax.numpy as jnp
from .boundary_padding import pad_1D, pad_2D
from ..boundary import boundary

##parallel settings##
devices = jax.devices()
num_devices = len(devices)

def boundary_conditions_1D(U,aux,theta=None):
    device_idx = jax.lax.axis_index('x')
    U_periodic_pad,aux_periodic_pad = pad_1D(U,aux)
    U_with_lb,aux_with_lb = jax.lax.cond(device_idx==0,lambda:boundary.boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad,theta),lambda:(U_periodic_pad,aux_periodic_pad))
    U_with_ghost_cell,aux_with_ghost_cell = jax.lax.cond(device_idx==(num_devices-1),lambda:boundary.boundary_func['right_boundary'](U_with_lb,aux_with_lb,theta),lambda:(U_with_lb,aux_with_lb))
    return U_with_ghost_cell,aux_with_ghost_cell


def boundary_conditions_2D(U,aux,theta=None):
    device_idx = jax.lax.axis_index('x')
    U_periodic_pad,aux_periodic_pad = pad_2D(U,aux)
    U_with_lb,aux_with_lb = jax.lax.cond(device_idx==0,lambda:boundary.boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad,theta),lambda:(U_periodic_pad,aux_periodic_pad))
    U_with_rb,aux_with_rb = jax.lax.cond(device_idx==(num_devices-1),lambda:boundary.boundary_func['right_boundary'](U_with_lb,aux_with_lb,theta),lambda:(U_with_lb,aux_with_lb))
    U_with_bb,aux_with_bb = boundary.boundary_func['bottom_boundary'](U_with_rb,aux_with_rb,theta)
    U_with_ghost_cell,aux_with_ghost_cell = boundary.boundary_func['top_boundary'](U_with_bb,aux_with_bb,theta)
    return U_with_ghost_cell,aux_with_ghost_cell

