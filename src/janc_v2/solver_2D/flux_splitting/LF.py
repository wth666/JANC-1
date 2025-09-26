import jax.numpy as jnp
from .. import aux_func

def split_flux(ixy, U, aux):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[3:4,:,:]
    zx = (ixy == 1) * 1.0
    zy = (ixy == 2) * 1.0
    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p), rho * u * Y], axis=0) + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p), rho * v * Y], axis=0)
    um = jnp.nanmax(abs(u) + a)
    vm = jnp.nanmax(abs(v) + a)
    theta = zx*um + zy*vm
    Hplus = 0.5  * (F + theta * U)
    Hminus = 0.5  * (F - theta * U)
    
    return Hplus, Hminus

