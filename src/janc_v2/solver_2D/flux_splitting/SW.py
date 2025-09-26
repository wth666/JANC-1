import jax.numpy as jnp
from .. import aux_func

def split_flux(ixy, U, aux):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[3:4,:,:]
    gamma = aux[0:1]
    
    zx = (ixy == 1) * 1.0
    zy = (ixy == 2) * 1.0
    theta = zx * u + zy * v

    H1 = (1 / (2 * gamma)) * jnp.concatenate([rho, rho * u - rho * a * zx, rho * v - rho * a * zy,
                          rhoE + p - rho * a * theta, rho * Y], axis=0)
    H2 = ((gamma - 1) / gamma) * jnp.concatenate(
        [rho, rho * u, rho * v, 0.5 * rho * (u ** 2 + v ** 2), rho * Y], axis=0)
    H4 = (1 / (2 * gamma)) * jnp.concatenate([rho, rho * u + rho * a * zx, rho * v + rho * a * zy,
                          rhoE + p + rho * a * theta, rho * Y], axis=0)

    lambda1 = theta - a
    lambda2 = theta
    lambda4 = theta + a
    eps = 1e-6

    lap1 = 0.5 * (lambda1 + jnp.sqrt(jnp.power(lambda1, 2) + eps**2))
    lam1 = 0.5 * (lambda1 - jnp.sqrt(jnp.power(lambda1, 2) + eps**2))

    lap2 = 0.5 * (lambda2 + jnp.sqrt(jnp.power(lambda2, 2) + eps**2))
    lam2 = 0.5 * (lambda2 - jnp.sqrt(jnp.power(lambda2, 2) + eps**2))

    lap4 = 0.5 * (lambda4 + jnp.sqrt(jnp.power(lambda4, 2) + eps**2))
    lam4 = 0.5 * (lambda4 - jnp.sqrt(jnp.power(lambda4, 2) + eps**2))

    Hplus = lap1 * H1 + lap2 * H2 + lap4 * H4
    Hminus = lam1 * H1 + lam2 * H2 + lam4 * H4

    return Hplus, Hminus
