import jax.numpy as jnp
from .. import aux_func

def split_flux(U, aux, metrics):
    rho,u,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[2:3]
    gamma = aux[0:1]
    J = 1.0
    zx = 1.0
    theta = zx * u

    H1 = J*(1 / (2 * gamma)) * jnp.concatenate([rho, rho * u - rho * a * zx,
                          rhoE + p - rho * a * theta, rho * Y], axis=0)
    H2 = J*((gamma - 1) / gamma) * jnp.concatenate(
        [rho, rho * u, 0.5 * rho * (u ** 2), rho * Y], axis=0)
    H4 = J*(1 / (2 * gamma)) * jnp.concatenate([rho, rho * u + rho * a * zx,
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
