import jax.numpy as jnp
from jax import jit
from ..solver import aux_func
from ..thermodynamics import thermo

p = 2
eps = 1e-6
C1 = 1 / 10
C2 = 3 / 5
C3 = 3 / 10

@jit
def flux(rho,u,v,p,Y,h):
    rhoE = rho*h-p+0.5*rho*(u**2+v**2)
    U = jnp.concatenate([rho,rho*u,rho*v,rhoE,rho*Y],axis=0)
    F = jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p), rho * u * Y], axis=0) 
    G = jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p), rho * v * Y], axis=0)
    return U,F,G
    
@jit
def KNP_flux(rhoL, uL, vL, pL, YL, aL, hL, rhoR, uR, vR, pR, YR, aR, hR):

    u_nL = uL
    u_nR = uR
    
    lambda_L_min = u_nL - aL
    lambda_L_max = u_nL + aL
    lambda_R_min = u_nR - aR
    lambda_R_max = u_nR + aR

    a_minus_F = jnp.minimum(0.0, jnp.minimum(lambda_L_min, lambda_R_min))
    a_plus_F  = jnp.maximum(0.0, jnp.maximum(lambda_L_max, lambda_R_max))

    denom_F = a_plus_F - a_minus_F + 1e-10

    u_nL = vL
    u_nR = vR
    
    lambda_L_min = u_nL - aL
    lambda_L_max = u_nL + aL
    lambda_R_min = u_nR - aR
    lambda_R_max = u_nR + aR

    a_minus_G = jnp.minimum(0.0, jnp.minimum(lambda_L_min, lambda_R_min))
    a_plus_G = jnp.maximum(0.0, jnp.maximum(lambda_L_max, lambda_R_max))

    denom_G = a_plus_G - a_minus_G + 1e-10
    
    UL,FL, GL = flux(rhoL, uL, vL, pL, YL, hL)
    UR,FR, GR = flux(rhoR, uR, vR, pR, YR, hR)
    
    F_KNP = (
        (a_plus_F * FL - a_minus_F * FR) / denom_F +
        (a_plus_F * a_minus_F) / denom_F * (UR - UL)
    )

    G_KNP = (
        (a_plus_G * GL - a_minus_G * GR) / denom_G +
        (a_plus_G * a_minus_G) / denom_G * (UR - UL)
    )
    return F_KNP,G_KNP


@jit
def splitFlux_LF(ixy, U, aux):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[3:4,:,:]

    zx = (ixy == 1) * 1
    zy = (ixy == 2) * 1

    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p), rho * u * Y], axis=0) + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p), rho * v * Y], axis=0)
    um = jnp.nanmax(abs(u) + a)
    vm = jnp.nanmax(abs(v) + a)
    theta = zx*um + zy*vm
    Hplus = 0.5 * (F + theta * U)
    Hminus = 0.5 * (F - theta * U)
    return Hplus, Hminus

@jit
def WENO_plus_x(f):
    fj = f[:,2:-3,3:-3]
    fjp1 = f[:,3:-2,3:-3]
    fjp2 = f[:,4:-1,3:-3]
    fjm1 = f[:,1:-4,3:-3]
    fjm2 = f[:,0:-5,3:-3]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    #dfj = fj_halfp[:,1:,:] - fj_halfp[:,0:-1,:]
    return fj_halfp

@jit
def WENO_plus_y(f):

    fj = f[:,3:-3,2:-3]
    fjp1 = f[:,3:-3,3:-2]
    fjp2 = f[:,3:-3,4:-1]
    fjm1 = f[:,3:-3,1:-4]
    fjm2 = f[:,3:-3,0:-5]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    #dfj = fj_halfp[:,:,1:] - fj_halfp[:,:,0:-1]

    return fj_halfp

@jit
def WENO_minus_x(f):

    fj = f[:,3:-2,3:-3]
    fjp1 = f[:,4:-1,3:-3]
    fjp2 = f[:,5:,3:-3]
    fjm1 = f[:,2:-3,3:-3]
    fjm2 = f[:,1:-4,3:-3]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    #dfj = (fj_halfm[:,1:,:] - fj_halfm[:,0:-1,:])

    return fj_halfm

@jit
def WENO_minus_y(f):

    fj = f[:,3:-3,3:-2]
    fjp1 = f[:,3:-3,4:-1]
    fjp2 = f[:,3:-3,5:]
    fjm1 = f[:,3:-3,2:-3]
    fjm2 = f[:,3:-3,1:-4]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    #dfj = (fj_halfm[:,:,1:] - fj_halfm[:,:,0:-1])

    return fj_halfm

@jit
def weno5(U,aux,dx,dy):
    Fplus, Fminus = splitFlux_LF(1, U, aux)
    Gplus, Gminus = splitFlux_LF(2, U, aux)

    dFp = WENO_plus_x(Fplus)
    dFp = (dFp[:,1:,:] - dFp[:,0:-1,:])
    dFm = WENO_minus_x(Fminus)
    dFm = (dFm[:,1:,:] - dFm[:,0:-1,:])

    dGp = WENO_plus_y(Gplus)
    dGp = (dGp[:,:,1:] - dGp[:,:,0:-1])
    dGm = WENO_minus_y(Gminus)
    dGm = (dGm[:,:,1:] - dGm[:,:,0:-1])

    dF = dFp + dFm
    dG = dGp + dGm

    netflux = dF/dx + dG/dy

    return -netflux

@jit
def weno5_KNP(U, aux, dx, dy):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    Y = U[4:]/U[0:1]
    var_p = jnp.concatenate([rho, u, v, p, Y], axis=0)

    var_p_l = WENO_plus_x(var_p)
    var_p_r = WENO_minus_x(var_p)

    rho_l,u_l,v_l,p_l,Y_l = var_p_l[0:1],var_p_l[1:2],var_p_l[2:3],var_p_l[3:4],var_p_l[4:]
    R_l = thermo.get_R(Y_l)
    T_l = p_l/(rho_l*R_l)
    _, gamma_l, h_l, _, _ = thermo.get_thermo(T_l,Y_l)
    a_l = jnp.sqrt(gamma_l*R_l*T_l)
    
    rho_r,u_r,v_r,p_r,Y_r = var_p_r[0:1],var_p_r[1:2],var_p_r[2:3],var_p_r[3:4],var_p_r[4:]
    R_r = thermo.get_R(Y_r)
    T_r = p_r/(rho_r*R_r) 
    _, gamma_r, h_r, _, _ = thermo.get_thermo(T_r,Y_r)
    a_r = jnp.sqrt(gamma_r*R_r*T_r)
    
    F_knp,G_knp = KNP_flux(rho_l, u_l, v_l, p_l, Y_l, a_l, h_l, rho_r, u_r, v_r, p_r, Y_r, a_r, h_r)
    dF = (F_knp[:, 1:, :] - F_knp[:, :-1, :])
    dG = (G_knp[:, :, 1:] - G_knp[:, :, :-1])    
    
    netflux = dF/dx + dG/dy

    return -netflux
