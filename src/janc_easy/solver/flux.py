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
def splitFlux_LF(ixy, U):
    rho,u,v,p,a = aux_func.U_to_prim(U)
    rhoE = U[3:4,:,:]

    zx = (ixy == 1) * 1
    zy = (ixy == 2) * 1

    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p)], axis=0) + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p)], axis=0)
    um = abs(u) + a#jnp.nanmax(abs(u) + a)
    vm = abs(v) + a#jnp.nanmax(abs(v) + a)
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
def weno5(U,dx,dy):
    Fplus, Fminus = splitFlux_LF(1, U)
    Gplus, Gminus = splitFlux_LF(2, U)

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

def HLLC_flux(q_L_x,q_R_x,q_L_y,q_R_y):
    rho_L_x,rho_R_x,rho_L_y,rho_R_y =  q_L_x[0:1],q_R_x[0:1],q_L_y[0:1],q_R_y[0:1] 
    p_L_x,p_R_x,p_L_y,p_R_y = q_L_x[3:4],q_R_x[3:4],q_L_y[3:4],q_R_y[3:4]
    #x-flux
    a_L = jnp.sqrt(thermo.gamma*p_L_x/rho_L_x)
    a_R = jnp.sqrt(thermo.gamma*p_R_x/rho_R_x)
    u_L,u_R = q_L_x[1:2],q_R_x[1:2]
    v_L,v_R = q_L_x[2:3],q_R_x[2:3]
    E_L = p_L_x/(thermo.gamma-1)+0.5*rho_L_x*(u_L**2+v_L**2)
    E_R = p_R_x/(thermo.gamma-1)+0.5*rho_R_x*(u_R**2+v_R**2)
    
    s_L = jnp.minimum(u_L-a_L,u_R-a_R)
    s_R = jnp.maximum(u_L+a_L,u_R+a_R)
    s_M = (p_R_x-p_L_x+rho_L_x*u_L*(s_L-u_L)-rho_R_x*u_R*(s_R-u_R))/(rho_L_x*(s_L-u_L)-rho_R_x*(s_R-u_R))
    U_L = jnp.concatenate([rho_L_x,rho_L_x*u_L,rho_L_x*v_L,E_L],axis=0)
    U_L_M = rho_L_x*(s_L-u_L)/(s_L-s_M)*jnp.concatenate([jnp.ones_like(s_M),s_M,v_L,E_L/rho_L_x+(s_M-u_L)*(s_M+p_L_x/(rho_L_x*(s_L-u_L)))],axis=0)
    F_L = jnp.concatenate([rho_L_x*u_L,rho_L_x*u_L**2+p_L_x,rho_L_x*u_L*v_L,
                           u_L*(E_L+p_L_x)],axis=0)
    F_L_M = F_L + s_L*(U_L_M-U_L)
    U_R = jnp.concatenate([rho_R_x,rho_R_x*u_R,rho_R_x*v_R,E_R],axis=0)
    U_R_M = rho_R_x*(s_R-u_R)/(s_R-s_M)*jnp.concatenate([jnp.ones_like(s_M),s_M,v_R,E_R/rho_R_x+(s_M-u_R)*(s_M+p_R_x/(rho_R_x*(s_R-u_R)))],axis=0)
    F_R = jnp.concatenate([rho_R_x*u_R,rho_R_x*u_R**2+p_R_x,rho_R_x*u_R*v_R,
                           u_R*(E_R+p_R_x)],axis=0)
    F_R_M = F_R + s_R*(U_R_M-U_R)
    
    cond1 = s_L>=0
    cond2 = jnp.logical_and(s_L<=0,s_M>=0)
    cond3 = jnp.logical_and(s_M<=0,s_R>=0)
    
    F1 = jnp.where(cond1,F_L,F_R)
    F2 = jnp.where(cond2,F_L_M,F1)
    F_HLLC = jnp.where(cond3,F_R_M,F2)
    
    #y-flux
    a_L = jnp.sqrt(thermo.gamma*p_L_y/rho_L_y)
    a_R = jnp.sqrt(thermo.gamma*p_R_y/rho_R_y)
    u_L,u_R = q_L_y[1:2],q_R_y[1:2]
    v_L,v_R = q_L_y[2:3],q_R_y[2:3]
    E_L = p_L_y/(thermo.gamma-1)+0.5*rho_L_y*(u_L**2+v_L**2)
    E_R = p_R_y/(thermo.gamma-1)+0.5*rho_R_y*(u_R**2+v_R**2)
    s_L = jnp.minimum(v_L-a_L,v_R-a_R)
    s_R = jnp.maximum(v_L+a_L,v_R+a_R)
    s_M = (p_R_y-p_L_y+rho_L_y*v_L*(s_L-v_L)-rho_R_y*v_R*(s_R-v_R))/(rho_L_y*(s_L-v_L)-rho_R_y*(s_R-v_R))
    U_L = jnp.concatenate([rho_L_y,rho_L_y*u_L,rho_L_y*v_L,E_L],axis=0)
    U_L_M = rho_L_y*(s_L-v_L)/(s_L-s_M)*jnp.concatenate([jnp.ones_like(s_M),u_L,s_M,E_L/rho_L_y+(s_M-v_L)*(s_M+p_L_y/(rho_L_y*(s_L-v_L)))],axis=0)
    G_L = jnp.concatenate([rho_L_y*v_L,rho_L_y*u_L*v_L,rho_L_y*v_L**2+p_L_y,
                           v_L*(E_L+p_L_y)],axis=0)
    G_L_M = G_L + s_L*(U_L_M-U_L)
    U_R = jnp.concatenate([rho_R_y,rho_R_y*u_R,rho_R_y*v_R,E_R],axis=0)
    U_R_M = rho_R_y*(s_R-v_R)/(s_R-s_M)*jnp.concatenate([jnp.ones_like(s_M),u_R,s_M,E_R/rho_R_y+(s_M-v_R)*(s_M+p_R_y/(rho_R_y*(s_R-v_R)))],axis=0)
    G_R = jnp.concatenate([rho_R_y*v_R,rho_R_y*u_R*v_R,rho_R_y*v_R**2+p_R_y,
                           v_R*(E_R+p_R_y)],axis=0)
    G_R_M = G_R + s_R*(U_R_M-U_R)
    
    cond1 = s_L>=0
    cond2 = jnp.logical_and(s_L<=0,s_M>=0)
    cond3 = jnp.logical_and(s_M<=0,s_R>=0)
    
    G1 = jnp.where(cond1,G_L,G_R)
    G2 = jnp.where(cond2,G_L_M,G1)
    G_HLLC = jnp.where(cond3,G_R_M,G2)
    
    return F_HLLC,G_HLLC



def to_characteristic_x(q,u,v,H,a):
    gamma = thermo.gamma
    gamma1 = gamma - 1
    D = -2*a**2/gamma1
    Lx = jnp.stack([
        jnp.concatenate([
            (-2*H*u - a*u**2 - a*v**2 + u**3 + u*v**2)/(2*a*D),
            (H + a*u - 0.5*(u**2+v**2))/(a*D),
            v/D,
            -1.0/D
        ], axis=0),
        
        jnp.concatenate([
            2*(-H + u**2 + v**2)/D,
            -2*u/D,
            -2*v/D,
            2.0/D
        ], axis=0),
        
        jnp.concatenate([-v, jnp.zeros_like(v), jnp.ones_like(v), jnp.zeros_like(v)], axis=0),
        
        jnp.concatenate([
            (2*H*u - a*u**2 - a*v**2 - u**3 - u*v**2)/(2*a*D),
            (-H + a*u + 0.5*(u**2+v**2))/(a*D),
            v/D,
            -1.0/D
        ], axis=0)
    ], axis=0)  # (4,4,Nx,Ny)
    
    return jnp.einsum('ijxy,jxy->ixy', Lx, q)

def from_characteristic_x(q_char,u,v,H,a):
    Rx = jnp.stack([
        jnp.concatenate([jnp.ones_like(u), jnp.ones_like(u), jnp.zeros_like(u), jnp.ones_like(u)], axis=0),
        jnp.concatenate([u-a, u, jnp.zeros_like(u), u+a], axis=0),
        jnp.concatenate([v, v, jnp.ones_like(u), v], axis=0),
        jnp.concatenate([H-u*a, 0.5*(u**2+v**2), v, H+u*a], axis=0)
    ], axis=0)  # (4,4,Nx,Ny)
    
    return jnp.einsum('ijxy,jxy->ixy', Rx, q_char)

# ========================
# Y direction
# ========================

def to_characteristic_y(q,u,v,H,a):
    gamma = thermo.gamma
    gamma1 = gamma - 1
    D = -2*a**2/gamma1
    
    Ly = jnp.stack([
        jnp.concatenate([
            (-2*H*v - a*u**2 - a*v**2 + v**3 + u**2*v)/(2*a*D),
            (H + a*v - 0.5*(u**2+v**2))/(a*D),
            u/D,
            -1.0/D
        ], axis=0),
        
        jnp.concatenate([
            2*(-H + u**2 + v**2)/D,
            -2*u/D,
            -2*v/D,
            2.0/D
        ], axis=0),
        
        jnp.concatenate([-u, jnp.ones_like(u), jnp.zeros_like(u), jnp.zeros_like(u)], axis=0),
        
        jnp.concatenate([
            (2*H*v - a*u**2 - a*v**2 - v**3 - u**2*v)/(2*a*D),
            (-H + a*v + 0.5*(u**2+v**2))/(a*D),
            u/D,
            -1.0/D
        ], axis=0)
    ], axis=0)  # (4,4,Nx,Ny)
    
    return jnp.einsum('ijxy,jxy->ixy', Ly, q)

def from_characteristic_y(q_char,u,v,H,a):
    Ry = jnp.stack([
        jnp.concatenate([jnp.ones_like(u), jnp.ones_like(u), jnp.zeros_like(u), jnp.ones_like(u)], axis=0),
        jnp.concatenate([u, u, jnp.ones_like(u), u], axis=0),
        jnp.concatenate([v-a, v, jnp.zeros_like(u), v+a], axis=0),
        jnp.concatenate([H-v*a, 0.5*(u**2+v**2), u, H+v*a], axis=0)
    ], axis=0)  # (4,4,Nx,Ny)
    
    return jnp.einsum('ijxy,jxy->ixy', Ry, q_char)

def T_matrix(q,rho,u,v):
    gamma = thermo.gamma
    gamma1 = gamma-1
    rho, u, v, E, p = primitive_vars(U)
    T = jnp.stack([
        jnp.concatenate([jnp.ones_like(rho), jnp.zeros_like(rho), jnp.zeros_like(rho), jnp.zeros_like(rho)], axis=0),
        jnp.concatenate([u, rho, jnp.zeros_like(rho), jnp.zeros_like(rho)], axis=0),
        jnp.concatenate([v, jnp.zeros_like(rho), rho, jnp.zeros_like(rho)], axis=0),
        jnp.concatenate([0.5*(u**2+v**2), rho*u, rho*v, 1.0/gamma1], axis=0)
    ], axis=0)  # (4,4,Nx,Ny)
    return jnp.einsum("ijxy,jxy->ixy", T, q)

def Tinv_matrix(q,rho,u,v):
    gamma = thermo.gamma
    gamma1 = gamma-1
    Tinv = jnp.stack([
        jnp.concatenate([jnp.ones_like(rho), jnp.zeros_like(rho), jnp.zeros_like(rho), jnp.zeros_like(rho)], axis=0),
        jnp.concatenate([-u/rho, 1.0/rho, jnp.zeros_like(rho), jnp.zeros_like(rho)], axis=0),
        jnp.concatenate([-v/rho, jnp.zeros_like(rho), 1.0/rho, jnp.zeros_like(rho)], axis=0),
        jnp.concatenate([gamma1*(0.5*(u**2+v**2)),
                   -gamma1*u,
                   -gamma1*v,
                   gamma1], axis=0)
    ], axis=0)  # (4,4,Nx,Ny)
    return jnp.einsum("ijxy,jxy->ixy", Tinv, qc)



    


@jit
def HLLC(U,dx,dy):
    rho,u,v,p,a = aux_func.U_to_prim(U)
    q = jnp.concatenate([rho,u,v,p],axis=0)
    q_L_x = WENO_plus_x(q)
    q_R_x = WENO_minus_x(q)
    q_L_y = WENO_plus_y(q)
    q_R_y = WENO_minus_y(q)
    F,G = HLLC_flux(q_L_x,q_R_x,q_L_y,q_R_y)
    dF = F[:,1:,:]-F[:,:-1,:]
    dG = G[:,:,1:]-G[:,:,:-1]
    netflux = dF/dx + dG/dy
    return -netflux






