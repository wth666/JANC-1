import jax.numpy as jnp
from ..model import thermo_model as thermo

user_source = None

def set_source_terms(user_set):
    global user_source
    
    def zero_source_terms(U,aux,theta=None):
        return jnp.zeros_like(U)

    if user_set is None:
        user_source = zero_source_terms
    
    else:    
        if user_set['self_defined_source_terms'] is not None:
            user_source  = user_set['self_defined_source_terms']
        else:
            user_source = zero_source_terms
        
def update_aux(U,aux):
    rho = U[0:1,:,:]
    u = U[1:2,:,:]/rho
    v = U[2:3,:,:]/rho
    e = U[3:4,:,:]/rho - 0.5*(u**2+v**2)
    Y = U[4:,:,:]/rho
    initial_T = aux[1:2]
    aux_new = thermo.get_T(e,Y,initial_T)
    return aux.at[0:2].set(aux_new)

def source_terms(U,aux,theta=None):
    return user_source(U,aux,theta)

def aux_to_thermo(U,aux):
    gamma = aux[0:1]
    T = aux[1:2]
    return gamma,T

def U_to_prim(U,aux):
    state = U
    gamma,T = aux_to_thermo(U,aux)
    rho = state[0:1,:,:]
    u = state[1:2,:,:]/rho
    v = state[2:3,:,:]/rho
    Y = state[4:,:,:]/rho
    R = thermo.get_R(Y)#.astype(jnp.float32)
    p = (rho*R*T)
    #rhoe = state[3:4,:,:]-0.5*rho*(u**2+v**2)
    #gamma = p/rhoe + 1
    a = jnp.sqrt(gamma*p/rho)
    return rho,u,v,Y,p,a

def prim_to_U(q):
    rho,u,v,p,Y = q[0:1],q[1:2],q[2:3],q[3:4],q[4:]
    R = thermo.get_R(Y)
    T = p/(rho*R)
    _, gamma, h, _, _ = thermo.get_thermo(T,Y)
    rhoE = rho*h-p + 0.5*rho(u**2+v**2)
    U = jnp.concatenate([rho,rho*u,rho*v,rhoE,rho*Y],axis=0)
    aux = jnp.concatenate([gamma,T],axis=0)
    return U,aux
    




