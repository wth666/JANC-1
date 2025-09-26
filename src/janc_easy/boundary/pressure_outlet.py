import jax
import jax.numpy as jnp
from ..thermodynamics import thermo

def pressure_outlet(state_out,normal_vel,Pb):
    rho_out = state_out[0:1,:,:]
    u_out = state_out[1:2,:,:]/rho_out
    v_out = state_out[2:3,:,:]/rho_out
    p_out = (thermo.gamma - 1)*(state_out[3:4]-0.5*rho_out*(u_out**2+v_out**2))
    a_out = jnp.sqrt(thermo.gamma*p_out/rho_out)
    mask = (normal_vel/a_out < 1)
    rho_cor_out = jax.lax.select(mask, Pb / (p_out / rho_out),rho_out)
    p_cor_out = jax.lax.select(mask, Pb*jnp.ones_like(p_out),p_out)
    rhoe_cor_out = p_cor_out/(thermo.gamma-1)
    U_bd = jnp.concatenate([rho_cor_out, rho_cor_out * u_out, rho_cor_out * v_out,
                      rhoe_cor_out + 0.5 * rho_cor_out * (u_out ** 2 + v_out ** 2)], axis=0)
    return U_bd

def left(U_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,0:1,:]
    normal_vel = -state_out[1:2,:,:]/state_out[0:1,:,:]
    U_bd = pressure_outlet(state_out, normal_vel, Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    return U_bd_ghost

def right(U_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,-1:,:]
    normal_vel = state_out[1:2,:,:]/state_out[0:1,:,:]
    U_bd = pressure_outlet(state_out, normal_vel, Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    return U_bd_ghost

def bottom(U_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,:,0:1]
    normal_vel = -state_out[2:3,:,:]/state_out[0:1,:,:]
    U_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    return U_bd_ghost

def up(U_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,:,-1:]
    normal_vel = state_out[2:3,:,:]/state_out[0:1,:,:]
    U_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    return U_bd_ghost


