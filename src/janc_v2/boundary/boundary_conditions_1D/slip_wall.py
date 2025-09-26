import jax.numpy as jnp

def left(U_bd, aux_bd, theta=None):
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1],U_bd[:,0:1],U_bd[:,0:1]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,0:1],aux_bd[:,0:1],aux_bd[:,0:1]],axis=1)
    rhou = U_bd_ghost[1:2]
    rhou_new = -rhou
    U_bd_ghost = U_bd_ghost.at[1:2].set(rhou_new)
    return U_bd_ghost, aux_bd_ghost

def right(U_bd, aux_bd, theta=None):
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:],U_bd[:,-1:],U_bd[:,-1:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,-1:],aux_bd[:,-1:],aux_bd[:,-1:]],axis=1)
    rhou = U_bd_ghost[1:2]
    rhou_new = -rhou
    U_bd_ghost = U_bd_ghost.at[1:2].set(rhou_new)

    return U_bd_ghost, aux_bd_ghost
