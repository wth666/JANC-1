import jax
import jax.numpy as jnp

def pad_1D(U,aux):
    field = jnp.concatenate([U,aux],axis=0)
    field_periodic_x = jnp.concatenate([field[:,-4:-3],field[:,-3:-2],field[:,-2:-1],field,field[:,1:2],field[:,2:3],field[:,3:4]],axis=1)
    U_periodic_pad,aux_periodic_pad = field_periodic_x[0:-2],field_periodic_x[-2:]
    return U_periodic_pad,aux_periodic_pad

def replace_lb_1D(U_bd, aux_bd, padded_U, padded_aux):
    U = padded_U.at[:,0:3].set(U_bd)
    aux = padded_aux.at[:,0:3].set(aux_bd)
    return U, aux

def replace_rb_1D(U_bd, aux_bd, padded_U, padded_aux):
    U = padded_U.at[:,-3:].set(U_bd)
    aux = padded_aux.at[:,-3:].set(aux_bd)
    return U, aux

def pad_2D(U,aux):
    field = jnp.concatenate([U,aux],axis=0)
    field_periodic_x = jnp.concatenate([field[:,-4:-3,:],field[:,-3:-2,:],field[:,-2:-1,:],field,field[:,1:2,:],field[:,2:3,:],field[:,3:4,:]],axis=1)
    field_periodic_pad = jnp.concatenate([field_periodic_x[:,:,-4:-3],field_periodic_x[:,:,-3:-2],field_periodic_x[:,:,-2:-1],field_periodic_x,field_periodic_x[:,:,1:2],field_periodic_x[:,:,2:3],field_periodic_x[:,:,3:4]],axis=2)
    U_periodic_pad,aux_periodic_pad = field_periodic_pad[0:-2],field_periodic_pad[-2:]
    return U_periodic_pad,aux_periodic_pad

def replace_lb_2D(U_bd, aux_bd, padded_U, padded_aux):
    U = padded_U.at[:,0:3,3:-3].set(U_bd)
    aux = padded_aux.at[:,0:3,3:-3].set(aux_bd)
    return U, aux
    
def replace_rb_2D(U_bd, aux_bd, padded_U, padded_aux):
    U = padded_U.at[:,-3:,3:-3].set(U_bd)
    aux = padded_aux.at[:,-3:,3:-3].set(aux_bd)
    return U, aux

def replace_ub_2D(U_bd, aux_bd, padded_U, padded_aux):
    U = padded_U.at[:,3:-3,-3:].set(U_bd)
    aux = padded_aux.at[:,3:-3,-3:].set(aux_bd)
    return U, aux

def replace_bb_2D(U_bd, aux_bd, padded_U, padded_aux):  
    U = padded_U.at[:,3:-3,0:3].set(U_bd)
    aux = padded_aux.at[:,3:-3,0:3].set(aux_bd)
    return U, aux
    


