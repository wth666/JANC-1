import jax.numpy as jnp


def min_mod_limiter(x,y):
    mask1 = jnp.minimum(x*jnp.sign(y),y*jnp.sign(x))
    mask = jnp.maximum(jnp.zeros_like(mask1),mask1)
    return jnp.sign(x)*mask

def interface_L_x(q,k=1/3,beta=1.0):
    U_x = q[:,1:-1,3:-3]
    d_minus_x = U_x[:,1:-1,:]-U_x[:,:-2,:]
    d_plus_x = U_x[:,2:,:]-U_x[:,1:-1,:]
        
    phi_i_x_fw = min_mod_limiter(d_plus_x,beta*d_minus_x)
    phi_i_x_bw = min_mod_limiter(d_minus_x,beta*d_plus_x)

    s_i_L_x = 1/4*((1-k)*phi_i_x_bw + (1+k)*phi_i_x_fw)

    q_L_x = U_x[:,1:-2,:] + s_i_L_x[:,:-1,:]
    
    return q_L_x

def interface_L_y(q,k=1/3,beta=1.0):
    
    U_y = q[:,3:-3,1:-1]
    d_minus_y = U_y[:,:,1:-1]-U_y[:,:,:-2]
    d_plus_y = U_y[:,:,2:]-U_y[:,:,1:-1]
    
    phi_i_y_fw = min_mod_limiter(d_plus_y,beta*d_minus_y)
    phi_i_y_bw = min_mod_limiter(d_minus_y,beta*d_plus_y)

    s_i_L_y = 1/4*((1-k)*phi_i_y_bw + (1+k)*phi_i_y_fw)

    q_L_y = U_y[:,:,1:-2] + s_i_L_y[:,:,:-1]
    
    return q_L_y

def interface_R_x(q,k=1/3,beta=1.0):
    U_x = q[:,1:-1,3:-3]
    d_minus_x = U_x[:,1:-1,:]-U_x[:,:-2,:]
    d_plus_x = U_x[:,2:,:]-U_x[:,1:-1,:]
        
    phi_i_x_fw = min_mod_limiter(d_plus_x,beta*d_minus_x)
    phi_i_x_bw = min_mod_limiter(d_minus_x,beta*d_plus_x)

    s_i_R_x = 1/4*((1-k)*phi_i_x_fw + (1+k)*phi_i_x_bw)
    
    q_R_x = U_x[:,2:-1,:] - s_i_R_x[:,1:,:]
    
    return q_R_x

def interface_R_y(q,k=1/3,beta=1.0):
    U_y = q[:,3:-3,1:-1]
    d_minus_y = U_y[:,:,1:-1]-U_y[:,:,:-2]
    d_plus_y = U_y[:,:,2:]-U_y[:,:,1:-1]
    
    phi_i_y_fw = min_mod_limiter(d_plus_y,beta*d_minus_y)
    phi_i_y_bw = min_mod_limiter(d_minus_y,beta*d_plus_y)

    s_i_R_y = 1/4*((1-k)*phi_i_y_fw + (1+k)*phi_i_y_bw)
    
    q_R_y = U_y[:,:,2:-1] - s_i_R_y[:,:,1:]
    
    return q_R_y

