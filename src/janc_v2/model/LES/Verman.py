import jax.numpy as jnp

c = 0.07

def mu_t_2D(rho,dx,dy,dz,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz):
    d1,d2 = dx,dy
    a11,a12 = dudx,dvdx
    a21,a22 = dudy,dvdy
    b11_x,b12_x = (d1**2)*a11*a11,(d1**2)*a11*a12
    b21_x,b22_x = (d1**2)*a12*a11,(d1**2)*a12*a12
    
    b11_y,b12_y = (d2**2)*a21*a21,(d2**2)*a21*a22
    b21_y,b22_y = (d2**2)*a22*a21,(d2**2)*a22*a22
 
    b11,b12 = b11_x+b11_y,b12_x+b12_y
    b21,b22 = b21_x+b21_y,b22_x+b22_y
    B_b = jnp.abs(b11*b22-b12**2)
    mask = (B_b>=0.0)
    B_b = jnp.where(mask,B_b,jnp.zeros_like(B_b))
    a_sum = a11**2+a12**2+a21**2+a22**2
    mask = a_sum>0.0
    ratio = jnp.where(mask,B_b/a_sum,jnp.zeros_like(a_sum))
    mu_t = rho*c*jnp.sqrt(ratio)
    return mu_t


    
    
