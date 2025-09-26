import jax.numpy as jnp

def d_dx(f, dx):
    fm3 = f[:,0:-6]
    fm2 = f[:,1:-5]
    fm1 = f[:,2:-4]
    fp1 = f[:,4:-2]
    fp2 = f[:,5:-1]
    fp3 = f[:,6:]
    
    center = (-fm3 + 9*fm2
            -45*fm1 + 45*fp1
            - 9*fp2 + fp3) / (60*dx)
    left3 = (f[:,0:1]-8*f[:,1:2]+8*f[:,3:4]-f[:,4:5])/(12*dx)
    left2 = (f[:,2:3] - f[:,0:1])/(2*dx)
    left1 = (f[:,1:2] - f[:,0:1])/dx
    
    right1 = (f[:,-5:-4]-8*f[:,-4:-3]+8*f[:,-2:-1]-f[:,-1:])/(12*dx)
    right2 = (f[:,-1:] - f[:,-3:-2])/(2*dx)
    right3 = (f[:,-1:] - f[:,-2:-1])/dx
    
    #padded_center = boundary_padding.exchange_halo(center)    
    return jnp.concatenate([left1,left2,left3,center,right1,right2,right3],axis=1)#padded_center


