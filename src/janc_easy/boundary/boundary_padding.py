import jax
import jax.numpy as jnp

def pad(U):
    field = U
    field_periodic_x = jnp.concatenate([field[:,-4:-3,:],field[:,-3:-2,:],field[:,-2:-1,:],field,field[:,1:2,:],field[:,2:3,:],field[:,3:4,:]],axis=1)
    field_periodic_pad = jnp.concatenate([field_periodic_x[:,:,-4:-3],field_periodic_x[:,:,-3:-2],field_periodic_x[:,:,-2:-1],field_periodic_x,field_periodic_x[:,:,1:2],field_periodic_x[:,:,2:3],field_periodic_x[:,:,3:4]],axis=2)
    return field_periodic_pad

def replace_lb(U_bd, padded_U):
    U = padded_U.at[:,0:3,3:-3].set(U_bd)
    return U

    
def replace_rb(U_bd, padded_U):
    U = padded_U.at[:,-3:,3:-3].set(U_bd)
    return U


def replace_ub(U_bd, padded_U):
    U = padded_U.at[:,3:-3,-3:].set(U_bd)
    return U



def replace_bb(U_bd, padded_U):  
    U = padded_U.at[:,3:-3,0:3].set(U_bd)
    return U



##parallel settings##
num_devices = jax.local_device_count()
devices = jax.devices()


def exchange_halo(device_grid):
    _, grid_nx, _ = device_grid.shape
    halo_size = 3

    send_right = device_grid[:,-halo_size:,:] #向右发送的数据
    recv_left = jax.lax.ppermute(send_right,'x',[(i,(i+1)%num_devices) for i in range(num_devices)])

    send_left = device_grid[:,:halo_size,:] #向左发送的数据
    recv_right = jax.lax.ppermute(send_left,'x',[(i,(i-1)%num_devices) for i in range(num_devices)])

    new_grid = jnp.concatenate([recv_left,device_grid,recv_right],axis=1)
    return new_grid


def parallel_pad(U):
    field = U
    field_periodic_x = exchange_halo(field)
    field_periodic_pad = jnp.concatenate([field_periodic_x[:,:,-4:-3],field_periodic_x[:,:,-3:-2],field_periodic_x[:,:,-2:-1],field_periodic_x,field_periodic_x[:,:,1:2],field_periodic_x[:,:,2:3],field_periodic_x[:,:,3:4]],axis=2)
    return field_periodic_pad
    

