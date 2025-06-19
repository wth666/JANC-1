import jax
import jax.numpy as jnp
from .grid_partion import num_devices


def exchange_halo(device_grid):
    _, grid_nx, _ = device_grid.shape
    halo_size = 3

    send_right = device_grid[:,-halo_size:,:] #向右发送的数据
    recv_left = jax.lax.ppermute(send_right,'x',[(i,(i+1)%num_devices) for i in range(num_devices)])

    send_left = device_grid[:,:halo_size,:] #向左发送的数据
    recv_right = jax.lax.ppermute(send_left,'x',[(i,(i-1)%num_devices) for i in range(num_devices)])

    new_grid = jnp.concatenate([recv_left,device_grid,recv_right],axis=1)
    return new_grid


def pad(U,aux):
    field = jnp.concatenate([U,aux],axis=0)
    field_periodic_x = exchange_halo(field)
    field_periodic_pad = jnp.concatenate([field_periodic_x[:,:,-4:-3],field_periodic_x[:,:,-3:-2],field_periodic_x[:,:,-2:-1],field_periodic_x,field_periodic_x[:,:,1:2],field_periodic_x[:,:,2:3],field_periodic_x[:,:,3:4]],axis=2)
    U_periodic_pad,aux_periodic_pad = field_periodic_pad[0:-2],field_periodic_pad[-2:]
    return U_periodic_pad,aux_periodic_pad


@jit
def get_ghost_block_data(blk_data, blk_info):

    num = 3

    neighbor = blk_info['local_neighbor_index']
    device_sent_list = blk_info['device_index']
    num_local_blocks = neighbor.shape(0)
    
    perm = device_sent_list[(my_device, device[i]) for i in range(num_local_blocks)]
    

    
    upper = blk_data[neighbor[:,0], :, -num:, :]
    lower = blk_data[neighbor[:,1], :, :num, :]
    left = blk_data[neighbor[:,2], :, :, -num:]
    right = blk_data[neighbor[:,3], :, :, :num]

    padded_horizontal = jnp.concatenate([left, blk_data, right], axis=3)

    pad_upper = jnp.pad(upper, ((0,0), (0,0), (0,0), (num,num)), mode='constant', constant_values=jnp.nan) 
    pad_lower = jnp.pad(lower, ((0,0), (0,0), (0,0), (num,num)), mode='constant', constant_values=jnp.nan)

    ghost_blk_data = jnp.concatenate([pad_upper, padded_horizontal, pad_lower], axis=2)

    return ghost_blk_data
