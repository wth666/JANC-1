import jax
import jax.numpy as jnp

##parallel settings##
devices = jax.devices()
num_devices = len(devices)

def split_and_distribute_grid(grid,split_axis=1):
    nx = jnp.size(grid,axis=split_axis)
    assert nx % num_devices == 0, "nx should be divisible by number of avaliable devices"
    shards = jnp.split(grid,num_devices,axis=1)
    sharded_grid = jax.device_put_sharded(shards, devices)
    return sharded_grid

def split_and_distribute_blk_info(blk_info):
    number = blk_info['number']
    index = blk_info['index']
    glob_index = blk_info['glob_index']
    neighbor_index = blk_info['neighbor_index']

    number = [number for i in range(num_devices)]
    index = [index for i in range(num_devices)]
    glob_index = [glob_index for i in range(num_devices)]
    neighbor_index = [neighbor_index for i in range(num_devices)]

    sharded_number = jax.device_put_sharded(number, devices)
    sharded_index = jax.device_put_sharded(index, devices)
    sharded_glob_index = jax.device_put_sharded(glob_index, devices)
    sharded_neighbor_index = jax.device_put_sharded(neighbor_index, devices)
    
    sharded_blk_info = {'number': sharded_number,
                        'index': sharded_index,
                        'glob_index': sharded_glob_index,
                        'neighbor_index': sharded_neighbor_index}
    return sharded_blk_info
