import jax
import jax.numpy as jnp

##parallel settings##
devices = jax.devices()
num_devices = len(devices)

def split_and_distribute_grid(grid,split_axis=1):
    nx = jnp.size(grid,axis=split_axis)
    assert nx % num_devices == 0, "nx should be divisible by number of avaliable devices"
    shards = jnp.split(grid,num_devices,axis=split_axis)
    sharded_grid = jax.device_put_sharded(shards, devices)
    return sharded_grid

def gather_grid(sharded_grid,gather_axis=1):
    return jnp.concatenate(sharded_grid,axis=gather_axis)

def split_and_distribute_block(blk_data,split_axis=0):
    n_blk = jnp.size(blk_data,axis=split_axis)
    assert n_blk % num_devices == 0, "number of blocks should be divisible by number of avaliable devices"
    shards = jnp.split(blk_data,num_devices,axis=split_axis)
    sharded_block = jax.device_put_sharded(shards, devices)
    return sharded_block

def gather_block(sharded_block,gather_axis=0):
    return jnp.concatenate(sharded_block,axis=gather_axis)

def split_and_distribute_blk_info(blk_info):
    number = blk_info['number']
    index = blk_info['index']
    glob_index = blk_info['glob_index']
    neighbor_index = blk_info['neighbor_index']

    number = [number for i in range(num_devices)]
    index = [index for i in range(num_devices)]
    glob_index = [glob_index for i in range(num_devices)]

    #add device id to neighbor_index
    num_blks = neighbor_index.shape(0)
    num_blks_per_device = num_blks // num_devices

    local_neighbor_index = [(neighbor_index % num_blks_per_device) for i in range(num_devices)]
    neighbor_device_index = [(neighbor_index // num_blks_per_device, i) for i in range(num_devices)]

    sharded_number = jax.device_put_sharded(number, devices)
    sharded_index = jax.device_put_sharded(index, devices)
    sharded_glob_index = jax.device_put_sharded(glob_index, devices)
    sharded_neighbor_index = jax.device_put_sharded(local_neighbor_index, devices)
    sharded_neighbor_device_index = jax.device_put_sharded(neighbor_device_index, devices)
    
    sharded_blk_info = {'number': sharded_number,
                        'index': sharded_index,
                        'glob_index': sharded_glob_index,
                        'neighbor_index': sharded_neighbor_index,
                        'neighbor_device_index':sharded_neighbor_device_index}
    return sharded_blk_info
    
def gather_blk_info(sharded_blk_info):
    number = sharded_blk_info['number']
    index = sharded_blk_info['index']
    glob_index = sharded_blk_info['glob_index']
    local_neighbor_index = sharded_blk_info['neighbor_index']
    neighbor_device_index = sharded_blk_info['neighbor_device_index']

    local_num_blks = local_neighbor_index.shape(0)
    
    number = jnp.concatenate(number,axis=0)
    index = jnp.concatenate(index,axis=0)
    glob_index = jnp.concatenate(glob_index,axis=0)
    neighbor_index = jnp.concatenate(neighbor_device_index*local_num_blks + local_neighbor_index,axis=0)

    blk_info = {'number': number,
                'index': index,
                'glob_index': glob_index,
                'neighbor_index': neighbor_index}
    return  blk_info


