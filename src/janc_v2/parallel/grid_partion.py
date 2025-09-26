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
