import jax
import jax.numpy as jnp

##parallel settings##
devices = jax.devices()
num_devices = len(devices)

def split_and_distribute_grid(grid):
    nu,nx,ny = grid.shape
    assert nx % num_devices == 0, "nx should be divisible by number of avaliable devices"
    shards = jnp.split(grid,num_devices,axis=1)
    sharded_grid = jax.device_put_sharded(shards, devices)
    return sharded_grid
