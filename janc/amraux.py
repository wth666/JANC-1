import jax.numpy as jnp
from janc.config import Lx, Ly, n_block, n_grid

def plot_block_data(blk_data_component, blk_info, fig_handle, vrange):

    valid_blk_num = blk_info['number']
    level = blk_info['glob_index'].shape[1]//2 - 1
    for i in range(valid_blk_num):
        x_min = 0
        y_min = 0
        dx = Lx
        dy = Ly
        for j in range(level+1):
            dx = dx/n_block[j][0]
            dy = dy/n_block[j][1]
            idx = blk_info['glob_index'][i, 2*j]
            idy = blk_info['glob_index'][i, 2*j+1]
            x_min = x_min + dx * idx
            y_min = y_min + dy * idy
        x_max = x_min + dx
        y_max = y_min + dy

        if level == 0:
            nx = n_grid[level][0]
            ny = n_grid[level][1]
        else:
            nx = n_grid[level][0] * 2
            ny = n_grid[level][1] * 2

        # 生成网格边界的坐标
        x_edges = jnp.linspace(x_min, x_max, nx)
        y_edges = jnp.linspace(y_min, y_max, ny)

        # 创建网格
        X, Y = jnp.meshgrid(x_edges, y_edges)

        fig = fig_handle.pcolormesh(X, Y, blk_data_component[i].transpose(1,0), shading='auto', vmin=vrange[0], vmax=vrange[1])

    return fig


def get_N_level_block_data(level, blk_data, blk_info):
    '''
    功能：
        获取第N层AMR中的block数据
    输入：
        level：当前加密层级
        blk_data：block数据
        blk_info：block信息
    输出：
        U_restored：展开到二维流场中的block数据（block外的数据赋值为0）
    '''
    nx = n_grid[0][0] * 2**level
    ny = n_grid[0][1] * 2**level
    nU = blk_data.shape[1]
    U = jnp.zeros((nU, nx, ny))

    # 动态构建 Reshape 维度
    reshape_dims = [nU]
    for i in range(1, level + 1):
        reshape_dims.extend([n_block[i][0]])
    reshape_dims.append(n_grid[level][0] * 2)
    for i in range(1, level + 1):
        reshape_dims.extend([n_block[i][1]])
    reshape_dims.append(n_grid[level][1] * 2)

    # 执行 Reshape 和 Transpose
    U_reshaped = U.reshape(reshape_dims)
    transpose_order = [0]  # 物理量维度固定在最前
    for i in range(1, level + 2):
        transpose_order.append(i)  # 块行索引
        transpose_order.append(i + level + 1)  # 块行索引
    U_transposed = U_reshaped.transpose(transpose_order)

    # 动态提取所有层级的块索引
    index_columns = blk_info['glob_index'][:blk_info['number'], 2:2 + 2 * level]
    index_tuple = tuple(index_columns[:, i] for i in range(2 * level))

    # 调整 blk_data 维度并填充
    blk_data_processed = blk_data[:blk_info['number']].transpose(1, 0, 2, 3)

    U_updated = U_transposed.at[(slice(None),) + index_tuple].set(blk_data_processed)

    # 逆向转置和 Reshape 恢复全局网格
    def get_inverse_order(order):
        '''计算转置顺序的逆排列'''
        inv_order = [0] * len(order)
        for i, pos in enumerate(order):
            inv_order[pos] = i
        return inv_order
    inv_transpose_order = get_inverse_order(transpose_order)
    U_restored = U_updated.transpose(inv_transpose_order).reshape((4, nx, ny))

    return U_restored