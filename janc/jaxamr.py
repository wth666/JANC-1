import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.signal import convolve2d
from functools import partial
from janc.config import n_block, n_grid, refinement_tolerance, template_node_num, grid_mask_buffer_kernel


@partial(jit, static_argnames=('level', 'criterion'))
def get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy):
    '''
    功能：
        获取需加密网格掩码
    输入：
        level：当前加密层级
        blk_data, blk_info：基础层block数据和信息
        criterion：加密准则
        dx, dy：加密后的网格尺寸
    输出：
        ref_grid_mask：需加密网格掩码
    备注：
        加密准则为：blk_data在x,y任意方向梯度大于加密阈值
    '''

    num = template_node_num

    if level == 0:
        '''待补充'''
        pass
    elif level == 1:
        '''因为当前没有基础层并行管理，这里暂时这么处理'''
        if criterion == 'density':
            data_component = blk_data[:, 0]
        elif criterion == 'schlieren':
            '''待补充'''
            pass
        elif criterion == 'velocity':
            data_component = blk_data[:, 1]

        grad_x, grad_y = vmap(jnp.gradient, in_axes=0)(data_component)
    else:
        ghost_blk_data = get_ghost_block_data(blk_data, blk_info)

        if criterion == 'density':
            data_component = ghost_blk_data[:, 0]
        elif criterion == 'schlieren':
            '''待补充'''
            pass
        elif criterion == 'velocity':
            data_component = ghost_blk_data[:, 1]

        grad_x, grad_y = vmap(jnp.gradient, in_axes=0)(data_component)

        # 外边界被标记为NaN，此处实际梯度应为0，进行修正
        grad_x = jnp.nan_to_num(grad_x[:, num:-num, num:-num])
        grad_y = jnp.nan_to_num(grad_y[:, num:-num, num:-num])

    mask_x = jnp.maximum(jnp.abs(grad_x / (dx*2.0)) - refinement_tolerance[criterion], 0) # 这里dx*2转化为基础层网格尺寸
    mask_y = jnp.maximum(jnp.abs(grad_y / (dy*2.0)) - refinement_tolerance[criterion], 0)

    mask = jnp.sign(mask_x + mask_y) # 加密点


    def extension_mask(mask):

        extended_mask = jnp.sign(convolve2d(mask, grid_mask_buffer_kernel, mode='same')) # 扩展加密点

        return extended_mask

    ref_grid_mask = vmap(extension_mask, in_axes=0)(mask)

    return ref_grid_mask



@partial(jit, static_argnames=('level'))
def get_refinement_block_mask(level, ref_grid_mask):
    '''
    功能：
        获取需加密block掩码
    输入：
        level：当前加密层级
        ref_grid_mask：需加密网格掩码
    输出：
        ref_blk_mask：需加密block位置掩码
    备注：
        加密准则为：block中存在需加密网格（可进行智能识别）
    '''
    ref_grid_mask = ref_grid_mask.reshape(ref_grid_mask.shape[0],
                        n_block[level][0], n_grid[level][0],
                        n_block[level][1], n_grid[level][1]).transpose(0, 1, 3, 2, 4) # 重塑矩阵以实现分块, 调整维度顺序以便于求和

    ref_blk_mask = jnp.sign(ref_grid_mask.sum(axis=(3, 4))) # 计算每个块中元素的和

    return ref_blk_mask




@partial(jit, static_argnames=('max_blk_num'))
def get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num):
    '''
    功能：
        获取加密block的坐标、数量和邻居
    输入：
        level：当前加密层级
        blk_info：基础层block信息
        ref_blk_mask：需加密block位置掩码
    输出：
        ref_blk_info：加密block信息
            number：加密block有效数量
            index：加密block在基础层中的坐标
            abs_index：加密block在全局的绝对坐标
            neighbor：加密block邻居序号
    备注：
        为兼容jax.jit，生成的block总数为max_block_number，
        其中[0:block_number]为有实际值的block，
        [block_number:]为冗余block，其block_index填充为[-1, -1]，block_neighbor_index填充为[0, 0, 0, 0]，
        特别地，最后一个block[-1]被赋值为NaN矩阵
    '''
    mask = ref_blk_mask != 0
    flat_mask = mask.ravel() # 生成行优先的编号矩阵
    flat_indices = jnp.cumsum(flat_mask) * flat_mask
    indices_matrix = flat_indices.reshape(ref_blk_mask.shape)

    indices_matrix = get_ghost_mask(blk_info, indices_matrix)

    # 位移逻辑：通过pad和切片实现正确位移
    up = jnp.pad(indices_matrix, ((0, 0), (1, 0), (0, 0)), mode="constant")[:, 1:-2, 1:-1] # 上方向：pad顶部一行，取前n-1行
    down = jnp.pad(indices_matrix, ((0, 0), (0, 1), (0, 0)), mode="constant")[:, 2:-1, 1:-1] # 下方向：pad底部一行，取后n-1行
    left = jnp.pad(indices_matrix, ((0, 0), (0, 0), (1, 0)), mode="constant")[:, 1:-1, 1:-2] # 左方向：pad左侧一列，取前m-1列
    right = jnp.pad(indices_matrix, ((0, 0), (0, 0), (0, 1)), mode="constant")[:, 1:-1, 2:-1] # 右方向：pad右侧一列，取后m-1列


    # 获取非零元素的坐标
    blks, rows, cols = jnp.nonzero(mask, size = max_blk_num, fill_value = -1) # 冗余block的坐标被填充为[-1,-1]

    # 提取邻居编号（越界处自动为-1）
    up_vals = up[blks, rows, cols] - 1
    down_vals = down[blks, rows, cols] - 1
    left_vals = left[blks, rows, cols] - 1
    right_vals = right[blks, rows, cols] - 1

    ref_glob_blk_index = jnp.column_stack([blk_info['glob_index'][blks], rows, cols])
    ref_blk_index = jnp.column_stack([blks, rows, cols])
    ref_blk_number = jnp.sum(jnp.sign(ref_blk_mask))
    ref_blk_neighbor = jnp.column_stack([up_vals, down_vals, left_vals, right_vals])

    # 生成行索引并创建掩码
    row_indices = jnp.arange(ref_blk_neighbor.shape[0])
    mask_nonzero = row_indices < ref_blk_number
    # 扩展掩码维度以匹配二维数组结构
    mask_nonzero = mask_nonzero[:, jnp.newaxis]

    # 使用掩码将后续行置零
    ref_blk_neighbor = jnp.where(mask_nonzero, ref_blk_neighbor, -1)

    ref_blk_info = {
        'number': ref_blk_number.astype(int),
        'index': ref_blk_index,
        'glob_index': ref_glob_blk_index,
        'neighbor_index': ref_blk_neighbor
    }

    return ref_blk_info # Pytree



@partial(jit, static_argnames=('level'))
def get_refinement_block_data(level, blk_data, ref_blk_info):
    '''
    功能：
        获取待加密的block数据
    输入：
        level：当前加密层级
        blk_data：基础层数据
        ref_blk_info：加密block信息
    输出：
        ref_blk_data：加密的block数据
    '''
    blk_data = blk_data.reshape(blk_data.shape[0], blk_data.shape[1],
                n_block[level][0], n_grid[level][0],
                n_block[level][1], n_grid[level][1]).transpose(0, 1, 2, 4, 3, 5)

    blks = ref_blk_info['index'][:, 0]
    rows = ref_blk_info['index'][:, 1]
    cols = ref_blk_info['index'][:, 2]
    ref_blk_data = blk_data[blks, :, rows, cols, :, :] # 第一维度为block集合

    '''最后一个block会被标记为NaN，因此要求预设的block数量比有效block多1'''
    '''主要目的是在进行block边界扩充时，将无效值（即无邻居矩阵的边界）标记为NaN'''
    ref_blk_data = ref_blk_data.at[-1].set(jnp.nan)

    ref_blk_data = interpolate_coarse_to_fine(ref_blk_data) # 插值完成加密

    return ref_blk_data



@jit
def interpolate_coarse_to_fine(ref_blk_data):
    '''
    功能：
        插值获取加密后的block数据
    输入：
        ref_blk_data：尚未加密的block数据
    输出：
        ref_blk_data：加密后的block数据
    备注：
        插值方式为：一分四
    '''
    kernel = jnp.ones((2, 2))

    ref_blk_data = jnp.kron(ref_blk_data, kernel) # block_data前两维是广播维度

    return ref_blk_data



@partial(jit, static_argnames=('level'))
def interpolate_fine_to_coarse(level, blk_data, ref_blk_data, ref_blk_info):
    '''
    功能：
        将加密层数据插值到基础层
    输入：
        level：当前加密层级
        blk_data：基础层数据
        ref_blk_data：加密block数据
        ref_blk_info：加密block信息
    输出：
        updated_blk_data：更新后的基础层数据
    备注：
        插值方式为：四合一
    '''
    updated_blk_data = blk_data

    ref_blk_data = ref_blk_data.reshape(ref_blk_data.shape[0], ref_blk_data.shape[1],
                        ref_blk_data.shape[2]//2, 2,
                        ref_blk_data.shape[3]//2, 2).mean(axis=(3, 5))


    updated_blk_data = updated_blk_data.reshape(updated_blk_data.shape[0], updated_blk_data.shape[1],
                    n_block[level][0], n_grid[level][0],
                    n_block[level][1], n_grid[level][1]).transpose(0, 1, 2, 4, 3, 5)

    blks = ref_blk_info['index'][:, 0]
    rows = ref_blk_info['index'][:, 1]
    cols = ref_blk_info['index'][:, 2]
    updated_blk_data = updated_blk_data.at[blks, :, rows, cols, :, :].set(ref_blk_data)

    '''这里将U[-1,-1]赋原值，是因为多余block的index都被标记为[-1,-1]'''
    updated_blk_data = (
                updated_blk_data.at[:, :, -1, -1, :, :]
                .set(blk_data[:, :, -n_grid[level][0]:, -n_grid[level][1]:])
                .transpose(0, 1, 2, 4, 3, 5)
                .reshape(updated_blk_data.shape[0], updated_blk_data.shape[1],
                    n_block[level][0] * n_grid[level][0],
                    n_block[level][1] * n_grid[level][1])
    )

    return updated_blk_data



'''Morton Index计算可能有问题'''
@jit
def compute_morton_index(coords):
    coords = jnp.asarray(coords, dtype=jnp.uint32) & 0xFFFF  # 确保输入为16位无符号整数
    d = coords.shape[0]  # 获取输入的维度数

    # 动态扩展每个坐标的二进制位，使得每个位之间相隔d-1个零
    shift = 8
    while shift >= 1:
        mask = 0
        # 生成掩码，每隔shift*d位设置一个shift位的块
        for i in range(0, 32, shift * d):
            mask |= ((1 << shift) - 1) << i
        # 将高位块左移并合并，中间插入shift*(d-1)位的零
        coords = (coords | (coords << (shift * (d - 1)))) & mask
        shift = shift // 2  # 处理更小的块

    # 合并各维度的位，生成Morton码
    shifts = jnp.arange(d, dtype=jnp.uint32)
    index = jnp.bitwise_or.reduce(coords << shifts[:, None], axis=0)
    return index.astype(jnp.uint32)



@jit
def compare_coords(A, B):

    matches = (A[:, None, :] == B[None, :, :])
    full_match = matches.all(axis=-1)

    return full_match.any(axis=1)


@jit
def find_unaltered_block_index(blk_info, prev_blk_info):
    '''
    功能：
        获取AMR更新后仍然保留的block（即不更新的block）
    输入：
        blk_info, prev_blk_info：更新前后的block信息
    输出：
        rows_A：保留的block在原block中的序号
        rows_B：保留的block在新block中的序号
        unaltered_num：序号中的有效位数（多余位填充-1）
    '''
    index_A, num_A = prev_blk_info['glob_index'], prev_blk_info['number']
    index_B, num_B = blk_info['glob_index'], blk_info['number']

    '''
    # 通过计算Morton index来加速匹配计算（暂未启用）
    morton_A = compute_morton_index(index_A.transpose(1,0))
    morton_B = compute_morton_index(index_B.transpose(1,0))

    mask_A = jnp.isin(morton_A, morton_B)
    mask_B = jnp.isin(morton_B, morton_A)

    '''
    # 处理index_A和index_B的匹配
    mask_A = compare_coords(index_A, index_B)
    mask_B = compare_coords(index_B, index_A)

    rows_A = jnp.nonzero(mask_A, size=index_A.shape[0], fill_value=-1)[0]
    rows_B = jnp.nonzero(mask_B, size=index_B.shape[0], fill_value=-1)[0]

    '''这里需要注意当扩充max_block_num时，morton_A.shape[0]不等于morton_B.shape[0]的情况'''
    '''正常来说，由A计算的unaltered_num应该和由B计算的一致'''
    unaltered_num = jnp.sum(jnp.sign(rows_A+1)) + num_A - index_A.shape[0]

    return rows_A, rows_B, unaltered_num



@jit
def get_ghost_mask(blk_info, mask):
    '''
        mask形状：[blks, rows, cols]
        mask为上一层blk加密掩码
        【形状与level无关】
    '''

    num = 1
    neighbor = blk_info['neighbor_index']

    # 边界数据（b, h, w = data.shape）
    upper = mask[neighbor[:,0], -num:, :] # 上边界(b, num, w)
    lower = mask[neighbor[:,1], :num, :] # 下边界(b, num, w)
    left = mask[neighbor[:,2], :, -num:] # 左边界(b, h, num)
    right = mask[neighbor[:,3], :, :num] # 右边界(b, h, num)

    # 水平扩展（左右边界）
    padded_horizontal = jnp.concatenate([left, mask, right], axis=2)  # (b, h, w+2*num)

    # 垂直扩展（上下边界），对上下边界做水平填充保证宽度一致
    '''注意这里pad填充值是0'''
    pad_upper = jnp.pad(upper, ((0,0), (0,0), (num,num)), mode='constant', constant_values=0)  # (b, num, w+2*num)
    pad_lower = jnp.pad(lower, ((0,0), (0,0), (num,num)), mode='constant', constant_values=0)  # (b, num, w+2*num)

    # 垂直拼接
    ghost_mask = jnp.concatenate([pad_upper, padded_horizontal, pad_lower], axis=1)  # (b, h+2*num, w+2*num)

    return ghost_mask


@jit
def get_ghost_block_data(blk_data, blk_info):
    '''
        【形状与level无关】
    '''
    num = template_node_num

    neighbor = blk_info['neighbor_index']

    # 边界数据（b, n, h, w = data.shape），基础层边界(-1)被标记为NaN
    upper = blk_data[neighbor[:,0], :, -num:, :] # 上边界(b, n, num, w)
    lower = blk_data[neighbor[:,1], :, :num, :] # 下边界(b, n, num, w)
    left = blk_data[neighbor[:,2], :, :, -num:] # 左边界(b, n, h, num)
    right = blk_data[neighbor[:,3], :, :, :num] # 右边界(b, n, h, num)

    # 水平扩展（左右边界）
    padded_horizontal = jnp.concatenate([left, blk_data, right], axis=3)  # (b, n, h, w+2*num)

    # 垂直扩展（上下边界），对上下边界做水平填充保证宽度一致
    '''注意这里pad填充值是NaN'''
    pad_upper = jnp.pad(upper, ((0,0), (0,0), (0,0), (num,num)), mode='constant', constant_values=jnp.nan)  # (b, n, num, w+2*num)
    pad_lower = jnp.pad(lower, ((0,0), (0,0), (0,0), (num,num)), mode='constant', constant_values=jnp.nan)  # (b, n, num, w+2*num)

    # 垂直拼接
    ghost_blk_data = jnp.concatenate([pad_upper, padded_horizontal, pad_lower], axis=2)  # (b, n, h+2*num, w+2*num)

    return ghost_blk_data



@partial(jit, static_argnames=('level'))
def update_external_boundary(level, blk_data, ref_blk_data, ref_blk_info):
    '''
    功能：
        获取加密block的外边界值（与基础层网格的边界）
    输入：
        level：当前加密层级
        blk_data：基础层数据
        ref_blk_data：加密block数据
        ref_blk_info：加密block信息
    输出：
        ref_blk_data：更新外边界后的加密block数据
    '''
    num = template_node_num

    raw_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    # 将neighbor_index中的0转成1，非0转成0
    neighbor = jnp.sign(ref_blk_info['neighbor_index'] + 1)[:, :, None, None, None]
    boundary_mask = jnp.ones_like(neighbor) - neighbor

    # get_ghost_block_data时，加密层边界处的neighbor_index为-1，此边界被标记为了NaN
    ref_blk_data = jnp.nan_to_num(ref_blk_data)

    value = ref_blk_data[..., :num, :] * neighbor[:,0] \
        + raw_blk_data[..., :num, :] * boundary_mask[:,0] # 上边界(b, n, num, w)
    ref_blk_data = ref_blk_data.at[..., :num, :].set(value)

    value = ref_blk_data[..., -num:, :] * neighbor[:,1] \
        + raw_blk_data[..., -num:, :] * boundary_mask[:,1] # 下边界(b, n, num, w)
    ref_blk_data = ref_blk_data.at[..., -num:, :].set(value)

    value = ref_blk_data[..., :, :num] * neighbor[:,2] \
        + raw_blk_data[..., :, :num] * boundary_mask[:,2] # 左边界(b, n, h, num)
    ref_blk_data = ref_blk_data.at[..., :, :num].set(value)

    value = ref_blk_data[..., :, -num:] * neighbor[:,3] \
        + raw_blk_data[..., :, -num:] * boundary_mask[:,3] # 右边界(b, n, h, num)
    ref_blk_data = ref_blk_data.at[..., :, -num:].set(value)

    '''最后一个block会被标记为NaN，因此要求预设的block数量比有效block多1'''
    ref_blk_data = ref_blk_data.at[-1].set(jnp.nan)

    return ref_blk_data



def initialize(level, blk_data, blk_info, criterion, dx, dy):
    '''
    功能：
        初始化AMR
    输入：
        level：当前加密层级
        blk_data：基础层block数据
        blk_info：基础层block信息
        criterion：加密准则
        dx, dy：加密后的网格尺寸
    输出：
        ref_blk_data：加密层block数据
        ref_blk_info：加密层block信息
    '''

    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy)

    ref_blk_mask = get_refinement_block_mask(level, ref_grid_mask)

    max_blk_num = initialize_max_block_number(level, ref_blk_mask)

    ref_blk_info = get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num)

    ref_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    print(f'\nAMR Initialized at Level [{level}] with [{max_blk_num}] blocks')

    return ref_blk_data, ref_blk_info, max_blk_num



def update(level, blk_data, blk_info, criterion, dx, dy, prev_ref_blk_data, prev_ref_blk_info, max_blk_num):
    '''
    功能：
        更新AMR
    输入：
        level：当前加密层级
        blk_data：基础层block数据
        blk_info：基础层block信息
        criterion：加密准则
        dx, dy：加密后的网格尺寸
        prev_ref_blk_data：更新前加密层block数据
        prev_ref_blk_info：更新前加密层block信息
    输出：
        ref_blk_data：更新后的加密层block数据
        ref_blk_info：更新后的加密层block信息
    '''

    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy)

    ref_blk_mask = get_refinement_block_mask(level, ref_grid_mask)

    updated_mask, updated_max_blk_num = update_max_block_number(ref_blk_mask, max_blk_num)
    if updated_mask:
        max_blk_num = updated_max_blk_num
        print('\nAMR max_blk_num Updated as[',max_blk_num,'] at Level [',level,']')

    ref_blk_info = get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num)

    ref_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    rows_A, rows_B, unaltered_num = find_unaltered_block_index(ref_blk_info, prev_ref_blk_info)
    ref_blk_data = ref_blk_data.at[rows_B[0:unaltered_num]].set(prev_ref_blk_data[rows_A[0:unaltered_num]])

    valid_blk_num = ref_blk_info['number']
    print(f'\nAMR Updated at Level [{level}] with [{valid_blk_num}/{max_blk_num}] blocks [valid/max]')

    return ref_blk_data, ref_blk_info, max_blk_num



def initialize_max_block_number(level, ref_blk_mask):

    ref_blk_num = jnp.sum(jnp.sign(ref_blk_mask))

    max_blk_num = int((ref_blk_num + 50 )//50 * 50)#int((ref_blk_num + 10 * 2**(level-1) )//10 * 10)

    return max_blk_num



def update_max_block_number(ref_blk_mask, max_blk_num):

    updated_mask = False
    updated_max_blk_num = max_blk_num
    ref_blk_num = jnp.sum(jnp.sign(ref_blk_mask))

    if (ref_blk_num + 1) > max_blk_num:
        updated_mask = True
        updated_max_blk_num = int(max_blk_num * 2.0)
    #elif (ref_blk_num + 1) < (max_blk_num/2.5):
        #updated_mask = True
        #updated_max_blk_num = int(max_blk_num / 2.0)
    #else:
        #updated_mask = False
        #updated_max_blk_num = max_blk_num

    return updated_mask, updated_max_blk_num


