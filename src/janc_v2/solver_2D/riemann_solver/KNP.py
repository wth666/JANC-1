import jax.numpy as jnp
from ...model import thermo_model as thermo


def riemann_flux(q_L_x, q_R_x, q_L_y, q_R_y):
    # 提取左右状态的基本物理量（x方向）
    rho_L_x, rho_R_x = q_L_x[0:1], q_R_x[0:1]
    p_L_x, p_R_x = q_L_x[3:4], q_R_x[3:4]
    Y_L_x, Y_R_x = q_L_x[4:], q_R_x[4:]
    
    # 计算热力学参数（x方向）
    R_L_x, R_R_x = thermo.get_R(Y_L_x), thermo.get_R(Y_R_x)
    T_L_x, T_R_x = p_L_x/(R_L_x*rho_L_x), p_R_x/(R_R_x*rho_R_x)
    _, gamma_L_x, h_L_x, _, _ = thermo.get_thermo(T_L_x, Y_L_x)
    _, gamma_R_x, h_R_x, _, _ = thermo.get_thermo(T_R_x, Y_R_x)
    
    # 计算x方向速度和能量（x方向）
    u_L_x, u_R_x = q_L_x[1:2], q_R_x[1:2]
    v_L_x, v_R_x = q_L_x[2:3], q_R_x[2:3]
    E_L_x = rho_L_x*h_L_x - p_L_x + 0.5*rho_L_x*(u_L_x**2 + v_L_x**2)
    E_R_x = rho_R_x*h_R_x - p_R_x + 0.5*rho_R_x*(u_R_x**2 + v_R_x**2)
    
    # 计算声速和特征速度（x方向）
    a_L_x = jnp.sqrt(gamma_L_x * p_L_x / rho_L_x)
    a_R_x = jnp.sqrt(gamma_R_x * p_R_x / rho_R_x)
    
    # KNP特征速度计算
    lambda_L_min_x = u_L_x - a_L_x
    lambda_L_max_x = u_L_x + a_L_x
    lambda_R_min_x = u_R_x - a_R_x
    lambda_R_max_x = u_R_x + a_R_x
    
    a_minus_x = jnp.minimum(0.0, jnp.minimum(lambda_L_min_x, lambda_R_min_x))
    a_plus_x  = jnp.maximum(0.0, jnp.maximum(lambda_L_max_x, lambda_R_max_x))
    
    # 计算左右通量（x方向）
    F_L = jnp.concatenate([
        rho_L_x * u_L_x,
        rho_L_x * u_L_x**2 + p_L_x,
        rho_L_x * u_L_x * v_L_x,
        u_L_x * (E_L_x + p_L_x),
        rho_L_x * u_L_x * Y_L_x
    ], axis=0)
    
    F_R = jnp.concatenate([
        rho_R_x * u_R_x,
        rho_R_x * u_R_x**2 + p_R_x,
        rho_R_x * u_R_x * v_R_x,
        u_R_x * (E_R_x + p_R_x),
        rho_R_x * u_R_x * Y_R_x
    ], axis=0)
    
    # 计算守恒变量（x方向）
    U_L_x = jnp.concatenate([rho_L_x, rho_L_x*u_L_x, rho_L_x*v_L_x, E_L_x, rho_L_x*Y_L_x], axis=0)
    U_R_x = jnp.concatenate([rho_R_x, rho_R_x*u_R_x, rho_R_x*v_R_x, E_R_x, rho_R_x*Y_R_x], axis=0)
    
    # KNP通量公式（x方向）
    denom_x = a_plus_x - a_minus_x + 1e-10  # 避免除零
    F_KNP = (a_plus_x * F_L - a_minus_x * F_R) / denom_x + \
            (a_plus_x * a_minus_x / denom_x) * (U_R_x - U_L_x)
    
    # 提取左右状态的基本物理量（y方向）
    rho_L_y, rho_R_y = q_L_y[0:1], q_R_y[0:1]
    p_L_y, p_R_y = q_L_y[3:4], q_R_y[3:4]
    Y_L_y, Y_R_y = q_L_y[4:], q_R_y[4:]
    
    # 计算热力学参数（y方向）
    R_L_y, R_R_y = thermo.get_R(Y_L_y), thermo.get_R(Y_R_y)
    T_L_y, T_R_y = p_L_y/(R_L_y*rho_L_y), p_R_y/(R_R_y*rho_R_y)
    _, gamma_L_y, h_L_y, _, _ = thermo.get_thermo(T_L_y, Y_L_y)
    _, gamma_R_y, h_R_y, _, _ = thermo.get_thermo(T_R_y, Y_R_y)
    
    # 计算速度和能量（y方向）
    u_L_y, u_R_y = q_L_y[1:2], q_R_y[1:2]
    v_L_y, v_R_y = q_L_y[2:3], q_R_y[2:3]
    E_L_y = rho_L_y*h_L_y - p_L_y + 0.5*rho_L_y*(u_L_y**2 + v_L_y**2)
    E_R_y = rho_R_y*h_R_y - p_R_y + 0.5*rho_R_y*(u_R_y**2 + v_R_y**2)
    
    # 计算声速和特征速度（y方向）
    a_L_y = jnp.sqrt(gamma_L_y * p_L_y / rho_L_y)
    a_R_y = jnp.sqrt(gamma_R_y * p_R_y / rho_R_y)
    
    # KNP特征速度计算
    lambda_L_min_y = v_L_y - a_L_y
    lambda_L_max_y = v_L_y + a_L_y
    lambda_R_min_y = v_R_y - a_R_y
    lambda_R_max_y = v_R_y + a_R_y
    
    a_minus_y = jnp.minimum(0.0, jnp.minimum(lambda_L_min_y, lambda_R_min_y))
    a_plus_y  = jnp.maximum(0.0, jnp.maximum(lambda_L_max_y, lambda_R_max_y))
    
    # 计算左右通量（y方向）
    G_L = jnp.concatenate([
        rho_L_y * v_L_y,
        rho_L_y * u_L_y * v_L_y,
        rho_L_y * v_L_y**2 + p_L_y,
        v_L_y * (E_L_y + p_L_y),
        rho_L_y * v_L_y * Y_L_y
    ], axis=0)
    
    G_R = jnp.concatenate([
        rho_R_y * v_R_y,
        rho_R_y * u_R_y * v_R_y,
        rho_R_y * v_R_y**2 + p_R_y,
        v_R_y * (E_R_y + p_R_y),
        rho_R_y * v_R_y * Y_R_y
    ], axis=0)
    
    # 计算守恒变量（y方向）
    U_L_y = jnp.concatenate([rho_L_y, rho_L_y*u_L_y, rho_L_y*v_L_y, E_L_y, rho_L_y*Y_L_y], axis=0)
    U_R_y = jnp.concatenate([rho_R_y, rho_R_y*u_R_y, rho_R_y*v_R_y, E_R_y, rho_R_y*Y_R_y], axis=0)
    
    # KNP通量公式（y方向）
    denom_y = a_plus_y - a_minus_y + 1e-10  # 避免除零
    G_KNP = (a_plus_y * G_L - a_minus_y * G_R) / denom_y + \
            (a_plus_y * a_minus_y / denom_y) * (U_R_y - U_L_y)
    
    return F_KNP, G_KNP
