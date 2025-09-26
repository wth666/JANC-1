import jax.numpy as jnp
from . import aux_func
from .reconstruction import reconstruction_L_x_dict,reconstruction_R_x_dict,\
                            reconstruction_L_y_dict,reconstruction_R_y_dict,\
                            reconstruction_x_dict,reconstruction_y_dict
from .finite_difference import d_dx_dict,d_dy_dict
from .riemann_solver import riemann_solver_dict
from .flux_splitting import split_flux_dict
from ..model import thermo_model, transport_model

solver_type = 'godunov'
interface_reconstruction = 'WENO5-JS'
riemann_solver = 'HLLC'
split_method = 'LF'
viscosity = 'off'
viscosity_discretization = 'CENTRAL6'


def set_flux_solver(flux_solver_config,transport_config=None,nondim_config=None):
    global solver_type,interface_reconstruction,riemann_solver,split_method,viscosity,viscosity_discretization
    solver_type = 'godunov'
    interface_reconstruction = 'WENO5-JS'
    riemann_solver = 'HLLC'
    split_method = 'LF'
    viscosity = 'off'
    viscosity_discretization = 'CENTRAL6'
    solver_type = flux_solver_config['solver_type']
    if solver_type == 'godunov':
        interface_reconstruction = flux_solver_config['interface_reconstruction']
        riemann_solver = flux_solver_config['riemann_solver']
    elif solver_type == 'flux_splitting':
        interface_reconstruction = flux_solver_config['interface_reconstruction']
        split_method = flux_solver_config['split_method']
    else:
        raise KeyError("JANC only support 'godunov' and 'flux_splitting'")
    
    if flux_solver_config['viscosity'] == 'on':
        viscosity = 'on'
        transport_model.set_transport(transport_config,nondim_config,'2D')
        if 'viscosity_discretization' in flux_solver_config:
            viscosity_discretization = flux_solver_config['viscosity_discretization']
        
def godunov_flux(U,aux,dx,dy):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U, aux)
    q = jnp.concatenate([rho,u,v,p,Y],axis=0)
    q_L_x = reconstruction_L_x_dict[interface_reconstruction](q)
    q_R_x = reconstruction_R_x_dict[interface_reconstruction](q)
    q_L_y = reconstruction_L_y_dict[interface_reconstruction](q)
    q_R_y = reconstruction_R_y_dict[interface_reconstruction](q)
    F, G = riemann_solver_dict[riemann_solver](q_L_x,q_R_x,q_L_y,q_R_y)
    dF = F[:,1:,:]-F[:,:-1,:]
    dG = G[:,:,1:]-G[:,:,:-1]
    net_flux = dF/dx + dG/dy
    return -net_flux

def flux_splitting(U,aux,dx,dy):
    Fplus,Fminus = split_flux_dict[split_method](1,U,aux)
    Gplus,Gminus = split_flux_dict[split_method](2,U,aux)
    Fp = reconstruction_L_x_dict[interface_reconstruction](Fplus)
    Fm = reconstruction_R_x_dict[interface_reconstruction](Fminus)
    Gp = reconstruction_L_y_dict[interface_reconstruction](Gplus)
    Gm = reconstruction_R_y_dict[interface_reconstruction](Gminus)
    F = Fp + Fm
    G = Gp + Gm
    dF = F[:,1:,:]-F[:,:-1,:]
    dG = G[:,:,1:]-G[:,:,:-1]
    net_flux = dF/dx + dG/dy
    return -net_flux

advective_flux_dict = {'godunov':godunov_flux,
                       'flux_splitting':flux_splitting}    

def advective_flux(U,aux,dx,dy):
    return advective_flux_dict[solver_type](U,aux,dx,dy)


def viscous_flux_node(U, aux, dx, dy):
    ρ,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    T = aux[1:2]
    cp_k, _, h_k = thermo_model.get_thermo_properties(T[0])
    cp, _, _, _, _ = thermo_model.get_thermo(T,Y)
    Y = thermo_model.fill_Y(Y)
    du_dx = d_dx_dict[viscosity_discretization](u,dx);  du_dy = d_dy_dict[viscosity_discretization](u,dy);
    dv_dx = d_dx_dict[viscosity_discretization](v,dx);  dv_dy = d_dy_dict[viscosity_discretization](v,dy);
    dT_dx = d_dx_dict[viscosity_discretization](T,dx);  dT_dy = d_dy_dict[viscosity_discretization](T,dy);
    dY_dx = d_dx_dict[viscosity_discretization](Y,dx);  dY_dy = d_dy_dict[viscosity_discretization](Y,dy);
    mu,mu_t = transport_model.mu(ρ,T,dx,dy,None,du_dx,du_dy,None,dv_dx,dv_dy,None,None,None,None)
    k = transport_model.kappa(mu, cp, mu_t)
    D_k = transport_model.D(mu,ρ,cp_k,mu_t)
    λ = -2/3*mu
    S11 = du_dx; S22 = dv_dy;
    S12 = 0.5 * (du_dy + dv_dx)
    div_u = S11 + S22
    τ_xx = 2*mu*du_dx + λ*div_u
    τ_yy = 2*mu*dv_dy + λ*div_u
    τ_xy = 2*mu*S12
    qx = -k * dT_dx
    qy = -k * dT_dy
    jx =  - ρ * D_k * dY_dx
    jy =  - ρ * D_k * dY_dy
    ex = jnp.sum(jx*h_k,axis=0,keepdims=True)
    ey = jnp.sum(jy*h_k,axis=0,keepdims=True)
    F_hat = jnp.concatenate([jnp.zeros_like(ρ),
                             τ_xx, τ_xy,
                             u*τ_xx + v*τ_xy - qx - ex, -jx[0:-1]], axis=0)
    G_hat = jnp.concatenate([jnp.zeros_like(ρ),
                             τ_xy, τ_yy,
                             u*τ_xy + v*τ_yy - qy - ey, -jy[0:-1]], axis=0)
    return F_hat, G_hat

def viscous_flux(U, aux, dx, dy):
    F_hat, G_hat = viscous_flux_node(U, aux, dx, dy)
    F_interface = reconstruction_x_dict[viscosity_discretization](F_hat)
    G_interface = reconstruction_y_dict[viscosity_discretization](G_hat)
    dF = F_interface[:,1:,:]-F_interface[:,:-1,:]
    dG = G_interface[:,:,1:]-G_interface[:,:,:-1]
    net_flux = dF/dx + dG/dy
    return net_flux

def NS_flux(U,aux,dx,dy):
    return advective_flux(U,aux,dx,dy) + viscous_flux(U, aux, dx, dy)

def Euler_flux(U,aux,dx,dy):
    return advective_flux(U,aux,dx,dy)

total_flux_dict = {'on':NS_flux,
                   'off':Euler_flux}

def total_flux(U,aux,dx,dy):
    return total_flux_dict[viscosity](U,aux,dx,dy)











