import jax.numpy as jnp
from . import aux_func
from .reconstruction import reconstruction_L_x_dict,reconstruction_R_x_dict,reconstruction_x_dict
from .finite_difference import d_dx_dict
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
        transport_model.set_transport(transport_config,nondim_config,'1D')
        if 'viscosity_discretization' in flux_solver_config:
            viscosity_discretization = flux_solver_config['viscosity_discretization']
        
def godunov_flux(U,aux,dx):
    rho,u,Y,p,a = aux_func.U_to_prim(U, aux)
    q = jnp.concatenate([rho,u,p,Y],axis=0)
    q_L_x = reconstruction_L_x_dict[interface_reconstruction](q)
    q_R_x = reconstruction_R_x_dict[interface_reconstruction](q)
    F = riemann_solver_dict[riemann_solver](q_L_x,q_R_x)
    dF = F[:,1:]-F[:,:-1]
    net_flux = dF/dx
    return -net_flux

def flux_splitting(U,aux,dx):
    
    Fplus,Fminus = split_flux_dict[split_method](U,aux)
    Fp = reconstruction_L_x_dict[interface_reconstruction](Fplus)
    Fm = reconstruction_R_x_dict[interface_reconstruction](Fminus)
    F = Fp + Fm
    dF = F[:,1:]-F[:,:-1]
    net_flux = dF/dx
    return -net_flux

advective_flux_dict = {'godunov':godunov_flux,
                       'flux_splitting':flux_splitting}    

def advective_flux(U,aux,dx):
    return advective_flux_dict[solver_type](U,aux,dx)


def viscous_flux_node(U, aux, dx):
    ρ,u,Y,p,a = aux_func.U_to_prim(U,aux)
    T = aux[1:2]
    cp_k, _, h_k = thermo_model.get_thermo_properties(T[0])
    cp, _, _, _, _ = thermo_model.get_thermo(T,Y)
    Y = thermo_model.fill_Y(Y)
    du_dx = d_dx_dict[viscosity_discretization](u,dx);
    dT_dx = d_dx_dict[viscosity_discretization](T,dx);
    dY_dx = d_dx_dict[viscosity_discretization](Y,dx); 
    mu,mu_t = transport_model.mu(ρ,T,dx,None,None,du_dx,None,None,None,None,None,None,None,None)
    k = transport_model.kappa(mu,cp, mu_t)
    D_k = transport_model.D(mu,ρ,cp_k,mu_t)
    S11 = du_dx;
    div_u = S11
    τ_xx = 2*mu*du_dx-2/3*mu*div_u
    qx = -k * dT_dx
    jx =  - ρ * D_k * dY_dx
    ex = jnp.sum(jx*h_k,axis=0,keepdims=True)
    F_hat = jnp.concatenate([jnp.zeros_like(ρ),
                             τ_xx,
                             u*τ_xx - qx - ex, -jx[0:-1]], axis=0)
    return F_hat

def viscous_flux(U, aux, dx):
    F_hat = viscous_flux_node(U, aux, dx)
    F_interface = reconstruction_x_dict[viscosity_discretization](F_hat)
    dF = F_interface[:,1:]-F_interface[:,:-1]
    net_flux = dF/dx
    return net_flux

def NS_flux(U,aux,dx):
    return advective_flux(U,aux,dx) + viscous_flux(U, aux, dx)

def Euler_flux(U,aux, dx):
    return advective_flux(U,aux,dx)

total_flux_dict = {'on':NS_flux,
                   'off':Euler_flux}

def total_flux(U,aux,dx):
    return total_flux_dict[viscosity](U,aux,dx)











