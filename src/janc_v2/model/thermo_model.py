"""
The module extracts thermodynamic parameters 
from Cantera’s database and employs dedicated functions 
to compute critical thermodynamic state variables for gas mixtures: 
gas constant (R), enthalpy (h), specific heat ratio (gamma) and et al.

dependencies: jax & cantera(python version)
"""

import jax.numpy as jnp
from jax import vmap,lax,custom_vjp
import cantera as ct
from ..preprocess.load import get_cantera_coeffs
import os


max_iter = 5
tol = 5e-9

R_constant = None
gamma_constant = None
species_M = None
Mex = None
Tcr = None
cp_cof_low = None
cp_cof_high = None
dcp_cof_low = None
dcp_cof_high = None
h_cof_low = None
h_cof_high = None
h_cof_low_chem = None
h_cof_high_chem = None
s_cof_low = None
s_cof_high = None
logcof_low = None
logcof_high = None
n = None
thermo_model = 'nasa7'
gas_constant = 'Y-dependent'


def set_thermo(thermo_config,nondim_config=None,dim='2D'):
    global gamma_constant,gas_constant,R_constant,ReactionParams,thermo_model,n,species_M,Mex,Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high
    thermo_model = 'nasa7'
    gas_constant = 'Y-dependent'
    if thermo_config['thermo_model']=='nasa7':
        assert 'mechanism_directory' in thermo_config,"Please specify 'mechanism_directory' in your dict of settings"
        _, ext = os.path.splitext(thermo_config['mechanism_directory'])
        assert ext.lower() == '.yaml', "janc only read mech file with 【.yaml】 format, check https://cantera.org/3.1/userguide/ck2yaml-tutorial.html for more details"
        if not os.path.isfile(thermo_config['mechanism_directory']):
            raise FileNotFoundError('No mechanism file detected in the specified directory.')
    elif thermo_config['thermo_model']=='constant_gamma':
        thermo_model = 'constant_gamma'
        assert ('species' in thermo_config) or ('gas_constant' in thermo_config), "A list of strings containing the name of the species should be provided in the dict of settings with key name 'species'(Example:['H2','O2',...]) or value for gas constant with key name 'gas_constant' should be provided."
        assert 'gamma' in thermo_config, "The constant_gamma model require the value of gamma to be specified in the setting dict with key name 'gamma'."
        gamma_constant = thermo_config['gamma']
        if 'gas_constant' in thermo_config:
            gas_constant = 'constant'
            R_constant = thermo_config['gas_constant']
            
    else:
        raise KeyError("The thermo model you specified is not supported, only 'nasa7' or 'constant_gamma' can be specified.")
          
        
    if thermo_config['thermo_model']=='nasa7':
        gas = ct.Solution(thermo_config['mechanism_directory'])
        species_list = gas.species_names
        mech = thermo_config['mechanism_directory']
    else:
        if 'gas_constant' in thermo_config:
            species_list = ['N2']
        else:
            species_list = thermo_config['species']
        mech = 'gri30.yaml'#thermo_config['mechanism_directory']

    
    species_M,Mex,Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high = get_cantera_coeffs(species_list,mech,nondim_config,dim)


def fill_Y(Y):
    Y_last = 1.0 - jnp.sum(Y,axis=0,keepdims=True)
    return jnp.concatenate([Y,Y_last],axis=0)

def get_R_constant(Y):
    return jnp.full_like(Y[0:1],R_constant)

def get_R_Y_dependent(Y):
    Y = fill_Y(Y)
    #expand_axes = range(species_M.ndim, Y.ndim)
    #Mex = jnp.expand_dims(species_M, tuple(expand_axes))
    R = jnp.sum(1/Mex*Y,axis=0,keepdims=True)
    return R

R_dict = {'Y-dependent':get_R_Y_dependent,
          'constant':get_R_constant}

def get_R(Y):
    return R_dict[gas_constant](Y)

def get_thermo_properties_single(Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,T):
    mask = T<Tcr
    cp = jnp.where(mask, jnp.polyval(cp_cof_low, T), jnp.polyval(cp_cof_high, T))
    dcp = jnp.where(mask, jnp.polyval(dcp_cof_low, T), jnp.polyval(dcp_cof_high, T))
    h = jnp.where(mask, jnp.polyval(h_cof_low, T), jnp.polyval(h_cof_high, T))
    return cp, dcp, h

    
def get_thermo_properties(T):
    return vmap(get_thermo_properties_single,in_axes=(0,0,0,0,0,0,0,None))(Tcr,cp_cof_low,cp_cof_high,
                                       dcp_cof_low,dcp_cof_high,
                                       h_cof_low,h_cof_high,
                                       T)


def get_thermo_nasa7(T, Y):
    """
    thermo properties evaluation with nasa7 polynomial
    """
    R = get_R(Y)
    Y = fill_Y(Y)
    cp_i, dcp_i, h_i = get_thermo_properties(T[0])
    cp = jnp.sum(cp_i*Y,axis=0,keepdims=True)
    h = jnp.sum(h_i*Y,axis=0,keepdims=True)
    dcp = jnp.sum(dcp_i*Y,axis=0,keepdims=True)
    gamma = cp/(cp-R)
    return cp, gamma, h, R, dcp

def e_eqn(T, e, Y):
    cp, gamma, h, R, dcp = get_thermo_nasa7(T, Y)
    res = ((h - R*T) - e)
    dres_dT = (cp - R)
    ddres_dT2 = dcp
    return res, dres_dT, ddres_dT2, gamma

@custom_vjp
def get_T_nasa7(e,Y,initial_T):
    
    initial_res, initial_de_dT, initial_d2e_dT2, initial_gamma = e_eqn(initial_T,e,Y)

    def cond_fun(args):
        res, de_dT, d2e_dT2, T, gamma, i = args
        return (jnp.max(jnp.abs(res)) > tol) & (i < max_iter)

    def body_fun(args):
        res, de_dT, d2e_dT2, T, gamma, i = args
        delta_T = -2*res*de_dT/(2*jnp.power(de_dT,2)-res*d2e_dT2)
        T_new = T + delta_T
        res_new, de_dT_new, d2e_dT2_new, gamma_new = e_eqn(T_new,e,Y)
        return res_new, de_dT_new, d2e_dT2_new, T_new, gamma_new, i + 1

    initial_state = (initial_res, initial_de_dT, initial_d2e_dT2, initial_T, initial_gamma, 0)
    _, _, _, T_final, gamma_final, it = lax.while_loop(cond_fun, body_fun, initial_state)
    return jnp.concatenate([gamma_final, T_final],axis=0)
    
def get_T_fwd(e,Y,initial_T):
    aux_new = get_T_nasa7(e,Y,initial_T)
    return aux_new, (e,Y,aux_new)
    
def get_T_bwd(res, g):
    e, Y, aux_new = res
    T = aux_new[1:2]
    cp, _, h, R, dcp_dT = get_thermo(T,Y)
    cv = cp - R
    dcv_dT = dcp_dT
    dT_de = 1/cv
    
    dgamma_dT = (dcp_dT*cv-dcv_dT*cp)/(cv**2)
    dgamma_de = dgamma_dT*dT_de
    
    cp_i, dcp_i_dT, h_i = get_thermo_properties(T[0])
    e_i = h_i - 1/Mex*T
    dT_dY = (-e_i[0:-1]+e_i[-1:])/cv
    
    dR_dY = 1/Mex[0:-1]-1/Mex[-1:]
    dcp_dY = cp_i[0:-1]-cp_i[-1:]
    dcv_dY = dcp_dY - dR_dY
    
    dgamma_dY = (dcp_dY*cv-dcv_dY*cp)/(cv**2)
    dgamma_dY = dgamma_dT*dT_dY + dgamma_dY
    
    dL_dgamma = g[0:1]
    dL_dT = g[1:2]
    
    dL_de = dL_dgamma*dgamma_de + dL_dT*dT_de
    dL_dY = dL_dgamma*dgamma_dY + dL_dT*dT_dY
    
    
    return (dL_de, dL_dY, jnp.zeros_like(T))
    
get_T_nasa7.defvjp(get_T_fwd, get_T_bwd)

def get_thermo_constant_gamma(T, Y):
    R = get_R(Y)
    gamma = gamma_constant#thermo_settings['gamma']
    cp = gamma/(gamma-1)*R
    h = cp*T
    gamma = jnp.full_like(T,gamma)
    return cp, gamma, h, R, None


def get_T_constant_gamma(e,Y,initial_T=None):
    gamma = gamma_constant#thermo_settings['gamma']
    R = get_R(Y)
    T_final = e*(gamma-1)/R
    gamma_final = jnp.full_like(e,gamma)
    return jnp.concatenate([gamma_final, T_final],axis=0)

get_thermo_func_dict = {'nasa7':get_thermo_nasa7,
                        'constant_gamma':get_thermo_constant_gamma}

get_T_func_dict = {'nasa7':get_T_nasa7,
                   'constant_gamma':get_T_constant_gamma}

def get_thermo(T,Y):
    return get_thermo_func_dict[thermo_model](T,Y)

def get_T(e,Y,initial_T):
    return get_T_func_dict[thermo_model](e,Y,initial_T)



    
    
    






