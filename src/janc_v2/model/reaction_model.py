"""
The module extracts thermodynamic parameters 
from Cantera’s database and employs dedicated functions 
to compute critical thermodynamic state variables for gas mixtures: 
gas constant (R), enthalpy (h), specific heat ratio (gamma) and et al.

dependencies: jax & cantera(python version)
"""


from jax import vmap
from .thermo_model import fill_Y
from ..preprocess import nondim
import jax.numpy as jnp
from ..preprocess.load import read_reaction_mechanism, get_cantera_coeffs
import os

Rg = 8.314463


species_M = None
Mex = None
Tcr = None
h_cof_low_chem = None
h_cof_high_chem = None
s_cof_low = None
s_cof_high = None
logcof_low = None
logcof_high = None
nr = None
nY = None
ReactionParams = {}
self_defined_source = None
source_func_type = 'detailed'
dimensions = '2D'

def set_reaction(reaction_config,nondim_config=None,dim='2D'):
    global nY,dimensions,source_func_type,self_defined_source,ReactionParams,nr,species_M,Mex,Tcr,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high
    dimensions = dim
    source_func_type = 'detailed'
    if reaction_config['is_detailed_chemistry']:
        assert 'mechanism_directory' in reaction_config,"You choosed detailed chemistry without specifying the directory of your mechanism files, please specify 'chemistry_mechanism_directory' in your dict of settings"
        _, ext = os.path.splitext(reaction_config['mechanism_directory'])
        assert ext.lower() == '.yaml', "janc only read mech file with 【.yaml】 format, check https://cantera.org/3.1/userguide/ck2yaml-tutorial.html for more details"
        if not os.path.isfile(reaction_config['mechanism_directory']):
            raise FileNotFoundError('No mechanism file detected in the specified directory.')
    else:
        if 'self_defined_reaction_source_terms' not in reaction_config:
            def self_defined_source(U,aux,theta=None):
                return jnp.zeros_like(U)
        else:
            assert 'self_defined_reaction_source_terms' in reaction_config,"Please pass the function handle of your own reaction source."
            assert callable(reaction_config['self_defined_reaction_source_terms']),'This source_term is not a python callable function.'
            self_defined_source = reaction_config['self_defined_reaction_source_terms']
            source_func_type = 'user_defined'
        
    if reaction_config['is_detailed_chemistry']:
        ReactionParams = read_reaction_mechanism(reaction_config['mechanism_directory'],nondim_config,dimensions)
        mech = reaction_config['mechanism_directory']
        species_list = ReactionParams['species']
        ns = ReactionParams['num_of_species']
        ni = ReactionParams['num_of_inert_species']
        nr = ns - ni
    else:
        species_list = ['N2']
        mech = 'gri30.yaml'

    if dim == '1D':
        nY = 3
    if dim == '2D':
        nY = 4
    if dim == '3D':
        nY = 5
     
    species_M,Mex,Tcr,_,_,_,_,_,_,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high = get_cantera_coeffs(species_list,mech,nondim_config,dimensions)


def get_gibbs_single(Tcr,h_cof_low,h_cof_high,s_cof_low,s_cof_high,logcof_low,logcof_high,T):
    mask = T<Tcr
    h = jnp.where(mask, jnp.polyval(h_cof_low, T), jnp.polyval(h_cof_high, T))
    s = jnp.where(mask, jnp.polyval(s_cof_low, T) + logcof_low*jnp.log((nondim.T0)*T), jnp.polyval(s_cof_high, T) + logcof_high*jnp.log((nondim.T0)*T))
    g = s - h/T
    return g

def get_gibbs(T):
    return vmap(get_gibbs_single,in_axes=(0,0,0,0,0,0,0,None))(Tcr[0:nr],h_cof_low_chem[0:nr],h_cof_high_chem[0:nr],
                                   s_cof_low[0:nr],s_cof_high[0:nr],
                                   logcof_low[0:nr],logcof_high[0:nr],
                                   T)

def reactionConstant_i(T, X, i, k, n):

    A = ReactionParams["A"][i]
    B = ReactionParams["B"][i]
    EakOverRu = ReactionParams["Ea/Ru"][i]
    vf_i = ReactionParams["vf"][i]
    vb_i = ReactionParams["vb"][i]
    vf_ik = vf_i[k]
    vb_ik = vb_i[k]
    vsum = ReactionParams["vsum"][i]
    aij = ReactionParams["third_body_coeffs"][i]
    is_third_body = ReactionParams['is_third_body'][i]


    kf_i = A*jnp.power(T,B)*jnp.exp(-EakOverRu/T)
    aij_X_sum = jnp.sum(aij*X,axis=0,keepdims=True)
    aij_X_sum = is_third_body*aij_X_sum + (1-is_third_body)
    X = jnp.clip(X,min = 1e-50)
    log_X = jnp.log(X[0:nr])
    kf = kf_i*jnp.exp(jnp.sum(vf_i*log_X,axis=0,keepdims=True))
    

    kb = kf_i/(jnp.exp(jnp.sum((vb_i-vf_i)*(get_gibbs(T[0])),axis=0,keepdims=True))*((101325/nondim.P0/T)**vsum))*jnp.exp(jnp.sum(vb_i*log_X,axis=0,keepdims=True))
    
    w_kOverM_i = (vb_ik-vf_ik)*aij_X_sum*(kf-kb)
    vb_in = vb_i[n]
    vf_in = vf_i[n]
    ain = ReactionParams["third_body_coeffs"][i,n]
    Mn = species_M[n]
    Xn = jnp.expand_dims(X[n],0)
    dwk_drhonYn_OverMk_i = (vb_ik-vf_ik)*(kf-kb)*(ain/Mn) + 1/(Mn*Xn)*(vb_ik-vf_ik)*aij_X_sum*(vf_in*kf-vb_in*kb)
    return w_kOverM_i, dwk_drhonYn_OverMk_i

def reaction_rate_with_derievative(T,X,k,n):
    Mk = species_M[k]
    i = jnp.arange(ReactionParams["num_of_reactions"])
    w_kOverM_i, dwk_drhonYn_OverMk_i = vmap(reactionConstant_i,in_axes=(None,None,0,None,None))(T,X,i,k,n)
    w_k = Mk*jnp.sum(w_kOverM_i,axis=0,keepdims=False)
    dwk_drhonYn = Mk*jnp.sum(dwk_drhonYn_OverMk_i,axis=0,keepdims=False)
    return w_k[0], dwk_drhonYn[0]

def construct_matrix_equation_1D(T,X,dt):
    matrix_fcn = vmap(vmap(reaction_rate_with_derievative,in_axes=(None,None,None,0)),in_axes=(None,None,0,None))
    k = jnp.arange(nr)
    n = jnp.arange(nr)
    w_k, dwk_drhonYn = matrix_fcn(T,X,k,n)
    S = jnp.transpose(w_k[:,0:1,:],(2,0,1))
    DSDU = jnp.transpose(dwk_drhonYn,(2,0,1))
    I = jnp.eye(nr)
    A = I/dt - DSDU
    b = S
    x = jnp.linalg.solve(A,b)
    drhoY = jnp.transpose(x[:,:,0],(1,0))
    return drhoY


def construct_matrix_equation_2D(T,X,dt):
    matrix_fcn = vmap(vmap(reaction_rate_with_derievative,in_axes=(None,None,None,0)),in_axes=(None,None,0,None))
    k = jnp.arange(nr)
    n = jnp.arange(nr)
    w_k, dwk_drhonYn = matrix_fcn(T,X,k,n)
    S = jnp.transpose(w_k[:,0:1,:,:],(2,3,0,1))
    DSDU = jnp.transpose(dwk_drhonYn,(2,3,0,1))
    I = jnp.eye(nr)
    A = I/dt - DSDU
    b = S
    x = jnp.linalg.solve(A,b)
    drhoY = jnp.transpose(x[:,:,:,0],(2,0,1))
    return drhoY

def construct_matrix_equation_3D(T,X,dt):
    matrix_fcn = vmap(vmap(reaction_rate_with_derievative,in_axes=(None,None,None,0)),in_axes=(None,None,0,None))
    k = jnp.arange(nr)
    n = jnp.arange(nr)
    w_k, dwk_drhonYn = matrix_fcn(T,X,k,n)
    S = jnp.transpose(w_k[:,0:1,:,:,:],(2,3,4,0,1))
    DSDU = jnp.transpose(dwk_drhonYn,(2,3,4,0,1))
    I = jnp.eye(nr)
    A = I/dt - DSDU
    b = S
    x = jnp.linalg.solve(A,b)
    drhoY = jnp.transpose(x[:,:,:,:,0],(3,0,1,2))
    return drhoY

matrix_equation_dict = {'1D':construct_matrix_equation_1D,
                        '2D':construct_matrix_equation_2D,
                        '3D':construct_matrix_equation_3D}

def construct_matrix_equation(T,X,dt):
    drhoY = matrix_equation_dict[dimensions](T,X,dt)
    return drhoY

def detailed_reaction(U,aux,dt,theta=None):
    rho = U[0:1]
    Y = fill_Y(U[nY:]/rho)
    rhoY = rho*Y
    T = aux[1:2]
    X = rhoY/(Mex)
    drhoY = construct_matrix_equation(T,X,dt)
    dY = drhoY/rho
    dY = jnp.clip(dY,min=-Y[0:-1],max=1-Y[0:-1])
    S = jnp.concatenate([jnp.zeros_like(U[:nY]),rho*dY],axis=0)
    return S

def user_reaction(U,aux,dt,theta=None):
    user_source = self_defined_source(U,aux,theta)
    return user_source*dt

reaction_func_dict = {'detailed':detailed_reaction,
                      'user_defined':user_reaction}

def reaction_source_terms(U,aux,dt,theta=None):
    return reaction_func_dict[source_func_type](U,aux,dt,theta)


















