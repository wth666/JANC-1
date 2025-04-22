"""
The module extracts thermodynamic parameters 
from Cantera’s database and employs dedicated functions 
to compute critical thermodynamic state variables for gas mixtures: 
gas constant (R), enthalpy (h), specific heat ratio (gamma) and et al.

dependencies: jax & cantera(python version)
"""

import jax.numpy as jnp
from jax import vmap,jit
from ..preprocess import nondim
from ..thermodynamics import thermo

Rg = 8.314463
P0 = nondim.P0


def reactionConstant_i(T, X, i, k, n):

    A = thermo.ReactionParams["A"][i]
    B = thermo.ReactionParams["B"][i]
    EakOverRu = thermo.ReactionParams["Ea/Ru"][i]
    vf_i = thermo.ReactionParams["vf"][i,:,:,:]
    vb_i = thermo.ReactionParams["vb"][i,:,:,:]
    vf_ik = vf_i[k,:,:]
    vb_ik = vb_i[k,:,:]
    vsum = thermo.ReactionParams["vsum"][i]
    aij = thermo.ReactionParams["third_body_coeffs"][i,:,:,:]


    kf_i = A*jnp.power(T,B)*jnp.exp(-EakOverRu/T)
    X = jnp.maximum(X,1e-20*jnp.ones_like(X))
    aij_X_sum = jnp.sum(aij*X,axis=0,keepdims=True)
    aij_X_sum = jnp.maximum(aij_X_sum, jnp.ones_like(aij_X_sum))
    X = X[0:thermo.n,:,:]
    log_X = jnp.log(X)
    kf = kf_i*jnp.exp(jnp.sum(vf_i*log_X,axis=0,keepdims=True))
    
    #cond = ReactionParams["is_reverse"][i]
    #kb = lax.cond(cond,
                  #lambda:kf_i/(jnp.exp(jnp.sum((vb_i-vf_i)*(get_gibbs(T[0,:,:])),axis=0,keepdims=True))*((101325/P0/T)**vsum))*jnp.exp(jnp.sum(vb_i*log_X,axis=0,keepdims=True)),
                  #lambda:jnp.zeros_like(kf))
    kb = kf_i/(jnp.exp(jnp.sum((vb_i-vf_i)*(thermo.get_gibbs(T[0,:,:])),axis=0,keepdims=True))*((101325/P0/T)**vsum))*jnp.exp(jnp.sum(vb_i*log_X,axis=0,keepdims=True))
    
    w_kOverM_i = (vb_ik-vf_ik)*aij_X_sum*(kf-kb)
    vb_in = vb_i[n]
    vf_in = vf_i[n]
    ain = thermo.ReactionParams["third_body_coeffs"][i,n]
    Mn = thermo.species_M[n]
    Xn = jnp.expand_dims(X[n,:,:],0)
    dwk_drhonYn_OverMk_i = (vb_ik-vf_ik)*(kf-kb)*ain/Mn + 1/(Mn*Xn)*(vb_ik-vf_ik)*aij_X_sum*(vf_in*kf-vb_in*kb)

    return w_kOverM_i, dwk_drhonYn_OverMk_i

def reaction_rate_with_derievative(T,X,k,n):
    Mk = thermo.species_M[k]
    i = jnp.arange(thermo.ReactionParams["num_of_reactions"])
    w_kOverM_i, dwk_drhonYn_OverMk_i = vmap(reactionConstant_i,in_axes=(None,None,0,None,None))(T,X,i,k,n)
    w_k = Mk*jnp.sum(w_kOverM_i,axis=0,keepdims=False)
    dwk_drhonYn = Mk*jnp.sum(dwk_drhonYn_OverMk_i,axis=0,keepdims=False)
    return w_k[0,:,:], dwk_drhonYn[0,:,:]

def construct_matrix_equation(T,X,dt):
    matrix_fcn = vmap(vmap(reaction_rate_with_derievative,in_axes=(None,None,None,0)),in_axes=(None,None,0,None))
    k = jnp.arange(thermo.n)
    n = jnp.arange(thermo.n)
    w_k, dwk_drhonYn = matrix_fcn(T,X,k,n)
    S = jnp.transpose(w_k[:,0:1,:,:],(2,3,0,1))
    DSDU = jnp.transpose(dwk_drhonYn,(2,3,0,1))
    I = jnp.eye(thermo.n)
    A = I/dt - DSDU
    b = S
    return A, b

@jit
def solve_implicit_rate(T,rho,Y,dt):
    Y = thermo.fill_Y(Y)
    rhoY = rho*Y
    X = rhoY/(thermo.Mex)
    A, b = construct_matrix_equation(T,X,dt)
    dU = jnp.linalg.solve(A,b)
    return jnp.transpose(dU[:,:,:,0],(2,0,1))


