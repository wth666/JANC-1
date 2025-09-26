"""
The module extracts thermodynamic parameters 
from Cantera’s database and employs dedicated functions 
to compute critical thermodynamic state variables for gas mixtures: 
gas constant (R), enthalpy (h), specific heat ratio (gamma) and et al.

dependencies: jax & cantera(python version)
"""
from .LES import LES_SGS_dict
mu_ref = 1.716e-5
T_ref = 273.15
S = 110.4
nu0 = 1e-4
Pr = 0.72
Pr_t = 0.72
Sc = 0.72
Sc_t = 0.72
Le = 1.0
Le_t = 1.0
k0 = 0.0
D0 = 0.0
nu_type = 'Sutherland'
k_type = 'Prandtl number'
D_type = 'Lewis number'
model = 'laminar'
LES_model = 'WALE'

def set_transport(transport_config,nondim_config=None,dim='2D'):
    global Pr_t,Sc_t, Le_t,LES_model,model,nu_type, k_type, D_type, Pr, Sc, Le, nu0, k0, D0,mu_ref,T_ref,S
    mu_ref = 1.716e-5
    T_ref = 273.15
    S = 110.4
    nu0 = 1e-4
    Pr = 0.72
    Pr_t = 0.72
    Sc = 0.72
    Sc_t = 0.72
    Le = 1.0
    Le_t = 1.0
    k0 = 0.0
    D0 = 0.0
    nu_type = 'Sutherland'
    k_type = 'Prandtl number'
    D_type = 'Lewis number'
    model = 'laminar'
    LES_model = 'WALE'
    if transport_config['viscosity_model'] == 'constant':
        nu_type = 'constant'
        nu0 = transport_config['dynamic_viscosity']
    elif transport_config['viscosity_model'] == 'Sutherland':
        if 'reference_viscosity' in transport_config:
            mu_ref = transport_config['reference_viscosity']
        if 'reference_temperature' in transport_config:
            T_ref = transport_config['reference_temperature']
        if 'effective_temperature' in transport_config:
            S = transport_config['effective_temperature']
    else:
        raise KeyError("Only 'Sutherland' and 'constant' viscosity model are supported.")
    if transport_config['thermal_conductivity_model'] == 'constant':
        k_type = 'constant'
        k0 = transport_config['thermal_conductivity']
    elif transport_config['thermal_conductivity_model'] == 'Prandtl number':
        assert 'Pr' in transport_config, "Key 'Pr' must be provided"
        Pr = transport_config['Pr']
    else:
        raise KeyError("Only 'Prandtl number' and 'constant' thermal conductivity model are supported.")
    if transport_config['thermal_conductivity_model'] == 'constant':
        k_type = 'constant'
        k0 = transport_config['thermal_conductivity']
    elif transport_config['thermal_conductivity_model'] == 'Prandtl number':
        assert 'Pr' in transport_config, "Key 'Pr' must be provided"
        Pr = transport_config['Pr']
    else:
        raise KeyError("Only 'Prandtl number' and 'constant' thermal conductivity model are supported.")
    if transport_config['species_diffusivity_model'] == 'constant':
        D_type = 'constant'
        D0 = transport_config['species_diffusivity']
    elif transport_config['species_diffusivity_model'] == 'Schmidt number':
        assert 'Sc' in transport_config, "Key 'Sc' must be provided"
        Sc = transport_config['Sc']
        Le = Sc/Pr
    elif transport_config['species_diffusivity_model'] == 'Lewis number':
        assert 'Le' in transport_config, "Key 'Le' must be provided"
        Le = transport_config['Le']
        Sc = Le*Pr
    else:
        raise KeyError("Only 'Schmidt number', 'Lewis number', and 'constant' species diffusivity model are supported.")
    if 'turbulence_model' in transport_config:
        if transport_config['turbulence_model'] == 'LES':
            model = 'LES'
            assert 'LES_model' in transport_config, 'Please specify the SGS model of LES.'
            LES_model = transport_config['LES_model'] + '_' + dim
            assert 'Pr_t' in transport_config, "Key 'Pr_t'(turbulent Prandtl number) must be provided"
            assert ('Le_t' in transport_config) or ('Sc_t' in transport_config), "Key 'Sc_t'(turbulent Schmidt number) or 'Le_t'(turbulent Lewis number) must be provided"
            Pr_t = transport_config['Pr_t']
            if 'Le_t' in transport_config:
                Le_t = transport_config['Le_t']
                Sc_t = Le_t*Pr_t
            else:
                Sc_t = transport_config['Sc_t']
                Le_t = Sc_t/Pr_t
        elif transport_config['turbulence_model'] == 'laminar':
            pass
        else:
            raise KeyError("Only 'laminar' and 'LES' model are supported.")

def mu_Sutherland(rho,T):
    mu = mu_ref * (T/T_ref)**1.5 * (T_ref+S)/(T+S)
    return mu

def mu_constant(rho,T):
    return rho*nu0

mu_dict = {'Sutherland':mu_Sutherland,
           'constant':mu_constant}

def mu_laminar(rho,T,dx,dy,dz,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz):
    return mu_dict[nu_type](rho,T), 0.0

def mu_LES(rho,T,dx,dy,dz,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz):
    mu = mu_dict[nu_type](rho,T)
    mu_t = LES_SGS_dict[LES_model](rho,dx,dy,dz,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz)
    return mu + mu_t, mu_t

total_mu_dict = {'laminar':mu_laminar,
                 'LES':mu_LES}

def mu(rho,T,dx,dy,dz,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz):
    mu,mu_t = total_mu_dict[model](rho,T,dx,dy,dz,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz)
    return mu, mu_t

def kappa_Pr(mu,cp,mu_t):
    return mu*cp/Pr + mu_t*cp/Pr_t

def kappa_constant(mu,cp,mu_t):
    return k0 + mu_t*cp/Pr_t

kappa_dict = {'Prandtl number':kappa_Pr,
              'constant':kappa_constant}

def kappa(mu,cp,mu_t):
    return kappa_dict[k_type](mu,cp,mu_t)

def D_Le(mu,rho,cp_i,mu_t):
    k_i = (k_type=='Prandtl number')*mu*cp_i/Pr + (k_type=='constant')*k0
    rhoD_i = k_i/(Le*cp_i)
    k_t = mu_t/Pr_t
    rhoD_t = k_t/(Le_t*cp_i)
    return (rhoD_i+rhoD_t)/rho

def D_constant(mu,rho,cp_i,mu_t):
    k_t = mu_t/Pr_t
    rhoD_t = k_t/Le_t*cp_i
    return D0 + rhoD_t/rho

D_dict = {'Lewis number':D_Le,
          'constant':D_constant}

def D(mu,rho,cp_i,mu_t):
    return D_dict[D_type](mu,rho,cp_i,mu_t)



    
    
    








