import cantera as ct
import jax.numpy as jnp
from jax import vmap
import janc.nondim as nondim
#import os

Rg = 8.314463
rho0 = nondim.rho0
P0 = nondim.P0
M0 = nondim.M0
e0 = nondim.e0
T0 = nondim.T0
t0 = nondim.t0

def read_reaction_mechanism(file_path:str):
    #if not os.path.isfile(file_path):
        #raise FileNotFoundError('No mechanism file named ‘chem.txt’ detected in the current working directory.The chemical mechanism file must be explicitly named ‘chem.txt’ to enable proper system recognition and data parsing.')
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('!') and line.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('ELEMENTS'):
            species_list = lines[i+1].split()
            break
    species_idx = {sp: idx for idx, sp in enumerate(species_list)}
    reactions = []
    in_reactions_section = False
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith('REACTIONS'):
            in_reactions_section = True
            i += 1
            continue
        if line.startswith('END'):
            break
        if not in_reactions_section:
            i += 1
            continue

        if not line[0].isdigit():
            reaction_line = line.split('!')[0].strip()
            if ('<=>'or'=>') not in reaction_line:
                i += 1
                continue
            
            reaction = {
                'reactants': {},
                'products': {},
                'A': None,
                'B': None,
                'Ea': None,
                'third_body': None
                #'is_reverse':None
            }
            
            #if '=>' in reaction_line:
                #reactants_part, products_part = reaction_line.split('=>')
                #reaction['is_reverse'] = False
            
            if '<=>' in reaction_line:
                reactants_part, products_part = reaction_line.split('<=>')
                #reaction['is_reverse'] = True
                
            reactants = [s.strip() for s in reactants_part.split('+')]
            products = [s.strip() for s in products_part.split('+')]

        
            for species in reactants:
                if species and species != 'M':
                    reaction['reactants'][species] = reaction['reactants'].get(species, 0) + 1
            for species in products:
                if species and species != 'M':
                    reaction['products'][species] = reaction['products'].get(species, 0) + 1

            has_thirdbody = 'Thirdbody' in line
            i += 1

            if i < len(lines) and lines[i][0].isdigit():
                params = lines[i].split()
                reaction['A'] = float(params[0])
                reaction['B'] = float(params[1])
                reaction['Ea'] = float(params[2]) if len(params) > 2 else 0.0
                i += 1

                if has_thirdbody and i < len(lines) and '/' in lines[i]:
                    third_body = {}
                    if '/' in lines[i]:
                        for item in lines[i].split():
                            parts = item.strip('/').split('/')
                            if len(parts) >= 2:
                                species = parts[0].strip()
                                coeff = float(parts[1])
                                third_body[species] = coeff
                    else:
                        for sp in species_list:
                            third_body[sp] = 1.0

                    reaction['third_body'] = third_body
                    i += 1

            reactions.append(reaction)
        else:
            i += 1

    n_species = len(species_list)
    n_reactions = len(reactions)
    vf = jnp.zeros((n_reactions, n_species),dtype=jnp.int32)
    vb = jnp.zeros((n_reactions, n_species),dtype=jnp.int32)
    A = jnp.zeros(n_reactions)
    B = jnp.zeros(n_reactions)
    Ea = jnp.zeros(n_reactions)
    EaOverRu = jnp.zeros(n_reactions)
    third_body_coeffs = jnp.ones((n_reactions, n_species))
    is_third_body = jnp.zeros(n_reactions, dtype=bool)
    #is_reverse = jnp.zeros(n_reactions, dtype=bool)

    for i, rxn in enumerate(reactions):
        for sp, coeff in rxn['reactants'].items():
            if sp in species_idx:
                vf = vf.at[i, species_idx[sp]].set(round(coeff))
        for sp, coeff in rxn['products'].items():
            if sp in species_idx:
                vb = vb.at[i, species_idx[sp]].set(round(coeff))

        A = A.at[i].set(rxn['A'])
        B = B.at[i].set(rxn['B'])
        Ea = Ea.at[i].set(rxn['Ea'])

        if rxn['third_body']:
            is_third_body = is_third_body.at[i].set(True)
            third_body_coeffs = third_body_coeffs.at[i].set(1.0)
            for sp, coeff in rxn['third_body'].items():
                if sp in species_idx:
                    third_body_coeffs = third_body_coeffs.at[i, species_idx[sp]].set(coeff)
        else:
            third_body_coeffs = third_body_coeffs.at[i].set(0.0)
        
        #if rxn['is_reverse']:
            #is_reverse = is_reverse.at[i].set(True)

    compute_vf_sum = lambda i: jnp.sum(vf[i,:n_species+1],axis=0)
    compute_vb_sum = lambda i: jnp.sum(vb[i,:n_species+1],axis=0)
    vf_sum = vmap(compute_vf_sum)(jnp.arange(n_reactions))
    vb_sum = vmap(compute_vb_sum)(jnp.arange(n_reactions))
    A = A * jnp.power(10.0, -6.0 * (vf_sum - 1))
    A = jnp.where(is_third_body == True, A * jnp.power(10.0, -6.0), A)
    A = (t0/rho0)*(T0**B)*((rho0/M0)**vf_sum)*M0*A 
    Ea = Ea * 4.184
    EaOverRu = Ea/(e0*M0)
    third_body_coeffs = third_body_coeffs*(rho0/M0) 
    third_body_coeffs = jnp.expand_dims(third_body_coeffs,(2,3))
    
    non_zero_mask = jnp.any(vf + vb != 0, axis=0)
    zero_col_mask = jnp.all(vf + vb == 0, axis=0)
    vf = vf[:,non_zero_mask]
    vb = vb[:,non_zero_mask]
    
    num_of_inert_species = jnp.sum(zero_col_mask)
    zero_col_mask = jnp.all(vf + vb == 0, axis=0)
    zero_col_mask = jnp.asarray(zero_col_mask)
    
    inert_check = jnp.logical_or(~jnp.any(zero_col_mask),jnp.all(zero_col_mask[jnp.argmax(zero_col_mask):]))
    assert inert_check, "Inert species must be the last elements in species_list"
    reaction_params = {
        'species': species_list,
        'vf': jnp.expand_dims(vf,(2,3)),
        'vb': jnp.expand_dims(vb,(2,3)),
        'A': A,
        'B': B,
        'Ea': Ea,
        'Ea/Ru': EaOverRu,
        'third_body_coeffs': third_body_coeffs,
        'is_third_body': is_third_body,
        #'is_reverse':is_reverse,
        "num_of_reactions": n_reactions,
        "num_of_species": n_species,
        "num_of_inert_species":num_of_inert_species,
        "vsum":vb_sum - vf_sum
        }
    return reaction_params


def get_cantera_coeffs(species_list,mech='gri30.yaml'):
    gas = ct.Solution(mech)
    species_M = []
    Tcr = []
    coeffs_low = []
    coeffs_high = []
    for specie_name in species_list:
        sp = gas.species(specie_name)
        nasa_poly = sp.thermo
        Tcr.append(nasa_poly.coeffs[0])
        coeffs_low.append(nasa_poly.coeffs[8:15])
        coeffs_high.append(nasa_poly.coeffs[1:8])
        species_M.append(sp.molecular_weight)
    
    coeffs_low = jnp.array(coeffs_low)*jnp.array([[1,T0,T0**2,T0**3,T0**4,1/T0,1]])
    coeffs_high = jnp.array(coeffs_high)*jnp.array([[1,T0,T0**2,T0**3,T0**4,1/T0,1]])
    species_M = jnp.array(species_M)/(1000*M0)
    Mex = jnp.expand_dims(species_M,(1,2))
    
    Tcr = jnp.array(Tcr)/T0
    
    cp_cof_low = jnp.flip(coeffs_low[:,0:5],axis=1)/species_M[:,None]
    cp_cof_high = jnp.flip(coeffs_high[:,0:5],axis=1)/species_M[:,None]
    
    dcp_cof_low = cp_cof_low[:,0:-1]*jnp.array([[4,3,2,1]])
    dcp_cof_high = cp_cof_high[:,0:-1]*jnp.array([[4,3,2,1]])
    
    h_cof_low = jnp.flip(jnp.roll(coeffs_low[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])/species_M[:,None]
    h_cof_high = jnp.flip(jnp.roll(coeffs_high[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])/species_M[:,None]
    
    h_cof_low_chem = jnp.flip(jnp.roll(coeffs_low[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])
    h_cof_high_chem = jnp.flip(jnp.roll(coeffs_high[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])
    
    s_cof_low = jnp.flip(jnp.concatenate([coeffs_low[:,-1:],coeffs_low[:,1:5]],axis=1),axis=1)*jnp.array([[1/4,1/3,1/2,1,1]])
    s_cof_high = jnp.flip(jnp.concatenate([coeffs_high[:,-1:],coeffs_high[:,1:5]],axis=1),axis=1)*jnp.array([[1/4,1/3,1/2,1,1]])
    
    logcof_low = coeffs_low[:,0]
    logcof_high = coeffs_high[:,0]
    
    return species_M,Mex,Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high

