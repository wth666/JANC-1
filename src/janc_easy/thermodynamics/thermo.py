from ..preprocess import nondim
R = 1.0
gamma = 1.4

def set_thermo(thermo_config,nondim_config=None):
    global R,gamma
    R = thermo_config['gas_constant']/nondim.R0
    gamma = thermo_config['gamma']


    
    
    


