usr_boundary_func = None
#boundaryParams = {}

def set_boundary(boundary:dict):
    global usr_boundary_func
    assert (('boundary_conditions' in boundary) and (boundary['boundary_conditions'] is not None)),"funtions on boundary conditions must be provided."
    usr_boundary_func = boundary['boundary_conditions']
    #boundaryParams = boundary
    #return boundaryParams

#user-defined-functions#
def boundary_conditions(U,aux):
    return usr_boundary_func(U,aux)