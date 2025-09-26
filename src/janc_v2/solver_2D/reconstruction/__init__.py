from . import MUSCL, WENO5_JS, CENTRAL6

reconstruction_L_x_dict = {
    "MUSCL": MUSCL.interface_L_x,
    "WENO5_JS": WENO5_JS.interface_L_x
}

reconstruction_R_x_dict = {
    "MUSCL": MUSCL.interface_R_x,
    "WENO5_JS": WENO5_JS.interface_R_x
}

reconstruction_L_y_dict = {
    "MUSCL": MUSCL.interface_L_y,
    "WENO5_JS": WENO5_JS.interface_L_y
}

reconstruction_R_y_dict = {
    "MUSCL": MUSCL.interface_R_y,
    "WENO5_JS": WENO5_JS.interface_R_y
}

reconstruction_x_dict = {
    "CENTRAL6": CENTRAL6.interface_x
}

reconstruction_y_dict = {
    "CENTRAL6": CENTRAL6.interface_y
}