from . import AUSMPW_plus, HLLC

riemann_solver_dict = {
    "AUSMPW_plus": AUSMPW_plus.riemann_flux,
    "HLLC": HLLC.riemann_flux
}
