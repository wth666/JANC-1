from . import AUSMPW_plus, HLLC, KNP

riemann_solver_dict = {
    "AUSMPW_plus": AUSMPW_plus.riemann_flux,
    "HLLC": HLLC.riemann_flux,
    "KNP": KNP.riemann_flux
}

