from . import RK2,TVD_RK3

time_step_dict = {
    "RK2": RK2.advance_one_step,
    "RK2_amr": RK2.advance_one_step_amr,
    "TVD_RK3": TVD_RK3.advance_one_step,
    "TVD_RK3_amr": TVD_RK3.advance_one_step_amr
}



