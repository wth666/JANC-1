from . import zero_gradient,no_slip_wall,pressure_outlet,slip_wall


left_bd_dict_2D = {'zero_gradient':zero_gradient.left,
                'no_slip_wall':no_slip_wall.left,
                'pressure_outlet':pressure_outlet.left,
                'slip_wall':slip_wall.left}

right_bd_dict_2D = {'zero_gradient':zero_gradient.right,
                'no_slip_wall':no_slip_wall.right,
                'pressure_outlet':pressure_outlet.right,
                'slip_wall':slip_wall.right}

top_bd_dict_2D = {'zero_gradient':zero_gradient.top,
                'no_slip_wall':no_slip_wall.top,
                'pressure_outlet':pressure_outlet.top,
                'slip_wall':slip_wall.top}

bottom_bd_dict_2D = {'zero_gradient':zero_gradient.bottom,
                'no_slip_wall':no_slip_wall.bottom,
                'pressure_outlet':pressure_outlet.bottom,
                'slip_wall':slip_wall.bottom}


