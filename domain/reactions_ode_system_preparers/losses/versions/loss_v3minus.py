
def lossV3minus(o, args):
    (
        X,
        P,
        S,
        V,
        dXdt,
        dPdt,
        dSdt,
        dVdt,
        rX,
        rP,
        rS,
        inlet,
        f_in,
        f_out,
        Xm,
        Pm,
        initial_state,
        process_params,
    ) = args
    if o == "X":
        deriv_calc = V * rX + (f_in * inlet.X - f_out * X)
        loss_derivative = dXdt * V - deriv_calc
        loss_X = loss_derivative
        return loss_X
    if o == "P":
        deriv_calc = V * rP + (f_in * inlet.P - f_out * P)
        loss_derivative = dPdt * V - deriv_calc
        loss_P = loss_derivative
        return loss_P
    if o == "S":
        deriv_calc = V * rS + (f_in * inlet.S - f_out * S)
        loss_derivative = dSdt * V - deriv_calc
        loss_S = loss_derivative
        return loss_S
    if o == "V":
        dVdt_calc = f_in - f_out
        loss_derivative = dVdt - dVdt_calc
        loss_V = loss_derivative
        return loss_V
