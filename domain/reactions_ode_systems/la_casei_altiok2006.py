import tensorflow as tf

from domain.params.solver_params import SolverParams
from domain.params.process_params import ProcessParams


def la_casei_altiok2006_ode(
    solver_params: SolverParams, process_params: ProcessParams, f_out_value_calc
):

    """
    Returns the ode_system after applying the giving constants
    ode_system --> ode_system(x, y)

    Models and parameters from Altiok (2006)
        X --> Cell concentration
        P --> Product (Lactic Acid) concentration
        S --> Substract (Lactose Whey) concentration

    *f_out_value_calc é uma função do tipo f_out_value_calc(max_reactor_volume, f_in_v, volume)
    com o volume, vazão de entrada e v_max do reator decide a vazão saída.

    """

    def ode_system(x, y):

        """

        Order of outputs:

        X, P, S, V

        """
        # --------------------------
        # Volume & flows

        V = y[:, 3:4]

        dV_dt = dde.grad.jacobian(y, x, i=3)

        f_in = inlet.volume

        f_out = f_out_value_calc(
            max_reactor_volume=process_params.max_reactor_volume, f_in_v=f_in, volume=V
        )

        # --------------------------
        # X P S
        X, P, S = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        dX_dt = dde.grad.jacobian(y, x, i=0)
        dP_dt = dde.grad.jacobian(y, x, i=1)
        dS_dt = dde.grad.jacobian(y, x, i=2)

        # --------------------------
        # Reactions
        rX = (
            (X * mu_max * S / (K_S + S))
            * (tf.math.pow(1 - X / Xm, f))
            * (tf.math.pow(1 - P / Pm, h))
        )
        rP = alpha * dX_dt + beta * X
        rS = -(1 / Y_PS) * dP_dt - ms * X

        # --------------------------
        # return the error (difference) of:
        # X
        # P
        # S
        # V
        return [
            solver_params.w_X * (dX_dt * V - (rX * V + f_in * inlet.X - f_out * X)),
            solver_params.w_P * (dP_dt * V - (rP * V + f_in * inlet.P - f_out * P)),
            solver_params.w_S * (dS_dt * V - (rS * V + f_in * inlet.S - f_out * S)),
            solver_params.w_volume * (dV_dt - (f_in - f_out)),
        ]

    return ode_system
