import tensorflow as tf
import deepxde as dde
import numpy as np

from domain.params.solver_params import SolverParams
from domain.params.process_params import ProcessParams
from domain.params.altiok_2006_params import Altiok2006Params

class ODEPreparer:
    def __init__(
        self,
        solver_params: SolverParams,
        eq_params: Altiok2006Params,
        process_params: ProcessParams,
        f_out_value_calc,
    ):
        self.solver_params = solver_params
        self.eq_params = eq_params
        self.process_params = process_params
        self.f_out_value_calc = f_out_value_calc

    def prepare(self):
        def ode_system(x, y):
            """
            Order of outputs:

            X, P, S, V

            """
            # ---------------------------
            # Parameters
            mu_max = self.eq_params.mu_max
            K_S = self.eq_params.K_S
            alpha = self.eq_params.alpha
            beta = self.eq_params.beta
            Y_PS = self.eq_params.Y_PS
            ms = self.eq_params.ms
            f = self.eq_params.f
            h = self.eq_params.h
            Pm = self.eq_params.Pm
            Xm = self.eq_params.Xm
            inlet = self.process_params.inlet
            process_params = self.process_params
            solver_params = self.solver_params
            f_out_value_calc = self.f_out_value_calc

            # --------------------------
            # Volume & flows

            V = y[:, 3:4]

            dV_dt = dde.grad.jacobian(y, x, i=3)
            f_in = inlet.volume

            f_out = f_out_value_calc(
                max_reactor_volume=process_params.max_reactor_volume,
                f_in_v=f_in,
                volume=V,
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
