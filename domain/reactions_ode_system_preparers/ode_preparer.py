import tensorflow as tf
import deepxde as dde
import numpy as np

from domain.params.solver_params import SolverParams
from domain.params.process_params import ProcessParams
from domain.params.altiok_2006_params import Altiok2006Params
from domain.optimization.non_dim_scaler import NonDimScaler


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

            # Nondim Scale
            scaler = solver_params.non_dim_scaler

            # --------------------------
            # Volume & flows

            V_nondim = y[:, 3:4]

            dV_dt_nondim = dde.grad.jacobian(y, x, i=3)
            f_in = inlet.volume

            f_out = f_out_value_calc(
                max_reactor_volume=process_params.max_reactor_volume,
                f_in_v=f_in,
                volume=V_nondim*scaler.V,
            )

            # --------------------------
            # X P S
            X_nondim, P_nondim, S_nondim = y[:, 0:1], y[:, 1:2], y[:, 2:3]

            dX_dt_nondim = dde.grad.jacobian(y, x, i=0)
            dP_dt_nondim = dde.grad.jacobian(y, x, i=1)
            dS_dt_nondim = dde.grad.jacobian(y, x, i=2)

            if solver_params.non_dim_scaler is not None:
                # Equações auxiliares. Usadas para operações matemática e contornar um erro
                # específico de versões entre numpy e tensorflow
                def div(x, y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.divide(x, y)

                def mult(x, y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.multiply(x, y)

                def add(x, y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.add(x, y)

                def sub(x, y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.subtract(x, y)

                # --------------------------
                # Nondim Equations. Daqui pra baixo X,P,S, V (variáveis) etc já tá tudo adimensionalizado.
                def f_x_calc_func():
                    return tf.pow(
                        tf.math.subtract(
                            tf.cast(1, tf.float32),
                            tf.math.divide(tf.math.multiply(X_nondim, scaler.X), Xm),
                        ),
                        f,
                    )

                def h_p_calc_func():
                    return tf.pow(
                        tf.math.subtract(
                            tf.cast(1, tf.float32),
                            tf.math.divide(tf.math.multiply(P_nondim, scaler.P), Pm),
                        ),
                        h,
                    )


                non_dim_rX = (
                  mult(
                        mult(mult(div(scaler.t, scaler.X), mu_max), X_nondim),# mult(X_nondim, scaler.X)),
                        div(mult(S_nondim, scaler.S), add(K_S, mult(S_nondim, scaler.S))),
                    )
                    * f_x_calc_func()
                    * h_p_calc_func()
                )

                non_dim_rP = (scaler.t / scaler.P) * (
                    alpha * (scaler.X / scaler.t) * non_dim_rX + beta * X_nondim * scaler.X/scaler.t
                )
                non_dim_rS = (scaler.t / scaler.S) * (
                    -(1 / Y_PS) * non_dim_rP * (scaler.P / scaler.t) - ms * X_nondim * scaler.X/scaler.t
                )
                
                # Última mudança: adicionei o scaler t aos inlets
                # e o scaler V/t no volume, talvez por isso desse problema
                
                # NOVO NONDIM
                loss_pde = [
                    1
                    * 
                    (
                        dX_dt_nondim * V_nondim
                        - (
                            non_dim_rX * V_nondim
                            + f_in/scaler.V * inlet.X *scaler.t/scaler.X
                            - f_out/scaler.V * X_nondim * scaler.t
                        )
                    ),
                    1
                    * (
                        dP_dt_nondim * V_nondim
                        - (
                            non_dim_rP * V_nondim
                            + f_in/scaler.V * inlet.P *scaler.t/scaler.P
                            - f_out/scaler.V * P_nondim * scaler.t
                        )
                    ),
                    1
                    * (
                        dS_dt_nondim * V_nondim
                        - (
                            non_dim_rS * V_nondim
                            + f_in/scaler.V * inlet.S *scaler.t/scaler.S
                            - f_out/scaler.V * S_nondim * scaler.t
                        )
                    ),
                   1
                   * (dV_dt_nondim - ((f_in - f_out)*(scaler.t/scaler.V))),
                ]

                return loss_pde


        return ode_system
