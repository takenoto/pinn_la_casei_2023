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

            # TODO comentei isso aqui
            # Não é pra fazer diferença quando for tudo 1...
            scaler = solver_params.non_dim_scaler
            # Isso aqui esculhamba tudo mesmo com todas as scalers = 1
            # X = X / scaler.X
            # P = Pm / scaler.P
            # S = S / scaler.S
            # V = V / scaler.V
            
            # Assim dá certo:
            # X = tf.math.divide(X, scaler.X)
            # P = tf.math.divide(P, scaler.X)
            # S = tf.math.divide(S, scaler.S)
            # V = tf.math.divide(V, scaler.V)
            # FIXME sem esse código de non_dim dá certo...
            if solver_params.non_dim_scaler is not None:
                # --------------------------
                # Nondim Scale
                scaler = solver_params.non_dim_scaler

                def div(x,y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.divide(x, y)

                def mult(x,y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.multiply(x, y)

                def add(x,y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.add(x, y)

                def sub(x,y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    return tf.math.subtract(x, y)

                # --------------------------
                # Nondim Equations. Daqui pra baixo X,P,S, V (variáveis) etc já tá tudo adimensionalizado.

                # def f_x_calc_func() : return tf.pow(1 - (X * scaler.X / Xm), f)
                # def h_p_calc_func() : return tf.pow(1 - (P * scaler.P / Pm), h)
                def f_x_calc_func():
                    return tf.pow(
                        tf.math.subtract(
                            tf.cast(1,tf.float32), tf.math.divide(tf.math.multiply(X, scaler.X), Xm)
                        ),
                        f,
                    )

                def h_p_calc_func():
                    return tf.pow(
                        tf.math.subtract(
                            tf.cast(1,tf.float32), tf.math.divide(tf.math.multiply(P, scaler.P), Pm)
                        ),
                        h,
                    )

                # def return_zero() : return tf.cast(0, tf.float32)

                # f_x_calc = tf.cond(
                #     tf.reshape(tf.math.greater_equal(X*scaler.X  / Xm, 1),[]),
                #     true_fn=return_zero,
                #     false_fn=f_x_calc_func,
                # )

                # h_p_calc = tf.cond(
                #     tf.reshape(tf.math.greater_equal(P * scaler.P / Pm, 1),[]),
                #     true_fn=return_zero,
                #     false_fn=h_p_calc_func,
                # )

                # print('AAAAAAAA PRINT')
                # print(f_x_calc)
                # print(h_p_calc)
                # Será que era isso? Os pows não podem ser avaliados inline, tinha que ser
                # calculado antes ou retornado por uma função:
                # non_dim_rX = (
                #     (scaler.t / scaler.X)
                #     * mu_max
                #     * (X * scaler.X)
                #     # Eu simplesmente tinha esquecido esses termos do S
                #     * (S*scaler.S/(K_S + S*scaler.S))
                #     * f_x_calc_func()
                #     * h_p_calc_func()
                # )
                
                # Isso resolve, realmente era um bug com numpy e tensorflow
                # e como que eu adivinho pelo amor???????????????????????
                non_dim_rX = (
                   mult(mult( mult(div(scaler.t, scaler.X)
                    , mu_max)
                    , mult(X , scaler.X))
                    , div(mult(S,scaler.S),add(K_S, mult(S,scaler.S))))
                    * f_x_calc_func()
                    * h_p_calc_func()
                )


                non_dim_rP = (scaler.t / scaler.P) * (
                    alpha * (scaler.X / scaler.t) * non_dim_rX + beta * X * scaler.X
                )
                non_dim_rS = (scaler.t / scaler.S) * (
                    -(1 / Y_PS) * non_dim_rP * (scaler.P / scaler.t) - ms * X * scaler.X
                )
                

                # # Reações método tradicional: 
                # rX = (
                # (X * mu_max * S / (K_S + S))
                # * (tf.math.pow(1 - X / Xm, f))
                # * (tf.math.pow(1 - P / Pm, h))
                # )
                # rP = alpha * rX + beta * X
                # rS = -(1 / Y_PS) * rP - ms * X

                # return [
                #     solver_params.w_X * (dX_dt * V - (rX * V + f_in * inlet.X - f_out * X)),
                #     solver_params.w_P * (dP_dt * V - (rP * V + f_in * inlet.P - f_out * P)),
                #     solver_params.w_S * (dS_dt * V - (rS * V + f_in * inlet.S - f_out * S)),
                #     solver_params.w_volume * (dV_dt - (f_in - f_out)),
                #     ]

                return [
                    # o problema é 100% no x
                    solver_params.w_X
                    * (
                        dX_dt * V * scaler.V
                        - (
                            non_dim_rX * V * scaler.V
                            + f_in * inlet.X  # * scaler.X
                            - f_out * X * scaler.X
                        )
                    ),
                    # dX_dt,
                    # p e s ok
                    solver_params.w_P
                    * (
                        dP_dt * V * scaler.V
                        - (
                            non_dim_rP * V * scaler.V
                            + f_in * inlet.P  # * scaler.P
                            - f_out * P * scaler.P
                        )
                    ),
                    solver_params.w_S
                    * (
                        dS_dt * V * scaler.V
                        - (
                            non_dim_rS * V * scaler.V
                            + f_in * inlet.S  # * scaler.S
                            - f_out * S * scaler.S
                        )
                    ),
                    # volume ok
                    solver_params.w_volume * (dV_dt - (f_in - f_out)),
                ]

                return [
                    solver_params.w_X
                    * (
                        dX_dt * V * scaler.V
                        - (
                            non_dim_rX * V * scaler.V
                            + f_in * inlet.X  # * scaler.X
                            - f_out * X * scaler.X
                        )
                    ),
                    solver_params.w_P
                    * (
                        dP_dt * V * scaler.V
                        - (
                            non_dim_rP * V * scaler.V
                            + f_in * inlet.P  # * scaler.P
                            - f_out * P * scaler.P
                        )
                    ),
                    solver_params.w_S
                    * (
                        dS_dt * V * scaler.V
                        - (
                            non_dim_rS * V * scaler.V
                            + f_in * inlet.S  # * scaler.S
                            - f_out * S * scaler.S
                        )
                    ),
                    # O erro era aqui
                    # dV_dt já vem escalado! É o fornecido pelo solver!
                    # FIXME mesmo retornando o dV_dt (ou seja, zerando a derivada) dá erro e NaN!
                    # dV_dt,
                    solver_params.w_volume * (dV_dt - (f_in - f_out)),
                ]

            # --------------------------
            # Reactions
            # FIXME Descobri um erro: não era dx_dt e sim rx, rp e rs nas reações
            # No dele só foi dx-dt e afins porque era batelada
            rX = (
                (X * mu_max * S / (K_S + S))
                * (tf.math.pow(1 - X / Xm, f))
                * (tf.math.pow(1 - P / Pm, h))
            )
            # rP = alpha * dX_dt + beta * X
            # rS = -(1 / Y_PS) * dP_dt - ms * X
            # rP = alpha * rX + beta * X
            # rS = -(1 / Y_PS) * rP - ms * X
            rP = alpha * rX + beta * X
            rS = -(1 / Y_PS) * rP - ms * X

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
                # isso não deixa o V constante.......... ave
                # dV_dt
                solver_params.w_volume * (dV_dt - (f_in - f_out)),
            ]

        return ode_system
