# 2023-12-08 mudei h_p_calc, estava fazendo a avaliação greater_equal pro X mds

# import tensorflow as tf
from deepxde.backend import tf
import deepxde as dde

from domain.params.solver_params import SolverParams
from domain.params.process_params import ProcessParams
from domain.params.altiok_2006_params import Altiok2006Params
import keras.backend as K

from domain.reactor.reactor_state import ReactorState


class ODEPreparer:
    def __init__(
        self,
        solver_params: SolverParams,
        eq_params: Altiok2006Params,
        process_params: ProcessParams,
        initial_state: ReactorState,
        f_out_value_calc,
    ):
        self.solver_params = solver_params
        self.eq_params = eq_params
        self.process_params = process_params
        self.f_out_value_calc = f_out_value_calc
        self.initial_state = initial_state

    def prepare(self):
        def ode_system(x, y):
            """
            Order of outputs:

            X, P, S, V

            """

            # ---------------------
            # PARAMETERS AND UTILITY
            # ---------------------
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
            inputSimulationType = self.solver_params.inputSimulationType
            outputSimulationType = self.solver_params.outputSimulationType
            initial_state = self.initial_state
            scaler = solver_params.non_dim_scaler

            X_nondim = None
            P_nondim = None
            S_nondim = None
            V_nondim = 1
            dX_dt_nondim = None
            dP_dt_nondim = None
            dS_dt_nondim = None
            dV_dt_nondim = 0
            # ---------------------
            # OUTPUT VARIABLES
            # ---------------------
            if outputSimulationType.X:
                X_nondim = y[
                    :, outputSimulationType.X_index : outputSimulationType.X_index + 1
                ]
                dX_dt_nondim = dde.grad.jacobian(
                    y, x, i=outputSimulationType.X_index, j=inputSimulationType.t_index
                )
            if outputSimulationType.P:
                P_nondim = y[
                    :, outputSimulationType.P_index : outputSimulationType.P_index + 1
                ]
                dP_dt_nondim = dde.grad.jacobian(
                    y, x, i=outputSimulationType.P_index, j=inputSimulationType.t_index
                )
            if outputSimulationType.S:
                S_nondim = y[
                    :, outputSimulationType.S_index : outputSimulationType.S_index + 1
                ]
                dS_dt_nondim = dde.grad.jacobian(
                    y, x, i=outputSimulationType.S_index, j=inputSimulationType.t_index
                )
            if outputSimulationType.V:
                V_nondim = y[
                    :, outputSimulationType.V_index : outputSimulationType.V_index + 1
                ]
                dV_dt_nondim = dde.grad.jacobian(
                    y, x, i=outputSimulationType.V_index, j=inputSimulationType.t_index
                )

            # ---------------------
            # INPUT VARIABLES
            # ---------------------
            if inputSimulationType.X:
                X_nondim = x[
                    :, inputSimulationType.X_index : inputSimulationType.X_index + 1
                ]
                dX_dt_nondim = dde.grad.jacobian(
                    x, x, i=inputSimulationType.X_index, j=outputSimulationType.t_index
                )
            if inputSimulationType.P:
                P_nondim = x[
                    :, inputSimulationType.P_index : inputSimulationType.P_index + 1
                ]
                dP_dt_nondim = dde.grad.jacobian(
                    x, x, i=inputSimulationType.P_index, j=inputSimulationType.t_index
                )
            if inputSimulationType.S:
                S_nondim = x[
                    :, inputSimulationType.S_index : inputSimulationType.S_index + 1
                ]
                dS_dt_nondim = dde.grad.jacobian(
                    x, x, i=inputSimulationType.S_index, j=inputSimulationType.t_index
                )
            if inputSimulationType.V:
                V_nondim = x[
                    :, inputSimulationType.V_index : inputSimulationType.V_index + 1
                ]
                dV_dt_nondim = dde.grad.jacobian(
                    x, x, i=inputSimulationType.V_index, j=inputSimulationType.t_index
                )

            # ------------------------
            # ---- DECLARING AS "N" --
            # ------------------------
            # Se não existir, faz o default de volume pra 1 e dVdt pra 0 pra
            # possibilitar cálculos
            N_nondim = {
                "X": X_nondim,
                "P": P_nondim,
                "S": S_nondim,
                "V": V_nondim,
                "dXdt": dX_dt_nondim,
                "dPdt": dP_dt_nondim,
                "dSdt": dS_dt_nondim,
                "dVdt": dV_dt_nondim,
            }

            # Nondim to dim
            N = {type: scaler.fromNondim(N_nondim, type) for type in N_nondim}
            X = N["X"]
            P = N["P"]
            S = N["S"]
            V = N["V"]
            dXdt = N["dXdt"]
            dPdt = N["dPdt"]
            dSdt = N["dSdt"]
            dVdt = N["dVdt"]

            # ------------------------
            # ---- INLET AND OUTLET --
            # ------------------------
            f_in = inlet.volume

            f_out = f_out_value_calc(
                max_reactor_volume=process_params.max_reactor_volume,
                f_in_v=f_in,
                volume=V,
            )

            # Declara a List da loss pra já deixar guardado e ir adicionando
            # conforme for sendo validado
            loss_pde = []

            # --------------------------
            # Nondim Equations. Daqui pra baixo X,P,S, V (variáveis) etc já tá tudo
            # adimensionalizado.

            # Ok, são 2 problemas
            # 1) Quando X>Xm
            # 2) Quando X=Xm, porque 0^algo também dá NaN (https://github.com/tensorflow/tensorflow/issues/16271)
            # Por algum motivo todas as tentativas anteriores, com keras.switch,
            # tf.wherenão funcionaram
            # Só funciona se fizer a atribuição do valor de X ou P por fora
            # e usar esses novos valores nos cáculos.
            # Se fizer um tf.where tendo como condicional o próprio X ou o valor da expressão não ser NaN, não presta, não adianta.

            def f_x_calc_func():
                loss_version = solver_params.get_loss_version_for_type("X")
                X_for_calc = X
                if loss_version > 2:
                    X_for_calc = tf.where(
                        tf.less(X, Xm), X, tf.ones_like(X) * 0.9999 * Xm
                    )

                value = tf.pow(1 - X_for_calc / Xm, f)
                return value

            def h_p_calc_func():
                loss_version = solver_params.get_loss_version_for_type("P")
                P_for_calc = P
                if loss_version > 2:
                    P_for_calc = tf.where(
                        tf.less(P, Pm), P, tf.ones_like(P) * 0.9999 * Pm
                    )

                value = tf.pow(
                    1 - (P_for_calc / Pm),
                    h,
                )
                return value

            rX = (X * mu_max * S / (K_S + S)) * f_x_calc_func() * h_p_calc_func()
            rP = alpha * rX + beta * X
            rS = -(1 / Y_PS) * rP - ms * X

            # -----------------------
            # Calculating the loss
            # Procura cada variável registrada como de saída e
            # adiciona o cálculo da sua função como componente da loss

            # Zerar reações na marra para testar modelo sem reação
            # rX = 0
            # rP = 0
            # rS = 0

            for o in outputSimulationType.order:
                # Pode dar NaN quando predizer valores abaixo de 0
                # Então evite divisões!!!! Por isso o V vem multiplicando no fim...
                loss_version = solver_params.get_loss_version_for_type(o)

                if o == "X":
                    loss_derivative = dXdt * V - (V * rX + (f_in * inlet.X - f_out * X))
                    loss_X = 0
                    if loss_version >= 6:
                        loss_maxmin = tf.where(
                            tf.less(X, 0),
                            X,
                            tf.where(
                                tf.greater(X, tf.ones_like(X) * Xm),
                                tf.ones_like(X) * Xm,
                                tf.zeros_like(X),
                            ),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_X = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version == 5:
                        loss_maxmin = tf.where(
                            tf.less(X, 0),
                            tf.math.pow(X, 3),
                            tf.where(
                                tf.greater(X, tf.ones_like(X) * Xm),
                                X - Xm,
                                tf.zeros_like(X),
                            ),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_X = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version == 4:
                        # if X<0, return X
                        # else if X>Xm, return X
                        # else return loss_X
                        loss_maxmin = tf.where(
                            tf.less(X, 0),
                            X,
                            tf.where(tf.greater(X, Xm), X - Xm, tf.zeros_like(X)),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_X = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version <= 3:
                        loss_X = loss_derivative
                    loss_pde.append(loss_X)

                elif o == "P":
                    loss_derivative = dPdt * V - (V * rP + (f_in * inlet.P - f_out * P))
                    loss_P = 0
                    if loss_version >= 6:
                        loss_maxmin = tf.where(
                            tf.less(P, 0),
                            P,
                            tf.where(
                                tf.greater(P, Pm),
                                tf.ones_like(P) * Pm,
                                tf.zeros_like(P),
                            ),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_P = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version == 5:
                        loss_maxmin = tf.where(
                            tf.less(P, 0),
                            tf.math.pow(P, 3),
                            tf.where(
                                tf.greater(P, Pm),
                                P - tf.ones_like(P) * Pm,
                                tf.zeros_like(P),
                            ),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_P = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version == 4:
                        loss_maxmin = tf.where(
                            tf.less(P, 0),
                            P,
                            tf.where(tf.greater(P, Pm), P - Pm, tf.zeros_like(P)),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_P = loss_derivative_abs + loss_maxmin_abs

                    elif loss_version <= 3:
                        loss_P = loss_derivative
                    loss_pde.append(loss_P)

                elif o == "S":
                    loss_derivative = dSdt * V - (V * rS + (f_in * inlet.S - f_out * S))
                    loss_S = 0
                    if loss_version >= 6:
                        loss_maxmin = tf.where(
                            tf.less(S, 0),
                            S,
                            tf.where(
                                tf.greater(S, tf.ones_like(S) * initial_state.S[0]),
                                tf.ones_like(S) * initial_state.S[0],
                                tf.zeros_like(S),
                            ),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_S = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version == 5:
                        loss_maxmin = tf.where(
                            tf.less(S, 0),
                            tf.math.pow(S, 3),
                            tf.where(
                                tf.greater(S, tf.ones_like(S) * initial_state.S[0]),
                                S - initial_state.S[0],
                                tf.zeros_like(S),
                            ),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_S = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version == 4:
                        loss_maxmin = tf.where(
                            tf.less(S, 0),
                            S,
                            tf.where(
                                tf.greater(S, initial_state.S[0]),
                                S - initial_state.S[0],
                                tf.zeros_like(S),
                            ),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_S = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version <= 3:
                        loss_S = loss_derivative
                    loss_pde.append(loss_S)

                elif o == "V":
                    dVdt_calc = f_in - f_out
                    loss_derivative = dVdt - dVdt_calc
                    loss_V = 0
                    if loss_version >= 6:
                        loss_maxmin = tf.where(
                            tf.less(V, 0),
                            V,
                            tf.zeros_like(V),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_V = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version == 5:
                        loss_maxmin = tf.where(
                            tf.less(V, 0),
                            tf.math.pow(V, 3),
                            tf.zeros_like(V),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_V = loss_derivative_abs + loss_maxmin_abs
                    if loss_version == 4:
                        loss_maxmin = tf.where(
                            tf.less(V, 0),
                            V,
                            tf.zeros_like(V),
                        )
                        loss_derivative_abs = tf.abs(loss_derivative)
                        loss_maxmin_abs = tf.abs(loss_maxmin)
                        loss_V = loss_derivative_abs + loss_maxmin_abs
                    elif loss_version <= 3:
                        loss_V = loss_derivative

                    loss_pde.append(loss_V)

            return loss_pde

        return ode_system
