# 2023-12-08 mudei h_p_calc, estava fazendo a avaliação greater_equal pro X mds

# import tensorflow as tf
from deepxde.backend import tf
import deepxde as dde

from domain.params.solver_params import SolverParams
from domain.params.process_params import ProcessParams
from domain.params.altiok_2006_params import Altiok2006Params
from domain.reactions_ode_system_preparers.losses import picker

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

            t_nondim = x[
                :, inputSimulationType.t_index : inputSimulationType.t_index + 1
            ]
            X_nondim = 0
            P_nondim = 0
            S_nondim = 0
            V_nondim = 1
            dX_dt_nondim = 0
            dP_dt_nondim = 0
            dS_dt_nondim = 0
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
                "t": t_nondim,
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
            t = N["t"]
            X = N["X"]
            P = N["P"]
            S = N["S"]
            V = N["V"]
            dXdt = N["dXdt"]
            dPdt = N["dPdt"]
            dSdt = N["dSdt"]
            dVdt = N["dVdt"]

            # Derivadas segundas:
            dXdt_2 = dde.grad.hessian(X, x, j=inputSimulationType.t_index, grad_y=dXdt)
            dPdt_2 = dde.grad.hessian(P, x, j=inputSimulationType.t_index, grad_y=dPdt)
            dSdt_2 = dde.grad.hessian(S, x, j=inputSimulationType.t_index, grad_y=dSdt)
            dVdt_2 = dde.grad.hessian(V, x, j=inputSimulationType.t_index, grad_y=dVdt)

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
            # Se fizer um tf.where tendo como condicional o próprio X ou
            # o valor da expressão não ser NaN, não presta, não adianta.

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

            if (
                outputSimulationType.X
                and outputSimulationType.P
                and outputSimulationType.S
            ):
                rX = (X * mu_max * S / (K_S + S)) * f_x_calc_func() * h_p_calc_func()
                rP = alpha * rX + beta * X
                rS = -(1 / Y_PS) * rP - ms * X

                # -----------------------
                # Calculating the loss
                # Procura cada variável registrada como de saída e
                # adiciona o cálculo da sua função como componente da loss

            # ---------------------------------------------
            # Calculate loss for each variables
            # ---------------------------------------------
            args = (
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
                dXdt_2,
                dPdt_2,
                dSdt_2,
                dVdt_2,
            )

            for o in outputSimulationType.order:
                # Pode dar NaN quando predizer valores abaixo de 0
                # Então evite divisões!!!! Por isso o V vem multiplicando no fim...
                # Mas isso terminou sendo uma faca de dois gumes
                loss_version = solver_params.get_loss_version_for_type(o)
                loss_o = picker.loss_picker(loss_version=loss_version, o=o, args=args)
                loss_pde.append(loss_o)

            # --------------------------

            return loss_pde
            # --------------------------

        # ------------------------------

        return ode_system
