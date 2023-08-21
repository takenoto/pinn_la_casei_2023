import numpy as np
import matplotlib.pyplot as plt

from domain.optimization.non_dim_scaler import NonDimScaler
from domain.numeric_solver.numeric_solver_model_results import NumericSolverModelResults
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.params.solver_params import SolverParams
from domain.reactor.cstr_state import CSTRState

new_version = True


class EulerMethod:
    def __init__(self):
        pass

    def solve(
        self,
        initial_state: CSTRState,
        eq_params: Altiok2006Params,
        process_params: ProcessParams,
        f_out_value_calc,
        non_dim_scaler: NonDimScaler,
        t_discretization_points,
        name,
    ):
        """
        Returns the result of the simulation
        t, X, P, S, V, dt
        """

        # Variables at t = 0
        N0 = {
            "X": initial_state.X[0],
            "P": initial_state.P[0],
            "S": initial_state.S[0],
            "V": initial_state.volume[0],
        }

        scaler = non_dim_scaler
        t_space_nondim = np.linspace(
            start=0,
            stop=non_dim_scaler.toNondim({"t": process_params.t_final}, "t"),
            num=t_discretization_points,
        )
        dt = non_dim_scaler.fromNondim({"dt": t_space_nondim[1]}, "dt")

        X_nondim_array = np.ones(len(t_space_nondim)) * non_dim_scaler.toNondim(N0, "X")
        P_nondim_array = np.ones(len(t_space_nondim)) * non_dim_scaler.toNondim(N0, "P")
        S_nondim_array = np.ones(len(t_space_nondim)) * non_dim_scaler.toNondim(N0, "S")
        V_nondim_array = np.ones(len(t_space_nondim)) * non_dim_scaler.toNondim(N0, "V")

        inlet = process_params.inlet
        f_in = inlet.volume

        # PARAMETERS
        mu_max = eq_params.mu_max
        K_S = eq_params.K_S
        alpha = eq_params.alpha
        beta = eq_params.beta
        Y_PS = eq_params.Y_PS
        ms = eq_params.ms
        f = eq_params.f
        h = eq_params.h
        Pm = eq_params.Pm
        Xm = eq_params.Xm

        for t in range(1, len(t_space_nondim)):
            # Declara valores do ponto imediatamente anterior para usar
            N_previous = {
                "X": X_nondim_array[t - 1],
                "P": P_nondim_array[t - 1],
                "S": S_nondim_array[t - 1],
                "V": V_nondim_array[t - 1],
            }
            # Nondimensionalization:
            X = non_dim_scaler.fromNondim(N_previous, "X")
            P = non_dim_scaler.fromNondim(N_previous, "P")
            S = non_dim_scaler.fromNondim(N_previous, "S")
            V = non_dim_scaler.fromNondim(N_previous, "V")

            # Calculating previous volume
            # V_previous = non_dim_scaler.fromNondim(V_nondim_array[t - 1], "V")
            V_previous = V

            # Calculating f_out at current point in time
            f_out = f_out_value_calc(
                max_reactor_volume=process_params.max_reactor_volume,
                f_in_v=f_in,
                volume=V,
            )

            dV_dt = f_in - f_out
            V = V + (dV_dt * dt)

            if X >= Xm:
                X = (Xm) * 0.9999999999999999999999999999999
            if P >= Pm:
                P = (Pm) * 0.9999999999999999999999999999999
            rX = (
                (X * mu_max * S / (K_S + S))
                * pow(1 - (X / Xm), f)
                * pow(1 - (P / Pm), h)
            )

            rP = alpha * rX + beta * X

            rS = -(1 / Y_PS) * rP - ms * X

            dX_dt = rX + (f_in * inlet.X - f_out * X) / (V)

            dP_dt = rP + (f_in * inlet.P - f_out * P) / (V)

            dS_dt = rS + (f_in * inlet.S - f_out * S) / (V)

            X = X * V_previous / V + (dX_dt * dt)
            P = P * V_previous / V + (dP_dt * dt)
            S = S * V_previous / V + (dS_dt * dt)

            N = {
                "X": X,
                "P": P,
                "S": S,
                "V": V,
            }

            X_nondim_array[t] = non_dim_scaler.toNondim(N, "X")
            P_nondim_array[t] = non_dim_scaler.toNondim(N, "P")
            S_nondim_array[t] = non_dim_scaler.toNondim(N, "S")
            V_nondim_array[t] = non_dim_scaler.toNondim(N, "V")

        return NumericSolverModelResults(
            model=self,
            model_name=name if name else "euler",
            X=X_nondim_array,
            P=P_nondim_array,
            S=S_nondim_array,
            V=V_nondim_array,
            t=t_space_nondim,
            dt=dt,
            non_dim_scaler=scaler,
        )


class EulerMethodOLD:
    def __init__(self):
        pass

    def solve(
        self,
        initial_state: CSTRState,
        eq_params: Altiok2006Params,
        process_params: ProcessParams,
        f_out_value_calc,
        non_dim_scaler: NonDimScaler,
        t_discretization_points,
        name,
    ):
        """
        Returns the result of the simulation
        t, X, P, S, V, dt
        """
        scaler = non_dim_scaler
        t_space = np.linspace(
            start=0,
            stop=process_params.t_final / scaler.t_not_tensor,
            num=t_discretization_points,
        )
        dt = t_space[1]

        X_array = np.ones(len(t_space)) * initial_state.X[0] / scaler.X_not_tensor
        P_array = np.ones(len(t_space)) * initial_state.P[0] / scaler.P_not_tensor
        S_array = np.ones(len(t_space)) * initial_state.S[0] / scaler.S_not_tensor
        V_array = np.ones(len(t_space)) * initial_state.volume[0] / scaler.V_not_tensor

        inlet = process_params.inlet
        f_in = inlet.volume

        # PARAMETERS
        mu_max = eq_params.mu_max
        K_S = eq_params.K_S
        alpha = eq_params.alpha
        beta = eq_params.beta
        Y_PS = eq_params.Y_PS
        ms = eq_params.ms
        f = eq_params.f
        h = eq_params.h
        Pm = eq_params.Pm
        Xm = eq_params.Xm

        for t in range(1, len(t_space)):
            # Declara valores do ponto imediatamente anterior para usar
            X = X_array[t - 1]
            P = P_array[t - 1]
            S = S_array[t - 1]
            V = V_array[t - 1]

            f_out = f_out_value_calc(
                max_reactor_volume=process_params.max_reactor_volume,
                f_in_v=f_in,
                volume=V * scaler.V_not_tensor,
            )

            # FIXME código 10/08/2023
            # Colocar o X e P na limiar caso estejam quebrando, só pra evitar o problema numérico mesmo.
            if X * scaler.X_not_tensor >= Xm:
                X = (Xm / scaler.X_not_tensor) * 0.9999999999999999999999999999999
            if P * scaler.P_not_tensor >= Pm:
                P = (Pm / scaler.P_not_tensor) * 0.9999999999999999999999999999999

            non_dim_rX = (scaler.t_not_tensor / scaler.X_not_tensor) * (
                (
                    X
                    * scaler.X_not_tensor
                    * mu_max
                    * S
                    * scaler.S_not_tensor
                    / (K_S + S * scaler.S_not_tensor)
                )
                * pow(1 - X * scaler.X_not_tensor / Xm, f)
                * pow(1 - P * scaler.P_not_tensor / Pm, h)
            )

            non_dim_rP = (scaler.t_not_tensor / scaler.P_not_tensor) * (
                alpha * (scaler.X_not_tensor / scaler.t_not_tensor) * non_dim_rX
                + beta * X * scaler.X_not_tensor
            )
            non_dim_rS = (scaler.t_not_tensor / scaler.S_not_tensor) * (
                -(1 / Y_PS) * non_dim_rP * (scaler.P_not_tensor / scaler.t_not_tensor)
                - ms * X * scaler.X_not_tensor
            )

            # FIXME removi o divisor do V no 1º termo, estava duplicado, tornando as taxas erradas.
            dX_dt = non_dim_rX + (scaler.t_not_tensor / scaler.X_not_tensor) * (
                1 if new_version else 1 / (V * scaler.V_not_tensor)
            ) * (f_in * inlet.X - f_out * X * scaler.X_not_tensor) / (
                V * scaler.V_not_tensor
            )
            dP_dt = non_dim_rP + (scaler.t_not_tensor / scaler.P_not_tensor) * (
                1 if new_version else 1 / (V * scaler.V_not_tensor)
            ) * (f_in * inlet.P - f_out * P * scaler.P_not_tensor) / (
                V * scaler.V_not_tensor
            )
            dS_dt = non_dim_rS + (scaler.t_not_tensor / scaler.S_not_tensor) * (
                1 if new_version else 1 / (V * scaler.V_not_tensor)
            ) * (f_in * inlet.S - f_out * S * scaler.S_not_tensor) / (
                V * scaler.V_not_tensor
            )
            # print(f'fIn/fOut = {f_in}/{f_out}')
            # print(f'fin_S/foutS/rS_nondim = {f_in * inlet.S/( V * scaler.V_not_tensor) }/{f_out * S * scaler.S_not_tensor/( V * scaler.V_not_tensor)}/{non_dim_rS}')
            # print(f'V = {V * scaler.V_not_tensor}')
            # print(f'dSdt = {dS_dt}')

            # dP_dt = non_dim_rP + (
            #     scaler.t_not_tensor
            #     * f_in
            #     * inlet.P
            #     / (V * scaler.V_not_tensor * scaler.P_not_tensor)
            #     - scaler.t_not_tensor * f_out * P
            # ) / (V * scaler.V_not_tensor)

            # dS_dt = non_dim_rS + (
            #     scaler.t_not_tensor
            #     * f_in
            #     * inlet.S
            #     / (V * scaler.V_not_tensor * scaler.S_not_tensor)
            #     - scaler.t_not_tensor * f_out * S
            # ) / (V * scaler.V_not_tensor)

            dV_dt = (scaler.t_not_tensor / scaler.V_not_tensor) * (f_in - f_out)

            # dX_dt = non_dim_rX
            # dP_dt = non_dim_rP
            # dS_dt = non_dim_rS
            X_array[t] = X + dX_dt * dt
            P_array[t] = P + dP_dt * dt
            S_array[t] = S + dS_dt * dt
            V_array[t] = V + dV_dt * dt

        return NumericSolverModelResults(
            model=self,
            model_name=name if name else "euler",
            X=X_array,
            P=P_array,
            S=S_array,
            V=V_array,
            t=t_space,
            dt=dt,
            non_dim_scaler=scaler,
        )
