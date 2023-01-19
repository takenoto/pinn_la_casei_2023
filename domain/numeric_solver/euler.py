import numpy as np
import matplotlib.pyplot as plt

from domain.optimization.non_dim_scaler import NonDimScaler
from domain.numeric_solver.numeric_solver_model_results import NumericSolverModelResults
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.params.solver_params import SolverParams
from domain.reactor.cstr_state import CSTRState


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
        name
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
                volume=V * scaler.V,
            )

            non_dim_rX = (scaler.t_not_tensor / scaler.X_not_tensor)*(
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

            dX_dt = non_dim_rX + (scaler.t_not_tensor / scaler.X_not_tensor) * (
                1 / (V * scaler.V_not_tensor)
            )*(f_in * inlet.X - f_out * X * scaler.X_not_tensor) / (
                V * scaler.V_not_tensor
            )
            dP_dt = non_dim_rP + (scaler.t_not_tensor / scaler.P_not_tensor) * (
                1 / (V * scaler.V_not_tensor)
            )*(f_in * inlet.P - f_out * P * scaler.P_not_tensor) / (
                V * scaler.V_not_tensor
            )
            dS_dt = non_dim_rS + (scaler.t_not_tensor / scaler.S_not_tensor) * (
                1 / (V * scaler.V_not_tensor)
            )*(f_in * inlet.S - f_out * S * scaler.S_not_tensor) / (
                V * scaler.V_not_tensor
            )

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
            non_dim_scaler=scaler
        )
