import numpy as np

from domain.optimization.non_dim_scaler import NonDimScaler
from domain.numeric_solver.numeric_solver_model_results import NumericSolverModelResults
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.reactor.reactor_state import ReactorState

new_version = True


class EulerMethod:
    def __init__(self):
        pass

    def solve(
        self,
        initial_state: ReactorState,
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
        t_space = np.linspace(
            start=0,
            stop=process_params.t_final,
            num=t_discretization_points,
        )

        dt = t_space[1]

        X_array = np.ones(len(t_space)) * N0["X"]
        P_array = np.ones(len(t_space)) * N0["P"]
        S_array = np.ones(len(t_space)) * N0["S"]
        V_array = np.ones(len(t_space)) * N0["V"]

        dX_dt_normal_array = np.zeros(len(t_space))
        dP_dt_normal_array = np.zeros(len(t_space))
        dS_dt_normal_array = np.zeros(len(t_space))
        dV_dt_normal_array = np.zeros(len(t_space))
        dX_dt_normal_array[0] = None
        dP_dt_normal_array[0] = None
        dS_dt_normal_array[0] = None
        dV_dt_normal_array[0] = None
        dN_dt_2 = {N: np.zeros(len(t_space)) for N in ["X", "P", "S", "V"]}

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
            N_previous = {
                "X": X_array[t - 1],
                "P": P_array[t - 1],
                "S": S_array[t - 1],
                "V": V_array[t - 1],
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

            X_array[t] = X
            P_array[t] = P
            S_array[t] = S
            V_array[t] = V

            dX_dt_normal_array[t] = dX_dt
            dP_dt_normal_array[t] = dP_dt
            dS_dt_normal_array[t] = dS_dt
            dV_dt_normal_array[t] = dV_dt

        # Seta a derivada em 0 como igual à em 1 pra não bagunçar muito o gráfico
        dX_dt_normal_array[0] = (X_array[1] - X_array[0]) / dt
        dP_dt_normal_array[0] = (P_array[1] - P_array[0]) / dt
        dS_dt_normal_array[0] = (S_array[1] - S_array[0]) / dt
        dV_dt_normal_array[0] = (V_array[1] - V_array[0]) / dt

        # ------------------------------
        # Calculate 2 order derivatives
        # ------------------------------
        # ref: https://personal.math.ubc.ca/~jfeng/CHBE553/Example7/Formulae.pdf
        # A função "f" são os próprios valores de XPSV calculados, posso usar eles!!
        XPSV = {
            "X": X_array,
            "P": P_array,
            "S": S_array,
            "V": V_array,
        }
        
        for N_key in XPSV:
            # First point
            N = XPSV[N_key]
            dN_dt_2[N_key][0] = (2*N[0] - 5*N[1] + 4*N[2] - N[3])/(dt**2)
            # Center points:
            for t in range(1, len(t_space)-1):
                dN_dt_2[N_key][t] = (N[t+1] - 2*N[t]  + N[t-1])/(dt**2)
            # Backward
            dN_dt_2[N_key][-1] = (2*N[-1] - 5*N[-2] + 4*N[-3] - N[-4])/(dt**2)
                

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
            dX_dt=dX_dt_normal_array,
            dP_dt=dP_dt_normal_array,
            dS_dt=dS_dt_normal_array,
            dV_dt=dV_dt_normal_array,
            dX_dt_2=dN_dt_2["X"],
            dP_dt_2=dN_dt_2["P"],
            dS_dt_2=dN_dt_2["S"],
            dV_dt_2=dN_dt_2["V"],
        )
