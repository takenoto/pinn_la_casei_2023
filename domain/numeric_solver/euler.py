import numpy as np
import matplotlib.pyplot as plt

from domain.optimization.non_dim_scaler import NonDimScaler
from domain.numeric_solver.numeric_solver_model_results import NumericSolverModelResults
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.params.solver_params import SolverParams
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
        
        dX_dt_normal_array = np.zeros(len(t_space_nondim)) 
        dP_dt_normal_array = np.zeros(len(t_space_nondim)) 
        dS_dt_normal_array = np.zeros(len(t_space_nondim)) 
        dV_dt_normal_array = np.zeros(len(t_space_nondim)) 
        dX_dt_normal_array[0] = None
        dP_dt_normal_array[0] = None
        dS_dt_normal_array[0] = None
        dV_dt_normal_array[0] = None
        dX_dt_2_normal_array = np.zeros(len(t_space_nondim)) 
        dP_dt_2_normal_array = np.zeros(len(t_space_nondim)) 
        dS_dt_2_normal_array = np.zeros(len(t_space_nondim)) 
        dV_dt_2_normal_array = np.zeros(len(t_space_nondim)) 
        dX_dt_2_normal_array[0] = None
        dP_dt_2_normal_array[0] = None
        dS_dt_2_normal_array[0] = None
        dV_dt_2_normal_array[0] = None

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
                "dXdt": dX_dt,
                "dPdt": dP_dt,
                "dSdt": dS_dt,
                "dVdt": dV_dt,
            }

            X_nondim_array[t] = non_dim_scaler.toNondim(N, "X")
            P_nondim_array[t] = non_dim_scaler.toNondim(N, "P")
            S_nondim_array[t] = non_dim_scaler.toNondim(N, "S")
            V_nondim_array[t] = non_dim_scaler.toNondim(N, "V")
            
            dX_dt_normal_array[t] = dX_dt
            dP_dt_normal_array[t] = dP_dt
            dS_dt_normal_array[t] = dS_dt
            dV_dt_normal_array[t] = dV_dt
                
            
        # Seta a derivada em 0 como igual à em 1 pra não bagunçar muito o gráfico
        dX_dt_normal_array[0] = (X[1] - X[0])/dt
        dP_dt_normal_array[0] = dP_dt_normal_array[1]
        dS_dt_normal_array[0] = dS_dt_normal_array[1]
        dV_dt_normal_array[0] = dV_dt_normal_array[1]
        
        # TODO  
        # ------------------------------
        # Calculate 2 order derivatives
        #------------------------------
        
        # A função "f" são os próprios valores de XPSV calculados, posso usar eles!!!
        # TODO declara XPSV como vetor e itera todos, acho que é mais fácil
        # e não precisa se repetir
        # TODO 1 ponto => único
        # TODO centered pontos intermediários => loop for
        # TODO backward ponto final
        # Centered when 1<t<len => único
        # if(t>1 and t<len(t_space_nondim)):
        #     dX_dt_2_normal_array[t] = 
    

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
            dX_dt=dX_dt_normal_array,
            dP_dt=dP_dt_normal_array,
            dS_dt=dS_dt_normal_array,
            dV_dt=dV_dt_normal_array,
            dX_dt_2=None,
            dP_dt_2=None,
            dS_dt_2=None,
            dV_dt_2=None,
        )
