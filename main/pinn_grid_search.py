import tensorflow as tf
import numpy as np

from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.params.solver_params import SolverParams, SolverLBFGSParams
from domain.run_reactor.plot_params import PlotParams
from domain.optimization.ode_system_caller import RunReactorSystemCaller
from domain.optimization.grid_search import grid_search
from domain.optimization.non_dim_scaler import NonDimScaler
from domain.reactor.cstr_state import CSTRState
from domain.flow.concentration_flow import ConcentrationFlow

from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer



def run_pinn_grid_search(
    solver_params_list=None,
    eq_params:Altiok2006Params=None,
    process_params: ProcessParams = None,
    initial_state: CSTRState = None,
    plot_params: PlotParams = None,
    f_out_value_calc=None,
):
    """
    f_out_value_calc --> f_out_value_calc(max_reactor_volume, f_in_v, volume)
    """
    assert f_out_value_calc is not None, "f_out_value_calc is necessary"

    if solver_params_list is None:

        # Tudo que não explicitamente setado aqui será considerado = 1
        scalers_to_try = {
            "default": {"t": 1},
            "case 0": {"t": process_params.t_final},
            "case 1": {
                "t": 1
                / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So))
            },
            "case 2": {
                "t": eq_params.alpha
                * eq_params.So
                * (eq_params.K_S + eq_params.So)
                / eq_params.mu_max
            },
            "case 3": {
                "t": (1 / eq_params.Y_PS)
                * eq_params.alpha
                * (eq_params.K_S + eq_params.So)
                / eq_params.mu_max
            },
            "case 4": {
                "t": process_params.max_reactor_volume / process_params.inlet.volume
                if process_params.inlet.volume > 0
                else 1
            },
        }
        # ---------------------------------------
        # ---------------------------------------

        def get_thing_for_key(scaler_key, thing_key, default=np.array([1])):
            return np.array(scalers_to_try[scaler_key].get(thing_key, default)).item()

        solver_params_list = [
            SolverParams(
                name=f'pinn {scaler_key}:t_s={"{0:.2f}".format(get_thing_for_key(scaler_key, "t"))}',
                num_domain=num_domain,
                num_boundary=10,
                num_test=1000,
                adam_epochs=adam_epochs,
                adam_display_every=3000,
                adam_lr=0.0001,
                l_bfgs=l_bfgs,
                layer_size=layer_size,
                activation="tanh",
                initializer="Glorot uniform",
                loss_weights=[X_weight, P_weight, S_weight, V_weight],
                non_dim_scaler=NonDimScaler(
                    X=get_thing_for_key(scaler_key, 'X'),
                    P=get_thing_for_key(scaler_key, 'P'),
                    S=get_thing_for_key(scaler_key, 'S'),
                    V=get_thing_for_key(scaler_key, 'V'),
                    t=get_thing_for_key(scaler_key, 't'),
                ),
            )
            for num_domain in [600]
            for adam_epochs in [300, 1200]#4000]  # 14500]
            for layer_size in [
                # # Muito espalhadas
                # [1] + [8] * 22 + [4],
                # [1] + [4] * 40 + [4],
                # # Muito concentradas
                # [1] + [140] * 2 + [4],
                # [1] + [320] * 1 + [4],
                # Equilibradas
                # [1] + [22] * 3 + [4],
                [1] + [20]*2 + [4]
                # PRINCIPAL -->>>>>>>>  # [1] + [36] * 4 + [4],
                # [1] + [80] * 5 + [4],
            ]
            for l_bfgs in [
                SolverLBFGSParams(do_pre_optimization=True, do_post_optimization=False),
            ]
            # Basicamente um teste com adimensionalização e um sem
            for scaler_key in scalers_to_try
            for X_weight in [1]
            for P_weight in [1]
            for S_weight in [1]
            for V_weight in [1]
        ]

    # ---------------------------------------

    # Create a default initial state if it is not given
    initial_state = (
        initial_state
        if initial_state
        else CSTRState(
            volume=np.array([4]),
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )
    )

    if process_params is None:
        # If the inlet flow is not specified, it is assumed that it is 0 L/h
        inlet = ConcentrationFlow(
            volume=0,
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )
        process_params = ProcessParams(max_reactor_volume=5, inlet=inlet, t_final=16)

    # Se plot params for nulo cria um padrão
    plot_params = plot_params if plot_params else PlotParams(force_y_lim=True)

    # Define um caller para os parâmetros atuais (só necessita do solver depois)
    run_reactor_system_caller = RunReactorSystemCaller(
        eq_params=eq_params,
        process_params=process_params,
        initial_state=initial_state,
        f_out_value_calc=f_out_value_calc,
    )

    return grid_search(
        pinn_system_caller=run_reactor_system_caller,
        solver_params_list=solver_params_list,
    )
