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


# Parâmetros default
_adam_epochs_default = 1000  # [1000, 1000]#,100, 500, 1100]#, 1200]#4000]  # 14500]
_layer_size_default = [1] + [12] * 2 + [4]


def run_pinn_grid_search(
    solver_params_list=None,
    eq_params: Altiok2006Params = None,
    process_params: ProcessParams = None,
    initial_state: CSTRState = None,
    plot_params: PlotParams = None,
    f_out_value_calc=None,
    cases_to_try=None,
):
    """
    f_out_value_calc --> f_out_value_calc(max_reactor_volume, f_in_v, volume)
    """

    assert f_out_value_calc is not None, "f_out_value_calc is necessary"
    assert cases_to_try is not None, "cases_to_try is necessary"

    if solver_params_list is None:

        # ---------------------------------------
        # ---------------------------------------

        def get_thing_for_key(case_key, thing_key, default=np.array([1])):
            if thing_key not in ["layer_size", "lbfgs_pre", "lbfgs_post"]:
                return np.array(cases_to_try[case_key].get(thing_key, default)).item()
            else:
                return cases_to_try[case_key].get(thing_key, default)

        solver_params_list = [
            SolverParams(
                name=f"{case_key}",
                num_domain=get_thing_for_key(case_key, "num_domain", default=600),
                num_boundary=10,
                num_test=get_thing_for_key(case_key, "num_test", default=1000),
                adam_epochs=get_thing_for_key(
                    case_key, "adam_epochs", default=_adam_epochs_default
                ),
                adam_display_every=30000,
                adam_lr=0.0001,
                l_bfgs=SolverLBFGSParams(
                    do_pre_optimization=get_thing_for_key(
                        case_key, "lbfgs_pre", default=True
                    ),
                    do_post_optimization=get_thing_for_key(
                        case_key, "lbfgs_post", default=False
                    ),
                ),
                layer_size=get_thing_for_key(
                    case_key, "layer_size", default=_layer_size_default
                ),
                activation=get_thing_for_key(case_key, "activation", default="tanh"),
                initializer=get_thing_for_key(case_key, "initializer", default="Glorot uniform"),
                loss_weights=[
                    get_thing_for_key(case_key, "w_X", 1), 
                    get_thing_for_key(case_key, "w_P", 1), 
                    get_thing_for_key(case_key, "w_S", 1), 
                    get_thing_for_key(case_key, "w_V", 1)],
                non_dim_scaler=NonDimScaler(
                    X=get_thing_for_key(case_key, "X_s"),
                    P=get_thing_for_key(case_key, "P_s"),
                    S=get_thing_for_key(case_key, "S_s"),
                    V=get_thing_for_key(case_key, "V_s"),
                    t=get_thing_for_key(case_key, "t_s"),
                ),
            )
            # Basicamente um teste com adimensionalização e um sem
            for case_key in cases_to_try
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
