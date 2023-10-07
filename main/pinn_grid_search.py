import numpy as np

from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.params.solver_params import (
    SolverParams,
    SolverLBFGSParams,
    SystemSimulationType,
)
from domain.run_reactor.plot_params import PlotParams
from domain.optimization.ode_system_caller import RunReactorSystemCaller
from domain.optimization.grid_search import grid_search
from domain.reactor.reactor_state import ReactorState
from domain.flow.concentration_flow import ConcentrationFlow


# Parâmetros default
_adam_epochs_default = 1000  # [1000, 1000]#,100, 500, 1100]#, 1200]#4000]  # 14500]
_layer_size_default = [1] + [12] * 2 + [4]


def run_pinn_grid_search(
    solver_params_list=None,
    eq_params: Altiok2006Params = None,
    process_params: ProcessParams = None,
    initial_state: ReactorState = None,
    plot_params: PlotParams = None,
    f_out_value_calc=None,
    cases_to_try=None,
    save_caller=None,
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
            if thing_key not in [
                "layer_size",
                "lbfgs_pre",
                "lbfgs_post",
                "output_variables",
                "input_variables",
                "isplot",
                "train_input_range",
                "loss_weights"
            ]:
                return np.array(cases_to_try[case_key].get(thing_key, default)).item()
            else:
                return cases_to_try[case_key].get(thing_key, default)

        solver_params_list = [
            SolverParams(
                name=f"{case_key}",
                num_domain=get_thing_for_key(case_key, "num_domain", default=600),
                num_boundary=get_thing_for_key(case_key, "num_boundary", default=10),
                num_init=get_thing_for_key(case_key, "num_init", default=10),
                num_test=get_thing_for_key(case_key, "num_test", default=1000),
                adam_epochs=get_thing_for_key(case_key, "adam_epochs", default=None),
                adam_display_every=200,  # 2000, #5000,#30000,
                sgd_epochs=get_thing_for_key(case_key, "sgd_epochs", default=None),
                adam_lr=get_thing_for_key(case_key, "LR", default=0.0001),
                l_bfgs=SolverLBFGSParams(
                    do_pre_optimization=get_thing_for_key(
                        case_key, "lbfgs_pre", default=0
                    ),
                    do_post_optimization=get_thing_for_key(
                        case_key, "lbfgs_post", default=1
                    ),
                ),
                layer_size=get_thing_for_key(
                    case_key, "layer_size", default=_layer_size_default
                ),
                activation=get_thing_for_key(case_key, "activation", default="tanh"),
                initializer=get_thing_for_key(
                    case_key, "initializer", default="Glorot uniform"
                ),
                loss_weights=get_thing_for_key(case_key, "loss_weights", None),
                input_non_dim_scaler=get_thing_for_key(case_key, "input_scaler", default=None),
                output_non_dim_scaler=get_thing_for_key(case_key, "output_scaler", default=None),
                mini_batch=get_thing_for_key(case_key, "mini_batch", default=None),
                hyperfolder=get_thing_for_key(case_key, "hyperfolder", default=None),
                isplot=get_thing_for_key(case_key, "isplot", default=False),
                outputSimulationType=SystemSimulationType(
                    get_thing_for_key(
                        case_key, "output_variables", default=["X", "P", "S", "V"]
                    )
                ),
                inputSimulationType=SystemSimulationType(
                    get_thing_for_key(case_key, "input_variables", default=["t"])
                ),
                loss_version=get_thing_for_key(case_key, "loss_version", default=2),
                custom_loss_version=get_thing_for_key(
                    case_key, "custom_loss_version", default={}
                ),
                train_distribution=get_thing_for_key(
                    case_key, "train_distribution", default="Hammersley"
                ),
                save_caller=get_thing_for_key(case_key, "save_caller", default=None),
                train_input_range=get_thing_for_key(
                    case_key, "train_input_range", default=None
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
        else ReactorState(
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
