import numpy as np

from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.params.solver_params import SolverParams
from domain.run_reactor.plot_params import PlotParams
from domain.optimization.ode_system_caller import RunReactorSystemCaller
from domain.optimization.grid_search import grid_search
from domain.reactor.cstr_state import CSTRState
from domain.flow.concentration_flow import ConcentrationFlow

from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer


def run_model(
    solver_params_list=None,
    eq_params=None,
    process_params: ProcessParams = None,
    initial_state: CSTRState = None,
    plot_params: PlotParams = None,
    f_out_value_calc=None,
):
    """
    f_out_value_calc --> f_out_value_calc(max_reactor_volume, f_in_v, volume)
    """
    
    assert f_out_value_calc is not None, "f_out_value_calc is necessary"

    print("Starting Batch Model")

    if solver_params_list is None:

        # Preenche com um único item caso nenhuma opção tenha sido passada

        solver_params = SolverParams(
            num_domain=600,
            num_boundary=10,
            num_test=1000,
            adam_epochs=7500,
            adam_display_every=300,
            adam_lr=0.0001,
            layer_size=[1] + [36] * 4 + [4],
            activation="tanh",
            initializer="Glorot uniform",
        )

        solver_params_list = [solver_params]

    # ---------------------------------------

    # Create a default initial state if it is not given
    initial_state = (
        initial_state
        if initial_state
        else CSTRState(
            volume=np.array([5]),  # L
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )
    )

    if process_params is None:
        # Se não for especificado, assume que entrada é 0
        inlet = ConcentrationFlow(
            volume=0,
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )
        process_params = ProcessParams(max_reactor_volume=5, inlet=inlet, t_final=16)

    plot_params = plot_params if plot_params else PlotParams(force_y_lim=True)

    ode_system_preparer = ODEPreparer(solver_params, eq_params, process_params, f_out_value_calc)

    # Define um caller para os parâmetros atuais (só necessita do solver depois)
    run_reactor_system_caller = RunReactorSystemCaller(
        ode_system_preparer=ode_system_preparer,
        eq_params=eq_params,
        process_params=process_params,
        initial_state=initial_state,
        plot_params=plot_params,  # Se plot params for nulo cria um padrão
        f_out_value_calc=f_out_value_calc,
    )

    return grid_search(
        pinn_system_caller=run_reactor_system_caller,
        solver_params_list=solver_params_list,
    )
