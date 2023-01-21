# python -m main.main

from timeit import default_timer as timer

import numpy as np
import deepxde


from domain.params.altiok_2006_params import get_altiok2006_params
from domain.reactor.cstr_state import CSTRState
from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer
from domain.params.process_params import ProcessParams
from domain.flow.concentration_flow import ConcentrationFlow
from main.pinn_grid_search import run_pinn_grid_search, _default_cases_to_try
from main.numerical_methods import run_numerical_methods
from main.plot_xpsv import plot_xpsv, multiplot_xpsv, XPSVPlotArg
from main.plotting.plot_pinn_3d_arg import PlotPINN3DArg
from main.plotting.pinn_conversor import get_ts_step_loss_as_xyz
from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults
from domain.params.solver_params import SolverParams
from domain.optimization.non_dim_scaler import NonDimScaler

from main.cases_to_try import *

from main.plotting.simple_color_bar import plot_simple_color_bar
from main.plotting.surface_3d import (
    plot_3d_surface,
    plot_3d_surface_ts_step_error,
    plot_3d_lines,
    PlotPINN3DArg,
)
from main.plotting.plot_lines_error_compare import plot_lines_error_compare


# For obtaining fully reproducible results
deepxde.config.set_random_seed(0)

xp_colors = ["#F2545B"]
"""
Cores apra dados experimentais
"""

pinn_colors = [
    "#7293A0",
    "#C2E812",
    "#ED9B40",
    "#B4869F",
    "#45B69C",
    "#FFE66D",
    "#E78F8E",
    "#861388",
    "#34312D",
]
"""
Lista de cores que representam diversos pinns
Ordem: RGBA
"""

num_colors = [
    "#F39B6D",
    "#F0C987",
]


def plot_compare_pinns_and_num(pinns, nums, eq_params):
    # --------------------------------------
    # PLOTTING
    # --------------------------------------

    # Plot simple color bar with ts, step, loss as x, y, z
    ts, step, loss = get_ts_step_loss_as_xyz(pinns=pinns)
    if False:
        plot_simple_color_bar(
            title="Time scaling factor, number of steps and error",
            x=ts,
            y=step,
            z=loss,
            x_label="t_s",
            y_label="step",
            z_label="loss",
        )

    if False:
        plot_lines_error_compare(pinns=pinns)

    multiplot_xpsv(
        title="Concentrations over time for different methods",
        y_label="g/L",
        x_label="time (h)",
        t=[n.t for n in nums] + [pinn.t for pinn in pinns],
        X=[n.X for n in nums] + [pinn.X for pinn in pinns],
        P=None,
        S=None,
        V=None,
        y_lim=[0, eq_params.So * 1.1],
        err=[0 for n in nums] + [pinn.best_loss_test for pinn in pinns],
        scaler=[n.non_dim_scaler for n in nums]
        + [pinn.solver_params.non_dim_scaler for pinn in pinns],
        suffix=[n.model_name for n in nums] + [pinn.model_name for pinn in pinns],
        plot_args=[
            XPSVPlotArg(
                ls=":",
                color=num_colors[i % len(num_colors)],
                linewidth=7.0,
                alpha=0.25,
            )
            for i in range(len(nums))
        ]
        + [
            XPSVPlotArg(
                ls="-",
                color=pinn_colors[i % len(pinn_colors)],
                linewidth=4,
                alpha=1,
            )
            for i in range(len(pinns))
        ],
    )
    if len(pinns) >= 1:
        plot_3d_surface_ts_step_error(pinns=pinns)


def main():

    run_batch = False

    run_fed_batch = True

    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)
    eq_params = altiok_models_to_run[0]
    initial_state = CSTRState(
        volume=np.array([4]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    if run_fed_batch:
        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.07,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=10,
        )
        # ----------------------------
        # NUMERICAL
        # ----------------------------
        num_results = run_numerical_methods(
            initial_state=initial_state,
            eq_params=eq_params,
            process_params=process_params,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
        )
        pinn_results = []
        # cases_to_try=_default_cases_to_try(eq_params, process_params)
        cases_to_try = cases_to_try_fed_batch_vs(eq_params, process_params)
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases_to_try,
        )
        # Plot
        plot_compare_pinns_and_num(
            pinns=pinn_results, nums=num_results, eq_params=eq_params
        )

    if run_batch:
        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.0,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=10,
        )
        # ----------------------------
        # NUMERICAL
        # ----------------------------
        num_results = run_numerical_methods(
            initial_state=initial_state,
            eq_params=eq_params,
            process_params=process_params,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
        )

        # ----------------------------
        # PINN
        # ----------------------------
        start_time = timer()
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=_default_cases_to_try(eq_params, process_params),
        )
        end_time = timer()
        print(f"elapsed batch reactor pinn grid time = {end_time - start_time} secs")

        # Plot
        plot_compare_pinns_and_num(
            pinns=pinn_results, num=num_results, eq_params=eq_params
        )

    print("--------------------")
    print("!!!!!!FINISED!!!!!!")
    print("--------------------")


if __name__ == "__main__":
    main()
