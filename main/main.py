# python -m main.main

import numpy as np
import deepxde


from domain.params.altiok_2006_params import get_altiok2006_params
from domain.reactor.cstr_state import CSTRState
from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer
from domain.params.process_params import ProcessParams
from domain.flow.concentration_flow import ConcentrationFlow
from main.pinn_grid_search import run_pinn_grid_search
from main.numerical_methods import run_numerical_methods
from main.plot_xpsv import plot_xpsv, multiplot_xpsv, XPSVPlotArg

from main.plotting.simple_color_bar import plot_simple_color_bar
from main.plotting.surface_3d import (
    plot_3d_surface,
    plot_3d_surface_ts_step_error,
    plot_3d_lines,
    PlotPINN3DArg,
)


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


def main():

    run_batch = True

    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)

    # Melhores resultados para o batch model:
    if run_batch:

        eq_params = altiok_models_to_run[0]

        initial_state = CSTRState(
            volume=np.array([4]),
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )

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
        pinn_results, best_pinn_test_index, best_pin_test_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
        )

        plot_3d_lines(
            pinns = [
                PlotPINN3DArg(
                    adam_epochs=p.solver_params.adam_epochs,
                    best_loss_test=p.best_loss_test,
                    t_not_tensor=p.solver_params.non_dim_scaler.t_not_tensor,
                )
                for p in pinn_results
            ])


        multiplot_xpsv(
            title="Concentrations over time for different methods",
            y_label="g/L",
            x_label="time (h)",
            t=[n.t for n in num_results] + [pinn.t for pinn in pinn_results],
            X=[n.X for n in num_results] + [pinn.X for pinn in pinn_results],
            P=None,
            S=None,
            V=None,
            scaler=[n.non_dim_scaler for n in num_results]
            + [pinn.solver_params.non_dim_scaler for pinn in pinn_results],
            suffix=[n.model_name for n in num_results]
            + [pinn.model_name for pinn in pinn_results],
            plot_args=[
                XPSVPlotArg(
                    ls=":",
                    color=num_colors[i % len(num_colors)],
                    linewidth=7.0,
                    alpha=0.25,
                )
                for i in range(len(num_results))
            ]
            + [
                XPSVPlotArg(
                    ls="-",
                    color=pinn_colors[i % len(pinn_colors)],
                    linewidth=4,
                    alpha=1,
                )
                for i in range(len(pinn_results))
            ],
        )

        # FIXME dá um erro estranho nas configs que tá, mas em outras deu certo?????????
        # Dá esse erro quando o adam é só 1 valor: 300
        # É só nesse último gráfico
        plot_3d_surface_ts_step_error(
            pinns=[
                PlotPINN3DArg(
                    adam_epochs=p.solver_params.adam_epochs,
                    best_loss_test=p.best_loss_test,
                    t_not_tensor=p.solver_params.non_dim_scaler.t_not_tensor,
                )
                for p in pinn_results
            ]
        )

        print("--------------------")
        print("!!!!!!FINISED!!!!!!")
        print("--------------------")


if __name__ == "__main__":
    main()
