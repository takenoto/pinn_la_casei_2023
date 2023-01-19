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
from main.plot_xpsv import plot_xpsv, multiplot_xpsv


# For obtaining fully reproducible results
deepxde.config.set_random_seed(0)


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

        # plot_xpsv(
        #     num_results[0].t,
        #     num_results[0].X,
        #     num_results[0].P,
        #     num_results[0].S,
        #     num_results[0].V,
        #     scaler=num_results[0].scaler,
        # )

        # plot_xpsv(
        #     num_result.t,
        #     num_result.X,
        #     num_result.P,
        #     num_result.S,
        #     num_result.V,
        #     scaler=num_result.scaler,
        # )
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

        
        multiplot_xpsv(
            t=[n.t for n in num_results] + [pinn.t for pinn in pinn_results],
            X=[n.X for n in num_results] + [pinn.X for pinn in pinn_results],
            P=None,
            S=None,
            V=None,
            scaler=[n.non_dim_scaler for n in num_results] + [pinn.solver_params.non_dim_scaler for pinn in pinn_results],
            suffix=[n.model_name for n in num_results] + [pinn.model_name for pinn in pinn_results],
        )
        # plot_xpsv(
        #     pinn.t, pinn.X, pinn.P, pinn.S, pinn.V, scaler=pinn.solver_params.scaler
        # )

        print("Concluído!!!!!!")
        print(pinn_results)
        print(best_pinn_index)
        print(num_results)
        print(best_num_index)


if __name__ == "__main__":
    main()
