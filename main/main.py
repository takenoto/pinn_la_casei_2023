# python -m main.main
from domain.params.altiok_2006_params import get_altiok2006_params
from domain.reactor.cstr_state import CSTRState
from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer
from domain.params.process_params import ProcessParams
from domain.flow.concentration_flow import ConcentrationFlow
from main.run_model import run_model

import deepxde
# For obtaining fully reproducible results
deepxde.config.set_random_seed(0)

def main():

    run_batch = True
    
    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)

    # Melhores resultados para o batch model:
    if run_batch:

        eq_params = altiok_models_to_run[0]

        process_params = ProcessParams(
            max_reactor_volume=5,
            inlet=ConcentrationFlow(
                volume=0.0,
                X=eq_params.Xo,
                P=eq_params.Po,
                S=eq_params.So,
            ),
            t_final=16,
        )

        batch_results = run_model(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params,
            # initial_state=initial_state,
            # A saída sempre é 0 (reator fechado)
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
        )
        pinn_results, best_pinn_index, num_results, best_num_index = batch_results[2]
        print("Concluído!!!!!!")
        print(pinn_results)
        print(best_pinn_index)
        print(num_results)
        print(best_num_index)


if __name__ == "__main__":
    main()
