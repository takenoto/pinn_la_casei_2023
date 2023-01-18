# TODO f_out tb é recebido como uma lista de parâmetros

# solver_params

# o parâmetro de processo deve ser só 1. O objetivo daqui é achar pontos ótimos do solver

# então n vou iterar por exemplo, V, f_in, etc


from domain.params.solver_params import SolverParams

from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults

from domain.optimization.ode_system_caller import RunReactorSystemCaller


def grid_search(
    pinn_system_caller: RunReactorSystemCaller,
    solver_params_list: list,  # SolverParams,
):

    "Receive a list of each kind of parameter and test them"

    pinn_results = []  # Physics-Informed Neural Network results

    num_results = []  # Numerical Results

    for i in range(len(solver_params_list)):
        print('\n--------------------------------------\n')
        print(f'---------GRIDSEARCH: SIM Nº {i}----------')
        print('\n--------------------------------------\n')
        solver_params = solver_params_list[i]
        pinn_model_results = pinn_system_caller.call(
            solver_params=solver_params,
        )

        pinn_results.append(pinn_model_results)

    # TODO faz a solução numérica para comparar, pelo menos um euler vai, vai ser fácil...

    return (pinn_results, best_pinn_index, num_results)
