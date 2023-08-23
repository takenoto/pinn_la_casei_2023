import numpy as np

from domain.params.solver_params import SolverParams
from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults
from domain.optimization.ode_system_caller import RunReactorSystemCaller


def __get_best_pinn(pinn_models):
    """
    Returns the index and error of the pinn that had the smallest error of all
    """

    best_pinn_test_index = 0
    "O Index do PINN que apresentou menor erro na fase de testes"

    best_pinn_test_error = None

    for i in range(len(pinn_models)):
        pinn = pinn_models[i]

        i_pinn_error = np.sum(pinn.best_loss_test) # The error of this pinn

        if best_pinn_test_error is None:
            best_pinn_test_error = i_pinn_error
        else:
            if best_pinn_test_error > i_pinn_error:
                best_pinn_test_error = i_pinn_error
                best_pinn_test_index = i

    return best_pinn_test_index, best_pinn_test_error


def grid_search(
    pinn_system_caller: RunReactorSystemCaller,
    solver_params_list: list,  # SolverParams,
):

    "Receive a list of each kind of parameter and test them"

    pinn_results = []  # Physics-Informed Neural Network results

    #---------------------------------------------------------
    for i in range(len(solver_params_list)):
        
        solver_params = solver_params_list[i]
        name=solver_params.name
        print(f'process {i} of {len(solver_params_list)}')
        # print("\n--------------------------------------\n")
        # print(f"---------GRIDSEARCH: SIM {name} ----------")
        # print("\n--------------------------------------\n")
        pinn_model_results = pinn_system_caller.call(
            solver_params=solver_params,
        )
        

        pinn_results.append(pinn_model_results)
    #---------------------------------------------------------

    best_pinn_test_index, best_pinn_test_error = __get_best_pinn(
        pinn_models=pinn_results
    )

    return (pinn_results, best_pinn_test_index, best_pinn_test_error)
