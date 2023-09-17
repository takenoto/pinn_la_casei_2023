import os
import numpy as np

from domain.params.solver_params import SolverParams
from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults
from domain.optimization.ode_system_caller import RunReactorSystemCaller


def __get_best_pinn(pinn_errors):
    """
    Returns the index and error of the pinn that had the smallest error of all
    """

    best_pinn_test_index = 0
    "O Index do PINN que apresentou menor erro na fase de testes"

    best_pinn_test_error = None

    for i in range(len(pinn_errors)):
        i_pinn_error = pinn_errors[i]

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

    # ---------------------------------------------------------
    for i in range(len(solver_params_list)):
        solver_params = solver_params_list[i]
        print(f"""
              ------------------------------------------
              process {i+1} of {len(solver_params_list)}
              {solver_params.name}
              ------------------------------------------
              """)
        # print("\n--------------------------------------\n")
        # print(f"---------GRIDSEARCH: SIM {name} ----------")
        # print("\n--------------------------------------\n")
        pinn_model_results = pinn_system_caller.call(
            solver_params=solver_params,
        )

        pinn_results.append(np.sum(pinn_model_results.best_loss_test))
        pinn_model_results = None #free memory???
    # ---------------------------------------------------------

    best_pinn_test_index, best_pinn_test_error = __get_best_pinn(
        pinn_errors=pinn_results
    )

    path_to_file = os.path.join(solver_params_list[0].hyperfolder, "best_pinn.txt")
    file = open(path_to_file, "a")
    file.writelines(
        [
            f"Pinn best index = {best_pinn_test_index}\n",
            f"Pinn best error = {best_pinn_test_error}",
        ]
    )
    file.close()

    path_to_file = os.path.join(solver_params_list[0].hyperfolder, "pinns.json")
    file = open(path_to_file, "a")
    file.writelines(
        [
            "{\n",
            f'"pinn_best_index": {best_pinn_test_index},\n',
            f'"pinn_best_error": {best_pinn_test_error},\n',
            '"pinns":{',
        ]
    )
    # Name of each pinn simulated
    for i in range(len(solver_params_list)):
        endChar = "\n"
        if i < (len(solver_params_list) - 1):
            endChar = ",\n"
        file.writelines([f'"{i}":', f'"{solver_params_list[i].name}"', endChar])

    file.writelines(
        [
            "}\n" "}",
        ]
    )

    return (best_pinn_test_index, best_pinn_test_error)
