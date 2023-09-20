import os
import numpy as np

from domain.optimization.ode_system_caller import RunReactorSystemCaller

def grid_search(
    pinn_system_caller: RunReactorSystemCaller,
    solver_params_list: list,  # SolverParams,
):
    "Receive a list of each kind of parameter and test them"

    best_pinn_test_error = None
    best_pinn_test_index = None
                
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

        i_pinn_error = np.sum(pinn_model_results.best_loss_test)
        if best_pinn_test_error is None:
            best_pinn_test_error = i_pinn_error
        else:
            if best_pinn_test_error > i_pinn_error:
                best_pinn_test_error = i_pinn_error
                best_pinn_test_index = i
                
    # ---------------------------------------------------------

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
    
    file.close()

    return (best_pinn_test_index, best_pinn_test_error)
