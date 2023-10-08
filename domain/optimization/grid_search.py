import gc
import numpy as np

import tensorflow as tf
import deepxde

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
        tf.keras.backend.clear_session()    
        # The seed was deleted and needs to be set again
        deepxde.config.set_random_seed(0)
        
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
          
        unreachable = gc.collect()  
        
        # Clear old objects, making the train in loop much easier
        tf.keras.backend.clear_session()    
        print(f"{unreachable} unreachable objects")
        unreachable = gc.collect()  
        print(f"{unreachable} unreachable objects AFTER CLEANING UP KERAS SESSION")
        
    # ---------------------------------------------------------

    return (best_pinn_test_index, best_pinn_test_error)
