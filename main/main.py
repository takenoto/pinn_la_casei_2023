# python -m main.main

# Changed the backend using:
# python -m deepxde.backend.set_default_backend BACKEND
# ref: https://deepxde.readthedocs.io/en/latest/user/installation.html#working-with-different-backends
#
# NOTE: Setting environment variables (both user and system) did not work.
# Windows 11, 2023-05-10.


import os
from timeit import default_timer as timer
from typing import List


import numpy as np
import deepxde
import tensorflow as tf

import matplotlib.pyplot as plt
from data.pinn_saver import PINNSaveCaller
from domain.optimization.non_dim_scaler import NonDimScaler


from domain.params.altiok_2006_params import (
    get_altiok2006_params,
)
from domain.reactor.reactor_state import ReactorState
from domain.params.process_params import ProcessParams
from domain.flow.concentration_flow import ConcentrationFlow
from main.cases_to_try import change_layer_fix_neurons_number
from main.pinn_grid_search import run_pinn_grid_search
from main.numerical_methods import run_numerical_methods


# For obtaining fully reproducible results
deepxde.config.set_random_seed(0)
# Increasing precision
# dde.config.real.set_float64()


def create_folder_to_save(subfolder):
    current_directory_path = os.getcwd()
    folder_to_save = os.path.join(
        current_directory_path, "results", "exported", subfolder
    )
    # ref: https://stackoverflow.com/questions/56012636/python-mathplotlib-savefig-filenotfounderror
    # Create the folder if it does not exist
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    return folder_to_save


# If true, also plots the nondim values from pinn
showNondim = False
showPINN = True


# -------- END MATCH
def compute_num_and_pinn(
    base_folder, eq_params, process_params, initial_state, f_out_num, f_out_pinn, cases
):
    print("-------STARTING---------")

    num_results = run_numerical_methods(
        eq_params=eq_params,
        process_params=process_params,
        initial_state=initial_state,
        f_out_value_calc=f_out_num,
        t_discretization_points=[400],
        non_dim_scaler=NonDimScaler(),
    )
    # Creates the saver for this session
    save_caller = PINNSaveCaller(
        num_results=num_results,
        showPINN=showPINN,
        showNondim=showNondim,
    )
    cases, cols, rows = change_layer_fix_neurons_number(eq_params, process_params)
    for case_name in cases:
        cases[case_name]["save_caller"] = save_caller
        hyperfolder_updated = os.path.join(base_folder, cases[case_name]["hyperfolder"])
        create_folder_to_save(hyperfolder_updated)
        cases[case_name]["hyperfolder"] = hyperfolder_updated

    start_time = timer()

    run_pinn_grid_search(
        solver_params_list=None,
        eq_params=eq_params,
        process_params=process_params,
        initial_state=initial_state,
        f_out_value_calc=f_out_pinn,
        cases_to_try=cases,
    )
    end_time = timer()
    print(
        f"""
        time for test = {end_time - start_time} s
        = {(end_time - start_time)/60} min"""
    )
    pass


def main():
    deepxde.config.set_random_seed(0)

    plt.style.use("./main/plotting/plot_styles.mplstyle")

    # ----------------------
    # ------SETTINGS--------
    # ----------------------

    # If None, the plots will be shown()
    # If a directory, the plots will be saved
    subfolder = "reactor_altiok2006"

    # ----------------------
    # -CHOSE OPERATION MODE-
    # ----------------------
    reactors_to_run = ["batch"]  # "batch" "fed-batch" "CR"

    cr_versions = [
        # (V0, Vmax, F_in, F_inE)
        # F_inE é o multiplicador por 10 de notação científica de Fin
        # Ex: Fin = 2.5*10^-1 ==> Fin=25 e F_inE = -2 ==> 25E-2
        # ---------------
        # CSTR:
        # (5, 5, "0", "0"),
        # ---------------
        # CRs:
        # Aumentando muito MUITO lentamente
        (1, 5, "1", "-4"),
        # Aumentando muito lentamente
        # (1, 5, "1", "-2"),
        # Normal
        (1, 5, "25", "-2"),
        # (3, 5, "25", "-2"),
        # Curva suave
        # (4, 5, "5", "-1"),
        # Enchimento rápido
        # (0, 5, "5", "0"),
    ]

    # --------------------------------------------
    # ----------------MAIN CODE-------------------
    # --------------------------------------------

    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)
    eq_params = altiok_models_to_run[0]

    for current_reactor in reactors_to_run:

        def f_out_num(max_reactor_volume, f_in_v, volume):
            return 0

        f_out_pinn = f_out_num
        initial_state = None
        process_params = None
        cases = []

        # Folder creation
        current_reactor_folder = create_folder_to_save(
            subfolder=os.path.join(subfolder, current_reactor)
        )

        match current_reactor:
            case "fed-batch":
                print("RUN FED-BATCH")

                def f_out_num(max_reactor_volume, f_in_v, volume):
                    return 0

                f_out_pinn = f_out_num
                process_params = ProcessParams(
                    max_reactor_volume=10,
                    inlet=ConcentrationFlow(
                        volume=0.25,  # L/h
                        X=eq_params.Xo,
                        P=eq_params.Po,
                        S=eq_params.So,
                    ),
                    t_final=2 * 10.2,
                )
                initial_state = ReactorState(
                    volume=np.array([1]),
                    X=eq_params.Xo,
                    P=eq_params.Po,
                    S=eq_params.So,
                )
                compute_num_and_pinn(
                    current_reactor_folder,
                    eq_params,
                    process_params,
                    initial_state,
                    f_out_num,
                    f_out_pinn,
                    cases,
                )
                pass

            case "batch":
                print("RUN BATCH")
                process_params = ProcessParams(
                    max_reactor_volume=5,
                    inlet=ConcentrationFlow(
                        volume=0.0,
                        X=eq_params.Xo,
                        P=eq_params.Po,
                        S=eq_params.So,
                    ),
                    t_final=10.2,
                )
                initial_state = ReactorState(
                    volume=np.array([5]),
                    X=eq_params.Xo,
                    P=eq_params.Po,
                    S=eq_params.So,
                )
                compute_num_and_pinn(
                    current_reactor_folder,
                    eq_params,
                    process_params,
                    initial_state,
                    f_out_num,
                    f_out_pinn,
                    cases,
                )
                pass

            case "CR":
                for cr_version in cr_versions:
                    cr_id, V0, Vmax, Fin = cr_get_variables(cr_version)

                    process_params = ProcessParams(
                        max_reactor_volume=Vmax,
                        inlet=ConcentrationFlow(
                            volume=Fin,
                            X=eq_params.Xo * 0,
                            P=eq_params.Po * 0,
                            S=eq_params.So,
                        ),
                        t_final=24 * 3,
                    )
                    initial_state = ReactorState(
                        volume=np.array([V0]),
                        X=eq_params.Xo,
                        P=eq_params.Po,
                        S=eq_params.So,
                    )

                    def cr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
                        return f_in_v * pow(volume / max_reactor_volume, 7)

                    f_out_num = cr_f_out_calc_numeric

                    def cr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
                        return f_in_v * tf.math.pow(volume / max_reactor_volume, 7)

                    f_out_pinn = cr_f_out_calc_tensorflow
                    compute_num_and_pinn(
                        current_reactor_folder,
                        eq_params,
                        process_params,
                        initial_state,
                        f_out_num,
                        f_out_pinn,
                        cases,
                    )
                    pass


def cr_get_variables(params: List):
    def create_id(V0, Vmax, F_in, F_inE):
        return f"V0-{int(V0)}--Vmax-{Vmax}--Fin-{F_in}E{F_inE}"

    Findict = {}
    for Fin in range(0, 101):
        for FinE in range(-5, 5):
            Findict[f"{Fin}-E{FinE}"] = Fin * (10**FinE)

    V0, Vmax, Fin, FinE = params
    id = create_id(V0, Vmax, Fin, FinE)
    return id, V0, Vmax, Findict[f"{Fin}-E{FinE}"]


if __name__ == "__main__":
    main()
