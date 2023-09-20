# python -m main.main

import os
from timeit import default_timer as timer


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

from data.plot.plot_comparer_multiple_grid import *
from main.plotting import *


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


def main():
    deepxde.config.set_random_seed(0)

    plt.style.use("./main/plotting/plot_styles.mplstyle")

    # ----------------------
    # ------SETTINGS--------
    # ----------------------

    # If None, the plots will be shown()
    # If a directory, the plots will be saved
    subfolder = "reactor-"
    folder_to_save = create_folder_to_save(subfolder=subfolder)

    # If true, also plots the nondim values from pinn
    showNondim = False
    showPINN = True

    # ----------------------
    # -CHOSE OPERATION MODE-
    # ----------------------
    run_fedbatch = False

    run_cr = True
    cr_versions = ["cr-1L", "cr-1E-1L", "cstr"]  # "cstr" "cr-1L" "cr-1-E1L"

    run_batch = False

    # --------------------------------------------
    # ----------------MAIN CODE-------------------
    # --------------------------------------------

    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)
    eq_params = altiok_models_to_run[0]

    # BATCH
    initial_state = ReactorState(
        volume=np.array([5]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    # Serve pra fed-batch
    initial_state_fed_batch = ReactorState(
        volume=np.array([1]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    process_params_feed_off = ProcessParams(
        max_reactor_volume=5,
        inlet=ConcentrationFlow(
            volume=0.0,
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        ),
        t_final=10.2,
    )

    process_params_feed_fb = ProcessParams(
        max_reactor_volume=10,
        inlet=ConcentrationFlow(
            volume=0.25,  # L/h
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        ),
        t_final=2 * 10.2,
    )

    # CR
    cr_states_dict = {
        "cstr": ReactorState(
            volume=np.array([5]),
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        ),
        "cr-1L": ReactorState(
            volume=np.array([1]),
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        ),
        "cr-1E-1L": ReactorState(
            volume=np.array([0.1]),
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        ),
    }

    if run_fedbatch:
        folder_to_save = create_folder_to_save(subfolder=subfolder + "-fb")
        print("RUN FED-BATCH")
        cases, cols, rows = change_layer_fix_neurons_number(
            eq_params, process_params_feed_fb, hyperfolder=folder_to_save
        )

        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_fb,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            t_discretization_points=[400],
            non_dim_scaler=NonDimScaler(),
        )

        # Creates the saver for this session
        save_caller = PINNSaveCaller(
            num_results=num_results,
            showPINN=showPINN,
            showNondim=showNondim,
        )
        for case_name in cases:
            cases[case_name]["save_caller"] = save_caller
        start_time = timer()

        run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_fb,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases,
        )
        end_time = timer()
        print(
            f"""
            time for test = {end_time - start_time} s
            = {(end_time - start_time)/60} min"""
        )
        pass

    if run_cr:
        for cr_version in cr_versions:
            initial_state_cr = cr_states_dict[cr_version]

            process_params_feed_cr = ProcessParams(
                max_reactor_volume=5,
                inlet=ConcentrationFlow(
                    volume=0.25,
                    X=eq_params.Xo * 0,  # *0.1,
                    P=eq_params.Po * 0,
                    S=eq_params.So,
                ),
                t_final=24 * 3,
            )
            folder_to_save = create_folder_to_save(subfolder=subfolder + cr_version)

            print(f"RUN CR {cr_version}")
            cases, cols, rows = change_layer_fix_neurons_number(
                eq_params, process_params_feed_cr, hyperfolder=folder_to_save
            )

            def cr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
                return f_in_v * pow(volume / max_reactor_volume, 7)

            num_results = run_numerical_methods(
                eq_params=eq_params,
                process_params=process_params_feed_cr,
                initial_state=initial_state_cr,
                f_out_value_calc=cr_f_out_calc_numeric,
                t_discretization_points=[400],
                non_dim_scaler=NonDimScaler(),
            )

            # Creates the saver for this session
            save_caller = PINNSaveCaller(
                num_results=num_results,
                showPINN=showPINN,
                showNondim=showNondim,
            )
            for case_name in cases:
                cases[case_name]["save_caller"] = save_caller

            start_time = timer()

            def cr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
                return f_in_v * tf.math.pow(volume / max_reactor_volume, 7)

            run_pinn_grid_search(
                solver_params_list=None,
                eq_params=eq_params,
                process_params=process_params_feed_cr,
                initial_state=initial_state_cr,
                f_out_value_calc=cr_f_out_calc_tensorflow,
                cases_to_try=cases,
            )
            end_time = timer()
            print(
                f"""
                time for test = {end_time - start_time} s
                = {(end_time - start_time)/60} min"""
            )
            pass

    if run_batch:
        folder_to_save = create_folder_to_save(subfolder=subfolder + "batch")

        print("RUN BATCH")
        cases, cols, rows = change_layer_fix_neurons_number(
            eq_params, process_params_feed_off, hyperfolder=folder_to_save
        )
        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            t_discretization_points=[400],
            non_dim_scaler=NonDimScaler(),
        )

        # Creates the saver for this session
        save_caller = PINNSaveCaller(
            num_results=num_results,
            showPINN=showPINN,
            showNondim=showNondim,
        )
        for case_name in cases:
            cases[case_name]["save_caller"] = save_caller

        start_time = timer()

        print(f"NUMBER OF CASES ={len(cases)}")

        run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases,
            save_caller=save_caller,
        )
        end_time = timer()

        print(
            f"""
            time for test = {end_time - start_time} s
            = {(end_time - start_time)/60} min"""
        )
        pass


if __name__ == "__main__":
    main()
