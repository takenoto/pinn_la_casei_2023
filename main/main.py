# python -m main.main

# Changed the backend using:
# python -m deepxde.backend.set_default_backend BACKEND
# ref: https://deepxde.readthedocs.io/en/latest/user/installation.html#working-with-different-backends
#
# NOTE: Setting environment variables (both user and system) did not work.
# Windows 11, 2023-05-10.

import matplotlib as mpl
import matplotlib.pyplot as plt


import os
from timeit import default_timer as timer
from typing import List

import deepxde
import tensorflow as tf

from data.pinn_saver import PINNSaveCaller
from domain.optimization.non_dim_scaler import NonDimScaler

from utils.colors import xp_colors

from domain.params.altiok_2006_params import (
    Altiok2006Params,
    get_altiok2006_params,
    get_altiok2006_xp_data,
)
from domain.reactor.reactor_state import ReactorState
from domain.params.process_params import ProcessParams
from domain.flow.concentration_flow import ConcentrationFlow
from main.cases_to_try import change_layer_fix_neurons_number
from main.pinn_grid_search import run_pinn_grid_search
from main.numerical_methods import run_numerical_methods


# Supostamente conserta erros pelo caminho
mpl.rcParams.update(mpl.rcParamsDefault)

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
    base_folder,
    eq_params,
    process_params,
    initial_state,
    f_out_num,
    f_out_pinn,
    cases,
    additional_plotting_points,
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
        additional_plotting_points=additional_plotting_points,
    )
    cases = change_layer_fix_neurons_number(eq_params, process_params)
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

    batch_versions = [
        # tempo de simulação, Xo, Po, So
        # -----------
        # Original xp time
        (10, "Xo", "Po", "So"),
        # Default 20
        # (20, "Xo", "Po", "So"),
        # -----------
        # Alternatives:
        # (11, "Xo", "Po", "So"),
        # (11, "Xo", "0", "So"),
        # (11, "0", "Po", "So"),
    ]

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
        # (1, 5, "1", "-4"),
        # Aumentando muito lentamente
        # (1, 5, "1", "-2"),
        # Normal
        # (1, 5, "25", "-2"),
        # (3, 5, "25", "-2"),
        # Curva suave
        (4, 5, "5", "-1"),
        # Enchimento rápido
        # (0, 5, "5", "0"),
    ]

    # --------------------------------------------
    # ----------------MAIN CODE-------------------
    # --------------------------------------------
    start_time = timer()

    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)
    eq_params = altiok_models_to_run[0]

    # Zero vezes volume para já converter por si só quando for um tensor
    def f_out_0(max_reactor_volume, f_in_v, volume):
        return 0.0 * volume

    for current_reactor in reactors_to_run:
        f_out_num = f_out_0
        f_out_pinn = f_out_0
        initial_state = None
        process_params = None
        cases = []

        # Folder creation
        mega_reactor_folder = create_folder_to_save(
            subfolder=os.path.join(subfolder, current_reactor)
        )

        match current_reactor:
            case "fed-batch":
                print("RUN FED-BATCH")
                f_out_num = f_out_0
                f_out_pinn = f_out_0
                process_params = ProcessParams(
                    max_reactor_volume=10.0,
                    inlet=ConcentrationFlow(
                        volume=0.25,  # L/h
                        X=eq_params.Xo*1.0,
                        P=eq_params.Po*1.0,
                        S=eq_params.So*1.0,
                    ),
                    t_final=22.0,
                    S_max=float("inf"),
                )
                initial_state = ReactorState(
                    volume=1.0,
                    X=eq_params.Xo*1.0,
                    P=eq_params.Po*1.0,
                    S=eq_params.So*1.0,
                )

                current_test_folder = mega_reactor_folder

                compute_num_and_pinn(
                    current_test_folder,
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
                f_out_num = f_out_0
                f_out_pinn = f_out_0
                for batch_version in batch_versions:
                    params = batch_version
                    # names:
                    t_sim, Xo, Po, So = params
                    current_test_folder = os.path.join(
                        mega_reactor_folder,
                        f"t{t_sim}-{Xo}-{Po}-{So}",
                    )
                    
                    # Show XP data
                    if Xo == "Xo" and Po == "Po" and So == "So":
                        xpdata = get_altiok2006_xp_data(xp_num=2)
                        XPSvals = {
                            "X": xpdata.X,
                            "P": xpdata.P,
                            "S": xpdata.S,
                            # "V": None,
                        }
                        additional_plotting_points = {
                            "XPSV": {
                                "title": "XP",
                                "cases": {
                                    key: {
                                        "x": xpdata.t,
                                        "y": XPSvals[key],
                                        # Essa label é o tipo pro gráfico
                                        # Então tipo X é XP porque já vai estar
                                        # no subplot de X
                                        # vai apenas dizer se é PINN, XP, Num...
                                        "label": "XP",
                                        "marker": "o",
                                        "color": xp_colors[0],
                                    }
                                    for key in XPSvals
                                }
                            }
                        }

                    # Get the real values
                    t_sim, Xo, Po, So = batch_get_variables(
                        params=params, eq_params=eq_params
                    )

                    process_params = ProcessParams(
                        max_reactor_volume=5.0,
                        inlet=ConcentrationFlow(
                            volume=0.0,
                            X=eq_params.Xo*1.0,
                            P=eq_params.Po*1.0,
                            S=eq_params.So*1.0,
                        ),
                        t_final=t_sim + 0.0,
                        Smax=So,
                    )
                    initial_state = ReactorState(
                        volume=5.0,
                        X=Xo*1.0,
                        P=Po*1.0,
                        S=So*1.0,
                    )
                    compute_num_and_pinn(
                        current_test_folder,
                        eq_params,
                        process_params,
                        initial_state,
                        f_out_num,
                        f_out_pinn,
                        cases,
                        additional_plotting_points,
                    )
                    pass

            case "CR":
                for cr_version in cr_versions:
                    cr_id, V0, Vmax, Fin = cr_get_variables(cr_version)
                    current_test_folder = os.path.join(
                        mega_reactor_folder,
                        cr_id,
                    )
                    process_params = ProcessParams(
                        max_reactor_volume=Vmax*1.0,
                        inlet=ConcentrationFlow(
                            volume=Fin*1.0,
                            X=eq_params.Xo * 0.0,
                            P=eq_params.Po * 0.0,
                            S=eq_params.So,
                        ),
                        t_final=24 * 3*1.0,
                        Smax=float("inf"),
                    )
                    initial_state = ReactorState(
                        volume=V0*1.0,
                        X=eq_params.Xo*1.0,
                        P=eq_params.Po*1.0,
                        S=eq_params.So*1.0,
                    )

                    def cr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
                        return f_in_v * pow(volume / max_reactor_volume, 7)

                    f_out_num = cr_f_out_calc_numeric

                    def cr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
                        return f_in_v * tf.math.pow(volume / max_reactor_volume, 7)

                    f_out_pinn = cr_f_out_calc_tensorflow
                    compute_num_and_pinn(
                        current_test_folder,
                        eq_params,
                        process_params,
                        initial_state,
                        f_out_num,
                        f_out_pinn,
                        cases,
                        additional_plotting_points={}
                    )
                    pass
    # -----------------------------------
    end_time = timer()
    print(
        f"""
        time for all tests = {end_time - start_time} s
        = {(end_time - start_time)/60} min"""
    )


def batch_get_variables(params, eq_params: Altiok2006Params):
    t_sim, Xo, Po, So = params

    match Xo:
        case "0":
            Xo = 0.0
        case "Xo":
            Xo = eq_params.Xo
    match Po:
        case "0":
            Po = 0.0
        case "Po":
            Po = eq_params.Po
    match So:
        case "0":
            So = 0.0
        case "So":
            So = eq_params.So

    return t_sim*1.0, Xo*1.0, Po*1.0, So*1.0


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
