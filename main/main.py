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


def compare_num_and_pinn(
    num_results,
    pinns,
    p_best_index,
    p_best_error,
    cols,
    rows,
    showPINN=True,
    showNondim=False,
    folder_to_save=None,
):
    # PRINTAR O MELHOR DOS PINNS
    items = {}

    path_to_file = os.path.join(folder_to_save, "best_pinn.txt")
    file = open(path_to_file, "a")
    file.writelines(
        [f"Pinn best index = {p_best_index}\n", f"Pinn best error = {p_best_error}"]
    )
    file.close()

    # Plotar todos os resultados, um a um
    num = num_results[0]

    for pinn in pinns:
        pred_start_time = timer()
        # O t já é nondim!!!!!!!!!!!!!!!
        # Mesmo que fosse é outra escala. Não é assim que funciona.

        t_num_normal = num.non_dim_scaler.fromNondim({"t": num.t}, "t")
        t_nondim = pinn.solver_params.non_dim_scaler.toNondim({"t": t_num_normal}, "t")

        _in = pinn.solver_params.inputSimulationType
        _out = pinn.solver_params.outputSimulationType
        if len(_in.order) == 1:
            vals = np.vstack(
                np.ravel(
                    t_nondim,
                )
            )

        elif len(_in.order) == 2:
            # Determina as entradas
            if _in.X:
                X_nondim = num.X
                vals = np.array(
                    [[X_nondim[i], t_nondim[i]] for i in range(len(t_nondim))]
                )
            elif _in.P:
                P_nondim = num.X
                vals = np.array(
                    [[P_nondim[i], t_nondim[i]] for i in range(len(t_nondim))]
                )
            elif _in.S:
                S_nondim = num.S
                vals = np.array(
                    [[S_nondim[i], t_nondim[i]] for i in range(len(t_nondim))]
                )
            elif _in.V:
                V_nondim = num.V
                vals = np.array(
                    [[V_nondim[i], t_nondim[i]] for i in range(len(t_nondim))]
                )

        prediction = pinn.model.predict(vals)

        N_nondim_pinn = {}
        for o in _out.order:
            if o == "X":
                N_nondim_pinn["X"] = prediction[:, _out.X_index]
            if o == "P":
                N_nondim_pinn["P"] = prediction[:, _out.P_index]
            if o == "S":
                N_nondim_pinn["S"] = prediction[:, _out.S_index]
            if o == "V":
                N_nondim_pinn["V"] = prediction[:, _out.V_index]

        N_pinn = {
            type: pinn.solver_params.non_dim_scaler.fromNondim(N_nondim_pinn, type)
            for type in N_nondim_pinn
        }

        pred_end_time = timer()
        pred_time = pred_end_time - pred_start_time
        path_to_file = os.path.join(folder_to_save, f"{pinn.model_name}.json")
        file = open(path_to_file, "a")
        file.writelines(
            [
                "{\n",
                '"name": ' + f'"{pinn.model_name}"' + ",\n",
                '"solver_params":',
                pinn.solver_params.toJson() + ",\n",
            ]
        )

        # tain data
        file.writelines(
            [
                '"train time":' + f"{pinn.total_training_time}" + ",\n",
                '"best loss test":' + f"{pinn.best_loss_test}" + ",\n",
                '"best loss train":' + f"{pinn.best_loss_train}" + ",\n",
                '"pred time":' + f"{pred_time}" + ",\n",
                '"initializer":' + f'"{pinn.solver_params.initializer}"' + ",\n"
                '"train_distribution":'
                + f'"{pinn.solver_params.train_distribution}"'
                + ",\n",
                '"pinn_x_test":'
                + f"{np.array(pinn.train_state.X_test).tolist()}"
                + ",\n",
                '"pinn_x_train":'
                + f"{np.array(pinn.train_state.X_train).tolist()}"
                + ",",
            ]
        )

        items = {}
        titles = ["X", "P", "S", "V"]
        pinn_vals = [N_pinn[type] if type in _out.order else None for type in titles]
        pinn_nondim_vals = [
            N_nondim_pinn[type] if type in _out.order else None for type in titles
        ]
        num_vals = [
            num.X,  # if _out.X else None,
            num.P,  # if _out.P else None,
            num.S,  # if _out.S else None,
            num.V,  # if _out.V else None,
        ]

        num_vals_json = """{
                "t":[%s],
                "X":[%s],
                "P":[%s],
                "S":[%s],
                "V":[%s]
                }""" % (
            ",".join(np.char.mod("%f", np.array(t_num_normal))),
            ",".join(np.char.mod("%f", np.array(num_vals[0]))),
            ",".join(np.char.mod("%f", np.array(num_vals[1]))),
            ",".join(np.char.mod("%f", np.array(num_vals[2]))),
            ",".join(np.char.mod("%f", np.array(num_vals[3]))),
        )

        # Armazena os 4 erros
        error_L = []
        # Calcula os erros de X P S V
        for u in range(len(num_vals)):
            if pinn_vals[u] is not None:
                diff = np.subtract(pinn_vals[u], num_vals[u])
                total_error = 0
                # Pega ponto a ponto e soma o absoluto
                for value in diff:
                    total_error += abs(value)

                error_L.append(total_error / len(pinn_vals[u]))
            else:
                error_L.append(np.nan)

        # print("ERROR XPSV")
        error_lines = []
        if _out.X:
            error_lines.append(f'"X": {error_L[0]}')
            # print(f"X = {error_L[0]}")
        if _out.P:
            error_lines.append(f'"P": {error_L[1]}')
            # print(f"P = {error_L[1]}")
        if _out.S:
            error_lines.append(f'"S": {error_L[2]}')
            # print(f"S = {error_L[2]}")
        if _out.V:
            error_lines.append(f'"V": {error_L[3]}')
            # print(f"V = {error_L[3]}")

        # print(f"total = {np.nansum(error_L)}")
        error_lines.append(f'"Total": {np.nansum(error_L)}')

        pinn_vals_json = """{
            "t_nondim":[%s],
            "X":[%s],
            "P":[%s],
            "S":[%s],
            "V":[%s]
            }""" % (
            ",".join(np.char.mod("%f", np.array(t_nondim))),
            '"None"'
            if pinn_vals[0] is None
            else ",".join(np.char.mod("%f", np.array(pinn_vals[0]))),
            '"None"'
            if pinn_vals[1] is None
            else ",".join(np.char.mod("%f", np.array(pinn_vals[1]))),
            '"None"'
            if pinn_vals[2] is None
            else ",".join(np.char.mod("%f", np.array(pinn_vals[2]))),
            '"None"'
            if pinn_vals[3] is None
            else ",".join(np.char.mod("%f", np.array(pinn_vals[3]))),
        )

        pinn_nondim_vals_json = """{
            "t_nondim":[%s],
            "X":[%s],
            "P":[%s],
            "S":[%s],
            "V":[%s]
            }""" % (
            ",".join(np.char.mod("%f", np.array(t_nondim))),
            '"None"'
            if pinn_nondim_vals[0] is None
            else ",".join(np.char.mod("%f", np.array(pinn_nondim_vals[0]))),
            '"None"'
            if pinn_nondim_vals[1] is None
            else ",".join(np.char.mod("%f", np.array(pinn_nondim_vals[1]))),
            '"None"'
            if pinn_nondim_vals[2] is None
            else ",".join(np.char.mod("%f", np.array(pinn_nondim_vals[2]))),
            '"None"'
            if pinn_nondim_vals[3] is None
            else ",".join(np.char.mod("%f", np.array(pinn_nondim_vals[3]))),
        )

        # Fecha o arquivo
        # Fecha o body e fecha o error
        file.write('\n"error": {\n')
        for l in range(len(error_lines)):
            line = error_lines[l]
            if l < len(error_lines) - 1:
                file.writelines([line, ",\n"])
            else:
                file.writelines(
                    [
                        line,
                        "},",
                        "\n",
                        '"num_vals": ',
                        num_vals_json,
                        ",\n",
                        '"pinn_vals":',
                        pinn_vals_json,
                        ",\n",
                        '"pinn_nondim_vals":',
                        pinn_nondim_vals_json,
                        "\n",
                        # Essa parte é confusa e não sai direito
                        ",\n",
                        '"pinn_epochs":' + f"{pinn.loss_history.steps}" ",\n",
                        '"pinn_loss_story_test":'
                        + f"{[loss for loss in np.array(np.sum(pinn.loss_history.loss_test, axis=1))]}"
                        ",\n",
                        '"pinn_loss_story_train":'
                        + f"{[loss for loss in np.sum(pinn.loss_history.loss_train, axis=1)]}"
                        ",\n",
                        '"total_training_time":' + f"{pinn.total_training_time}" "\n",
                        "}",
                    ]
                )
        file.close()

        units = ["g/L", "g/L", "g/L", "L"]

        for i in range(4):
            items[i + 1] = {
                "title": titles[i],
                "y_label": units[i],
                "cases": [
                    # Numeric
                    {
                        "x": num.t,
                        "y": num_vals[i],
                        "color": pinn_colors[0],
                        "l": "-",
                    },
                ],
            }

            if showPINN:
                pinn_nondim_vals
                items[i + 1]["cases"].append(
                    # PINN
                    {
                        "x": num.t,
                        "y": pinn_vals[i],
                        "color": pinn_colors[1],
                        "l": "--",
                    }
                )

            if showNondim:
                pinn_nondim_vals
                items[i + 1]["cases"].append(
                    # PINN nondim
                    {
                        "x": num.t,
                        "y": pinn_nondim_vals[i],
                        "color": pinn_colors[2],
                        "l": ":",
                    }
                )

        labels = ["Euler"]
        if showPINN:
            labels.append("PINN")

        if showNondim:
            labels.append("ND PINN")

        plot_comparer_multiple_grid(
            suptitle=pinn.model_name,
            labels=labels,
            figsize=(8, 6),
            gridspec_kw={"hspace": 0.042, "wspace": 0.03},
            yscale="linear",
            sharey=False,
            nrows=2,
            ncols=2,
            items=items,
            title_for_each=True,
            supxlabel="t (h)",
            # supylabel=pinn.model_name,
            folder_to_save=folder_to_save,
            filename=f"{pinn.model_name}.png" if folder_to_save else None,
            showPlot=False if folder_to_save else True,
            legend_bbox_to_anchor=(0.5, -0.1),
        )

    # ------------------------------
    # PLOTAR LOSS DE TODOS
    items = {}
    for i in range(len(pinns)):
        items[i + 1] = {
            "title": pinns[i].model_name,
            "cases": [
                # Loss test = LoV
                {
                    "x": pinns[i].loss_history.steps,
                    "y": np.sum(pinns[i].loss_history.loss_test, axis=1),
                    "color": pinn_colors[-3],
                    "l": "-",
                },
                # Loss Train = LoT
                {
                    "x": pinns[i].loss_history.steps,
                    "y": np.sum(pinns[i].loss_history.loss_train, axis=1),
                    "color": pinn_colors[-1],
                    "l": ":",
                },
            ],
        }

    plot_comparer_multiple_grid(
        labels=["LoV", "LoT"],
        figsize=(10, 10),
        gridspec_kw={"hspace": 0.10, "wspace": 0.05},
        yscale="log",
        sharey=True,
        sharex=True,
        nrows=rows,
        ncols=cols,
        items=items,
        suptitle="Loss",
        title_for_each=True,
        supxlabel="i",
        supylabel="loss",
        folder_to_save=folder_to_save,
        filename="loss.png" if folder_to_save else None,
        showPlot=False if folder_to_save else True,
    )


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
