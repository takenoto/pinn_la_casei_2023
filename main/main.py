# python -m main.main

from timeit import default_timer as timer

import numpy as np
import deepxde
import tensorflow as tf

import matplotlib.pyplot as plt


from domain.params.altiok_2006_params import (
    get_altiok2006_params,
)
from domain.reactor.cstr_state import CSTRState
from domain.params.process_params import ProcessParams
from domain.flow.concentration_flow import ConcentrationFlow
from main.cases_to_try import change_layer_fix_neurons_number
from main.pinn_grid_search import run_pinn_grid_search
from main.numerical_methods import run_numerical_methods


from main.plotting import plot_comparer_multiple_grid


# For obtaining fully reproducible results
deepxde.config.set_random_seed(0)
# Increasing precision
# dde.config.real.set_float64()

xp_colors = ["#F2545B"]
"""
Cores apra dados experimentais
"""

pinn_colors = [
    "#7293A0",
    "#C2E812",
    "#ED9B40",
    "#B4869F",
    "#45B69C",
    "#FFE66D",
    "#E78F8E",
    "#861388",
    "#34312D",
]
"""
Lista de cores que representam diversos pinns
Ordem: RGBA
"""

num_colors = [
    "b",
    "#F39B6D",
    "#F0C987",
]


def compare_num_and_pinn(
    num_results, pinns, p_best_index, p_best_error, cols, rows, showNondim=False
):
    # PRINTAR O MELHOR DOS PINNS
    items = {}
    print(f"Pinn best index = {p_best_index}")
    print(f"Pinn best error = {p_best_error}")

    # Plotar todos os resultados, um a um
    num = num_results[0]
    for pinn in pinns:
        pred_start_time = timer()
        t_nondim = pinn.solver_params.non_dim_scaler.toNondim({"t": num.t}, "t")
        # t = num.t / pinn.solver_params.non_dim_scaler.t_not_tensor

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
        print(f"name = {pinn.model_name}")
        print(f"train time = {pinn.total_training_time} s")
        print(f"best loss test = {pinn.best_loss_test}")
        print(f"best loss train = {pinn.best_loss_train}")
        print(f"pred time = {pred_time} s")
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
        print("ERROR XPSV")
        if _out.X:
            print(f"X = {error_L[0]}")
        if _out.P:
            print(f"P = {error_L[1]}")
        if _out.S:
            print(f"S = {error_L[2]}")
        if _out.V:
            print(f"V = {error_L[3]}")

        print(f"total = {np.nansum(error_L)}")

        for i in range(4):
            items[i + 1] = {
                "title": titles[i],
                "cases": [
                    # Numeric
                    {
                        "x": num.t,
                        "y": num_vals[i],
                        "color": pinn_colors[0],
                        "l": "-",
                    },
                    # PINN
                    {
                        "x": num.t,
                        "y": pinn_vals[i],
                        "color": pinn_colors[1],
                        "l": "--",
                    },
                ],
            }
            if showNondim:
                pinn_nondim_vals
                items[i + 1]["cases"].append(
                    # PINN
                    {
                        "x": num.t,
                        "y": pinn_nondim_vals[i],
                        "color": pinn_colors[2],
                        "l": "--",
                    }
                )

        labels = ["Euler", "PINN"]

        if showNondim:
            labels.append("ND PINN")

        plot_comparer_multiple_grid(
            suptitle=pinn.model_name,
            labels=labels,
            figsize=(6 * 1.5, 8 * 1.5),
            gridspec_kw={"hspace": 0.6, "wspace": 0.25},
            yscale="linear",
            sharey=False,
            nrows=2,
            ncols=2,
            items=items,
            title_for_each=True,
            supxlabel="tempo (h)",
            supylabel="g/L",
        )

    # ------------------------------
    # PLOTAR LOSS DE TODOS
    items = {}
    for i in range(len(pinns)):
        items[i + 1] = {
            "title": pinns[i].model_name,
            "cases": [
                {
                    "x": pinns[i].loss_history.steps,
                    "y": np.sum(pinns[i].loss_history.loss_test, axis=1),
                    "color": pinn_colors[0],
                    "l": "-",
                },
                {
                    "x": pinns[i].loss_history.steps,
                    "y": np.sum(pinns[i].loss_history.loss_train, axis=1),
                    "color": pinn_colors[1],
                    "l": "--",
                },
            ],
        }

    plot_comparer_multiple_grid(
        labels=["Loss (teste)", "Loss (treino)"],
        figsize=(7.2 * 2, 8.2 * 2),
        gridspec_kw={"hspace": 0.35, "wspace": 0.14},
        yscale="log",
        sharey=True,
        sharex=True,
        nrows=rows,
        ncols=cols,
        items=items,
        suptitle=None,
        title_for_each=True,
        supxlabel="epochs",
        supylabel="loss",
    )


def main():
    deepxde.config.set_random_seed(0)

    plt.style.use("./main/plotting/plot_styles.mplstyle")

    run_fedbatch = False

    run_cstr = False

    run_batch = True

    altiok_models_to_run = [get_altiok2006_params().get(2)]  # roda só a fig2

    # Parâmetros de processo (será usado em todos)
    eq_params = altiok_models_to_run[0]
    # Serve pra batch e pra cstr
    initial_state = CSTRState(
        volume=np.array([5]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    initial_state_cstr = CSTRState(
        volume=np.array([1]),
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    )

    # Serve pra fed-batch
    initial_state_fed_batch = CSTRState(
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
        max_reactor_volume=5,
        inlet=ConcentrationFlow(
            volume=2,  # L/h
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        ),
        # TODO e esse tempo????
        # t_final=10.2,
        t_final=24 * 4,
    )

    process_params_feed_cstr = ProcessParams(
        max_reactor_volume=5,
        inlet=ConcentrationFlow(
            volume=1,  # 1,  # L/h => 1 é o valor padrão
            # baixei artificialmente. Pareceu evitar bem os nans...
            X=eq_params.Xo,  # *0.1,
            P=eq_params.Po,
            S=eq_params.So,
        ),
        t_final=24 * 4,  # => ERA O PADRÃO!
        # t_final=10.2,
    )

    if run_fedbatch:
        print("RUN FED-BATCH")
        cases, cols, rows = change_layer_fix_neurons_number(
            eq_params, process_params_feed_fb
        )

        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_fb,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            t_discretization_points=[400],
        )

        start_time = timer()

        pinns, p_best_index, p_best_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_fb,
            initial_state=initial_state_fed_batch,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases,
        )
        end_time = timer()
        print(f"elapsed time for test = {end_time - start_time} secs")

        compare_num_and_pinn(
            num_results, pinns, p_best_index, p_best_error, cols, rows, showNondim=True
        )

        pass

    if run_cstr:
        print("RUN CSTR")
        cases, cols, rows = change_layer_fix_neurons_number(
            eq_params, process_params_feed_cstr
        )

        def cstr_f_out_calc_numeric(max_reactor_volume, f_in_v, volume):
            return f_in_v * pow(volume / max_reactor_volume, 7)

        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_numeric,
            t_discretization_points=[400],
        )

        start_time = timer()

        def cstr_f_out_calc_tensorflow(max_reactor_volume, f_in_v, volume):
            return f_in_v * tf.math.pow(volume / max_reactor_volume, 7)

        pinns, p_best_index, p_best_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_cstr,
            initial_state=initial_state_cstr,
            f_out_value_calc=cstr_f_out_calc_tensorflow,
            cases_to_try=cases,
        )
        end_time = timer()
        print(f"elapsed time for test = {end_time - start_time} secs")

        compare_num_and_pinn(num_results, pinns, p_best_index, p_best_error, cols, rows)

        pass

    if run_batch:
        print("RUN BATCH")
        cases, cols, rows = change_layer_fix_neurons_number(
            eq_params, process_params_feed_off
        )
        num_results = run_numerical_methods(
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            t_discretization_points=[400],
        )

        start_time = timer()

        print(f"NUMBER OF CASES ={len(cases)}")

        pinns, p_best_index, p_best_error = run_pinn_grid_search(
            solver_params_list=None,
            eq_params=eq_params,
            process_params=process_params_feed_off,
            initial_state=initial_state,
            f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
            cases_to_try=cases,
        )
        end_time = timer()

        print(f"elapsed time for BATCH NONDIM test = {end_time - start_time} secs")
        compare_num_and_pinn(
            num_results, pinns, p_best_index, p_best_error, cols, rows, showNondim=True
        )

        pass


if __name__ == "__main__":
    main()
