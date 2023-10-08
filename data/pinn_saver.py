import os
import json
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer
import deepxde as dde

from data.plot.plot_comparer_multiple_grid import plot_comparer_multiple_grid
from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults

xp_colors = ["#F2545B"]
"""
Cores apra dados experimentais
"""
pinn_colors = [
    "olivedrab",  # "#7293A0",
    "darkorange",  # "#C2E812",
    "#ED9B40",
    "#B4869F",
    "#45B69C",
    "#FFE66D",
    "#E78F8E",
    "#861388",
    "#045275",
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


class PINNSaveCaller:
    def __init__(self, num_results, showPINN, showNondim):
        self.num_results = num_results
        self.showPINN = showPINN
        self.showNondim = showNondim

    def save_pinn(
        self,
        pinn,
        folder_to_save,
        showPINN=None,
        showNondim=None,
        # Derivatives:
        plot_derivatives=True,
        # Second order derivatives:
        plot_derivatives_2=True,
        # Individual losses of each output and inicial conditions
        plot_individual_losses=True,
    ):
        "Saves the plot and json of the pinn"
        save_each_pinn(
            num_results=self.num_results,
            pinn=pinn,
            showPINN=showPINN if showPINN else self.showPINN,
            showNondim=showNondim if showNondim else self.showNondim,
            plot_derivatives=plot_derivatives,
            plot_derivatives_2=plot_derivatives_2,
            plot_individual_losses=plot_individual_losses,
            folder_to_save=folder_to_save,
        )


def save_each_pinn(
    num_results,
    pinn: PINNReactorModelResults,
    showPINN=True,
    showNondim=False,
    folder_to_save=None,
    plot_derivatives=True,
    plot_derivatives_2=True,
    plot_individual_losses=True,
    showTimeSpan=True,
    create_time_points_plot=False,
):
    # PRINTAR O MELHOR DOS PINNS
    items = {}

    # Plotar todos os resultados, um a um
    num = num_results[0]

    _in = pinn.solver_params.inputSimulationType
    _out = pinn.solver_params.outputSimulationType
    if len(_in.order) == 1:
        input_x_dde = np.vstack(
            np.ravel(
                num.t,
            )
        )

    elif len(_in.order) >= 2:
        inputs_dict = {
            "t": num.t,
            "X": num.X,
            "P": num.P,
            "S": num.S,
            "V": num.V,
            # etc => o resto viria aqui
        }
        values_of_input = []
        for N in pinn.solver_params.inputSimulationType.order:
            N_input_index = pinn.solver_params.inputSimulationType.get_index_for(N)
            if N_input_index is not None:
                values_of_input.append(inputs_dict[N])
        input_x_dde = np.concatenate(values_of_input, axis=1)

    pred_start_time = timer()
    prediction_y = pinn.model.predict(input_x_dde)
    pred_end_time = timer()
    pred_time = pred_end_time - pred_start_time

    dNdt_keys = []
    dNdt_2_keys = []

    # Obtenção da derivada:
    # Ref: https://github.com/lululxvi/deepxde/issues/177
    # def dydx(x, y):
    #     return dde.grad.jacobian(y, x, i=0, j=0)

    N_pinn = {}
    pinn_dNdt_dict = {}
    pinn_dNdt_2_dict = {}

    # Initialize with none
    for N in ["X", "P", "S", "V"]:
        pinn_dNdt_dict[f"d{N}dt"] = None
        pinn_dNdt_2_dict[f"d{N}dt_2"] = None
        dNdt_keys.append(f"d{N}dt")
        dNdt_2_keys.append(f"d{N}dt_2")

    for N in _out.order:
        N_index = _out.get_index_for(N)
        N_pinn[N] = prediction_y[:, N_index]

        pinn_dNdt_dict[f"d{N}dt"] = np.array(
            pinn.model.predict(
                input_x_dde,
                operator=lambda x, y: dde.grad.jacobian(y, x, i=N_index, j=_in.t_index),
            )
        ).tolist()

        pinn_dNdt_2_dict[f"d{N}dt_2"] = np.array(
            pinn.model.predict(
                input_x_dde,
                operator=lambda x, y: dde.grad.hessian(
                    y,
                    x,
                    # Deepxde throws => ::Do not use component for 1D y::
                    component=N_index if len(_out.order) > 1 else None,
                    i=_in.t_index,
                    j=_in.t_index,
                    # Se ligar isso ele quebra:
                    # grad_y=N_pinn_derivatives[dNdtkey]
                ),
            )
        ).tolist()

    file_dict = {
        "name": pinn.model_name,
    }

    items = {}
    titles = ["X", "P", "S", "V"]
    derivatives = {}
    derivatives_2 = {}
    derivatives_titles = [
        "$dX/dt$",
        "$dP/dt$",
        "$dS/dt$",
        "$dV/dt$",
    ]
    derivatives_2_titles = [
        "$d^{2}X/dt^{2}$",
        "$d^{2}P/dt^{2}$",
        "$d^{2}S/dt^{2}$",
        "$d^{2}V/dt^{2}$",
    ]
    pinn_vals = [N_pinn[type] if type in _out.order else None for type in titles]

    if showNondim:
        pinn_nondim_vals = [
            pinn.solver_params.output_non_dim_scaler.toNondim(N_pinn, type)
            if type in _out.order
            else None
            for type in titles
        ]

    pinn_derivative_vals = [
        np.array(pinn_dNdt_dict[type]).tolist() for type in dNdt_keys
    ]
    pinn_derivative_2_vals = [
        np.array(pinn_dNdt_2_dict[type]).tolist() for type in dNdt_2_keys
    ]

    num_vals = [
        np.array(num.X).tolist(),
        np.array(num.P).tolist(),
        np.array(num.S).tolist(),
        np.array(num.V).tolist(),
    ]
    num_dNdt = [num.dX_dt, num.dP_dt, num.dS_dt, num.dV_dt]
    num_dNdt_dict = {
        "dXdt": np.array(num.dX_dt).tolist(),
        "dPdt": np.array(num.dP_dt).tolist(),
        "dSdt": np.array(num.dS_dt).tolist(),
        "dVdt": np.array(num.dV_dt).tolist(),
    }
    num_dNdt_2 = [num.dX_dt_2, num.dP_dt_2, num.dS_dt_2, num.dV_dt_2]
    num_dNdt_2_dict = {
        "dXdt_2": np.array(num.dX_dt_2).tolist(),
        "dPdt_2": np.array(num.dP_dt_2).tolist(),
        "dSdt_2": np.array(num.dS_dt_2).tolist(),
        "dVdt_2": np.array(num.dV_dt_2).tolist(),
    }

    num_vals_dict = {
        "t": np.array(num.t).tolist(),
        "X": np.array(num.X).tolist(),
        "P": np.array(num.P).tolist(),
        "S": np.array(num.S).tolist(),
        "V": np.array(num.V).tolist(),
        "dNdt": num_dNdt_dict,
        "dNdt_2": num_dNdt_2_dict,
    }

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
            error_L.append(None)

    error_mad_dict = {
        "X": error_L[0],
        "P": error_L[1],
        "S": error_L[2],
        "V": error_L[3],
    }

    pinn_vals_dict = {
        # "t" = num.t, nem precisa repetir zzz
        "X": np.array(pinn_vals[0]).tolist(),
        "P": np.array(pinn_vals[1]).tolist(),
        "S": np.array(pinn_vals[2]).tolist(),
        "V": np.array(pinn_vals[3]).tolist(),
        "dNdt": pinn_dNdt_dict,
        "dNdt_2": pinn_dNdt_2_dict,
    }

    # LOSS ICSBCS
    loss_icsbcs_test = {}
    for N_index in range(len(_out.order)):
        N = _out.order[N_index]
        # As condições de contorno vem no final, e são a mesma qtde das
        # saídas. Então se vão de 0 a 4, bcs e ics são de 4-8
        index = _out.get_index_for(N) + len(_out.order)
        loss_icsbcs_test[N] = np.array(pinn.loss_history.loss_test)[
            :, index : index + 1
        ].tolist()

    # tain data
    file_dict["pinn_model"] = {
        "params": {
            "input_order": _in.order,
            "output_order": _out.order,
            "eq_params": pinn.eq_params.to_dict(),
            "process_params": pinn.process_params.to_dict(),
            "solver_params": pinn.solver_params.to_dict(),
        },
        "train": {
            "epochs": np.array(pinn.loss_history.steps).tolist(),
            "x_test": np.array(pinn.train_state.X_test).tolist(),
            "x_train": np.array(pinn.train_state.X_train).tolist(),
            "train time": pinn.total_training_time,
            "best loss test": np.array(pinn.best_loss_test).tolist(),
            "best loss train": np.array(pinn.best_loss_train).tolist(),
            "loss_history_train_SUM": [
                loss for loss in np.sum(pinn.loss_history.loss_train, axis=1).tolist()
            ],
            "loss_history_train": np.array(pinn.loss_history.loss_train).tolist(),
            "loss_history_test": np.array(pinn.loss_history.loss_test).tolist(),
            "loss_history_test_ICSBCS": loss_icsbcs_test,
            "loss_history_test_SUM": [
                loss
                for loss in np.array(
                    np.sum(pinn.loss_history.loss_test, axis=1)
                ).tolist()
            ],
        },
    }
    file_dict["pred time"] = pred_time
    file_dict["error_MAD"] = error_mad_dict
    file_dict["num_vals"] = num_vals_dict
    file_dict["pinn_vals_predicted"] = pinn_vals_dict

    # ------------------------
    # SAVING THE FILE
    path_to_file = os.path.join(folder_to_save, f"{pinn.model_name}.json")
    file = open(path_to_file, "a")
    pretty_json = json.dumps(file_dict, indent=1)
    file.write(pretty_json)
    file.close()

    units = ["g/L", "g/L", "g/L", "L"]

    if len(_in.order) >= 2:
        pinn_time_normal = pinn.train_state.X_test[:, _in.t_index : _in.t_index + 1]
    else:
        pinn_time_normal = pinn.train_state.X_test

    for i in range(4):
        items[i + 1] = {
            "title": titles[i],
            "y_label": units[i],
            "cases": [
                # Numeric
                {"x": num.t, "y": num_vals[i], "color": pinn_colors[0], "l": "-"},
            ],
        }

        derivatives[i + 1] = {
            "title": derivatives_titles[i],
            "y_label": None,
            "cases": [
                {
                    "x": num.t,
                    "y": num_dNdt[i],
                    "color": pinn_colors[0],
                    "l": "-",
                },
            ],
        }

        derivatives_2[i + 1] = {
            "title": derivatives_2_titles[i],
            "y_label": None,
            "cases": [
                {
                    "x": num.t,
                    "y": num_dNdt_2[i],
                    "color": pinn_colors[0],
                    "l": "-",
                },
            ],
        }
        if showPINN:
            # Isso é necessário para mostrar a legenda corretamente
            # Se não os valores de Y ficam como None e se não estiverem no último não
            # serão exibidos
            if pinn_vals[i] is None:
                pinn_x = 0
                pinn_y = 0
                deriv_pinn_y = 0
                deriv_pinn_y_2 = 0
            else:
                pinn_y = pinn_vals[i]
                deriv_pinn_y = pinn_derivative_vals[i]
                deriv_pinn_y_2 = pinn_derivative_2_vals[i]
                pinn_x = num.t
            items[i + 1]["cases"].append(
                # PINN
                {
                    "x": pinn_x,
                    "y": pinn_y,
                    "color": pinn_colors[1],
                    "l": "-",
                }
            )

            derivatives[i + 1]["cases"].append(
                # PINN
                {
                    "x": pinn_x,
                    "y": deriv_pinn_y,
                    "color": pinn_colors[1],
                    "l": "-",
                }
            )

            derivatives_2[i + 1]["cases"].append(
                {
                    "x": pinn_x,
                    "y": deriv_pinn_y_2,
                    "color": pinn_colors[1],
                    "l": "-",
                }
            )

        if showNondim:
            items[i + 1]["cases"].append(
                # PINN nondim
                {
                    "x": num.t,
                    "y": pinn_nondim_vals[i],
                    "color": pinn_colors[2],
                    "l": ":",
                }
            )

        if showTimeSpan:
            items[i + 1]["cases"].append(
                # Background
                {
                    "x": None,
                    "y": None,
                    "axvspan": {
                        "from": np.min(pinn_time_normal),
                        "to": np.max(pinn_time_normal),
                        "edgecolor": "lightgrey",
                        "facecolor": "white",
                        "hatch": "++",
                    },
                },
            )

            derivatives[i + 1]["cases"].append(
                # Background
                {
                    "x": None,
                    "y": None,
                    "axvspan": {
                        "from": np.min(pinn_time_normal),
                        "to": np.max(pinn_time_normal),
                        "edgecolor": "lightgrey",
                        "facecolor": "white",
                        "hatch": "++",
                    },
                },
            )

            derivatives_2[i + 1]["cases"].append(
                # Background
                {
                    "x": None,
                    "y": None,
                    "axvspan": {
                        "from": np.min(pinn_time_normal),
                        "to": np.max(pinn_time_normal),
                        "edgecolor": "lightgrey",
                        "facecolor": "white",
                        "hatch": "++",
                    },
                },
            )

    labels = ["Euler"]
    if showPINN:
        labels.append("PINN")

    if showNondim:
        labels.append("ND PINN")

    if showTimeSpan:
        labels.append("$t_{TR}$")

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

    # --------------------
    # Derivatives
    # --------------------
    if plot_derivatives:
        plot_comparer_multiple_grid(
            suptitle=f"Derivatives ({pinn.model_name})",
            labels=labels,
            figsize=(8, 6),
            gridspec_kw={"hspace": 0.042, "wspace": 0.03},
            yscale="linear",
            sharey=False,
            nrows=2,
            ncols=2,
            items=derivatives,
            title_for_each=True,
            supxlabel="t (h)",
            folder_to_save=folder_to_save,
            filename=f"DERIV-{pinn.model_name}.png" if folder_to_save else None,
            showPlot=False if folder_to_save else True,
            legend_bbox_to_anchor=(0.5, -0.1),
        )
        pass

        if plot_derivatives_2:
            plot_comparer_multiple_grid(
                suptitle=f"2 order derivatives ({pinn.model_name})",
                labels=labels,
                figsize=(8, 6),
                gridspec_kw={"hspace": 0.042, "wspace": 0.03},
                yscale="linear",
                sharey=False,
                nrows=2,
                ncols=2,
                items=derivatives_2,
                title_for_each=True,
                supxlabel="t (h)",
                folder_to_save=folder_to_save,
                filename=f"DERIV2-{pinn.model_name}.png" if folder_to_save else None,
                showPlot=False if folder_to_save else True,
                legend_bbox_to_anchor=(0.5, -0.1),
            )
        pass

    # ---------------------
    # TOTAL LOSS
    # ---------------------
    fig = plt.figure()
    plt.plot(
        pinn.loss_history.steps,
        np.sum(pinn.loss_history.loss_train, axis=1),
        linestyle="solid",
        color=pinn_colors[-3],
        label="LoT",
    )
    (line,) = plt.plot(
        pinn.loss_history.steps,
        np.sum(pinn.loss_history.loss_test, axis=1),
        color=pinn_colors[-1],
        label="LoV",
    )
    # ref: https://matplotlib.org/stable/gallery/lines_bars_and_markers/line_demo_dash_control.html
    line.set_dashes([2, 3, 5, 3])
    line.set_dash_capstyle("round")

    plt.yscale("log")
    plt.xlabel("i")
    plt.ylabel("Loss")
    plt.title("Loss x i")
    plt.legend()
    if folder_to_save:
        file_path = os.path.join(folder_to_save, f"LOSS-{pinn.model_name}.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ---------------------
    # INDIVIDUAL LOSSES
    # ---------------------
    if plot_individual_losses:
        # TRAIN
        fig = plt.figure()
        for N in _out.order:
            index = _out.get_index_for(N)
            if index is not None:
                plt.plot(
                    pinn.loss_history.steps,
                    np.array(pinn.loss_history.loss_train)[:, index : index + 1],
                    linestyle="solid",
                    label=N,
                )
        plt.yscale("log")
        plt.xlabel("i")
        plt.ylabel("Loss")
        plt.title("LoT x i")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if folder_to_save:
            file_path = os.path.join(folder_to_save, f"LoT-{pinn.model_name}.png")
        plt.savefig(file_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

        # TEST/VALIDATION LOSS
        fig = plt.figure()
        for N in _out.order:
            index = _out.get_index_for(N)
            if index is not None:
                plt.plot(
                    pinn.loss_history.steps,
                    np.array(pinn.loss_history.loss_test)[:, index : index + 1],
                    linestyle="solid",
                    label=N,
                )
        plt.yscale("log")
        plt.xlabel("i")
        plt.ylabel("Loss")
        plt.title("LoV x i")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if folder_to_save:
            file_path = os.path.join(folder_to_save, f"LoV-{pinn.model_name}.png")
        plt.savefig(file_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

        # LoV of boundary conditions
        fig = plt.figure()
        for N in loss_icsbcs_test:
            plt.plot(
                pinn.loss_history.steps,
                loss_icsbcs_test[N],
                linestyle="solid",
                label=f"${N}_0$",
            )

        plt.yscale("log")
        plt.xlabel("i")
        plt.ylabel("Loss")
        plt.title("IC LoV x i")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if folder_to_save:
            file_path = os.path.join(folder_to_save, f"LoV_IC-{pinn.model_name}.png")
        plt.savefig(file_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

    # ------------------------
    # POINTS USED IN SIMULATION
    if create_time_points_plot:
        fig = plt.figure()
        plt.plot(
            num.t,
            num.t,
            linestyle="solid",
            linewidth=7,
            color=pinn_colors[-5],
            label="$t_{SIM}$",
            zorder=1,
        )
        plt.plot(
            np.array(pinn_time_normal),
            np.array(pinn_time_normal),
            "x",
            mew=2,
            ms=4,
            alpha=0.7,
            color=pinn_colors[-1],
            label="$t_{TR}$",
            zorder=2,
        )
        plt.ylabel("t(h)")
        plt.title("Time (train) vs time (discret)")
        plt.legend()
        if folder_to_save:
            file_path = os.path.join(folder_to_save, f"TIME-{pinn.model_name}.png")
        # Save the figure
        # plt.savefig(file_path)
        plt.savefig(file_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
