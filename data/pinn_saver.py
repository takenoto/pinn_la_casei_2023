import os
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer
import deepxde as dde

from data.plot.plot_comparer_multiple_grid import plot_comparer_multiple_grid

xp_colors = ["#F2545B"]
"""
Cores apra dados experimentais
"""
pinn_colors = [
    "olivedrab", # "#7293A0",
    "darkorange", # "#C2E812",
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
        plot_derivatives=True,
    ):
        "Saves the plot and json of the pinn"
        save_each_pinn(
            num_results=self.num_results,
            pinn=pinn,
            showPINN=showPINN if showPINN else self.showPINN,
            showNondim=showNondim if showNondim else self.showNondim,
            plot_derivatives=plot_derivatives,
            folder_to_save=folder_to_save,
        )


def save_each_pinn(
    num_results,
    pinn,
    showPINN=True,
    showNondim=False,
    folder_to_save=None,
    plot_derivatives=True,
    showTimeSpan=True,
    create_time_points_plot = False,
):
    # PRINTAR O MELHOR DOS PINNS
    items = {}

    # Plotar todos os resultados, um a um
    num = num_results[0]

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
            vals = np.array([[X_nondim[i], t_nondim[i]] for i in range(len(t_nondim))])
        elif _in.P:
            P_nondim = num.X
            vals = np.array([[P_nondim[i], t_nondim[i]] for i in range(len(t_nondim))])
        elif _in.S:
            S_nondim = num.S
            vals = np.array([[S_nondim[i], t_nondim[i]] for i in range(len(t_nondim))])
        elif _in.V:
            V_nondim = num.V
            vals = np.array([[V_nondim[i], t_nondim[i]] for i in range(len(t_nondim))])

    prediction = pinn.model.predict(vals)
    pred_end_time = timer()
    pred_time = pred_end_time - pred_start_time

    dNdt_keys = ["dXdt", "dPdt", "dSdt", "dVdt"]

    # Obtenção da derivada:
    # Ref: https://github.com/lululxvi/deepxde/issues/177
    # def dydx(x, y):
    #     return dde.grad.jacobian(y, x, i=0, j=0)

    N_nondim_pinn = {}
    N_nondim_pinn_derivatives = {}
    pinn_dNdt_keys = []

    # Predicting values
    if "X" in _out.order:
        pinn_dNdt_keys.append("dXdt")
        N_nondim_pinn["X"] = prediction[:, _out.X_index]
        N_nondim_pinn_derivatives["dXdt"] = pinn.model.predict(
            vals,
            operator=lambda x, y: dde.grad.jacobian(
                y, x, i=_out.X_index, j=_in.t_index
            ),
        )
    else:
        N_nondim_pinn_derivatives["dXdt"] = None
    if "P" in _out.order:
        pinn_dNdt_keys.append("dPdt")
        N_nondim_pinn["P"] = prediction[:, _out.P_index]
        N_nondim_pinn_derivatives["dPdt"] = pinn.model.predict(
            vals,
            operator=lambda x, y: dde.grad.jacobian(
                y, x, i=_out.P_index, j=_in.t_index
            ),
        )
    else:
        N_nondim_pinn_derivatives["dPdt"] = None
    if "S" in _out.order:
        pinn_dNdt_keys.append("dSdt")
        N_nondim_pinn["S"] = prediction[:, _out.S_index]
        N_nondim_pinn_derivatives["dSdt"] = pinn.model.predict(
            vals,
            operator=lambda x, y: dde.grad.jacobian(
                y, x, i=_out.S_index, j=_in.t_index
            ),
        )
    else:
        N_nondim_pinn_derivatives["dSdt"] = None
    if "V" in _out.order:
        pinn_dNdt_keys.append("dVdt")
        N_nondim_pinn["V"] = prediction[:, _out.V_index]
        N_nondim_pinn_derivatives["dVdt"] = pinn.model.predict(
            vals,
            operator=lambda x, y: dde.grad.jacobian(
                y, x, i=_out.V_index, j=_in.t_index
            ),
        )
    else:
        N_nondim_pinn_derivatives["dVdt"] = None

    N_pinn = {
        type: pinn.solver_params.non_dim_scaler.fromNondim(N_nondim_pinn, type)
        for type in N_nondim_pinn
    }
    N_pinn_derivative = {}
    for type in N_nondim_pinn_derivatives:
        if N_nondim_pinn_derivatives[type] is not None:
            N_pinn_derivative[type] = pinn.solver_params.non_dim_scaler.fromNondim(
                N_nondim_pinn_derivatives, type
            )
        else:
            N_pinn_derivative[type] = None

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
            '"pinn_x_test":' + f"{np.array(pinn.train_state.X_test).tolist()}" + ",\n",
            '"pinn_x_train":' + f"{np.array(pinn.train_state.X_train).tolist()}" + ",",
        ]
    )

    items = {}
    titles = ["X", "P", "S", "V"]
    derivatives = {}
    derivatives_titles = [
        "$dX/dt$",
        "$dP/dt$",
        "$dS/dt$",
        "$dV/dt$",
    ]
    pinn_vals = [N_pinn[type] if type in _out.order else None for type in titles]
    pinn_nondim_vals = [
        N_nondim_pinn[type] if type in _out.order else None for type in titles
    ]
    pinn_derivative_vals = [N_pinn_derivative[type] for type in dNdt_keys]
    pinn_nondim_derivative_vals = [
        N_nondim_pinn_derivatives[type] for type in dNdt_keys
    ]

    num_vals = [
        num.X,  # if _out.X else None,
        num.P,  # if _out.P else None,
        num.S,  # if _out.S else None,
        num.V,  # if _out.V else None,
    ]
    num_dNdt = [num.dX_dt, num.dP_dt, num.dS_dt, num.dV_dt]

    num_vals_json = """{
            "t":[%s],
            "X":[%s],
            "P":[%s],
            "S":[%s],
            "V":[%s],
            "dXdt_nondim":[%s],
            "dPdt_nondim":[%s],
            "dSdt_nondim":[%s],
            "dVdt_nondim":[%s]
            }""" % (
        ",".join(np.char.mod("%f", np.array(t_num_normal))),
        ",".join(np.char.mod("%f", np.array(num_vals[0]))),
        ",".join(np.char.mod("%f", np.array(num_vals[1]))),
        ",".join(np.char.mod("%f", np.array(num_vals[2]))),
        ",".join(np.char.mod("%f", np.array(num_vals[3]))),
        ",".join(np.char.mod("%f", np.array(num.dX_dt))),
        ",".join(np.char.mod("%f", np.array(num.dP_dt))),
        ",".join(np.char.mod("%f", np.array(num.dS_dt))),
        ",".join(np.char.mod("%f", np.array(num.dV_dt))),
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

    # -------------
    # ERROR XPSV
    error_lines = []
    if _out.X:
        error_lines.append(f'"X": {error_L[0]}')
    if _out.P:
        error_lines.append(f'"P": {error_L[1]}')
    if _out.S:
        error_lines.append(f'"S": {error_L[2]}')
    if _out.V:
        error_lines.append(f'"V": {error_L[3]}')

    error_lines.append(f'"Total": {np.nansum(error_L)}')

    pinn_vals_json = """{
        "t_nondim":[%s],
        "X":[%s],
        "P":[%s],
        "S":[%s],
        "V":[%s]
        }""" % (
        # Values
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
        "V":[%s],
        "dXdt":[%s],
        "dPdt":[%s],
        "dSdt":[%s],
        "dVdt":[%s]
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
        # Derivatives
        '"None"'
        if pinn_nondim_derivative_vals[0] is None
        else ",".join(
            np.char.mod(
                "%f", np.array(N_nondim_pinn_derivatives["dXdt"][:, 0]).tolist()
            )
        ),
        '"None"'
        if pinn_nondim_derivative_vals[1] is None
        else ",".join(
            np.char.mod(
                "%f", np.array(N_nondim_pinn_derivatives["dPdt"][:, 0]).tolist()
            )
        ),
        '"None"'
        if pinn_nondim_derivative_vals[2] is None
        else ",".join(
            np.char.mod(
                "%f", np.array(N_nondim_pinn_derivatives["dSdt"][:, 0]).tolist()
            )
        ),
        '"None"'
        if pinn_nondim_derivative_vals[3] is None
        else ",".join(
            np.char.mod(
                "%f", np.array(N_nondim_pinn_derivatives["dSdt"][:, 0]).tolist()
            )
        ),
    )

    # Fecha o arquivo
    # Fecha o body e fecha o error
    file.write('\n"error": {\n')
    for lll in range(len(error_lines)):
        line = error_lines[lll]
        if lll < len(error_lines) - 1:
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
                    + f"{[loss for loss in np.array(np.sum(pinn.loss_history.loss_test, axis=1))]}"  # noqa: E501
                    ",\n",
                    '"pinn_loss_story_train":'
                    + f"{[loss for loss in np.sum(pinn.loss_history.loss_train, axis=1)]}"  # noqa: E501
                    ",\n",
                    '"total_training_time":' + f"{pinn.total_training_time}" "\n",
                    "}",
                ]
            )
    file.close()

    units = ["g/L", "g/L", "g/L", "L"]

    # print("!!!!!!!!!!!!!!!!!")
    # print("!PINN TRAIN STATE X!")
    # print(np.array(pinn.train_state.X_test))
    # print(np.array(pinn.train_state.X_test)[0])
    # # Gera lista só com os primeiros:
    # print(np.array(pinn.train_state.X_test)[:,0])
    # print("----------------------------")
    pinn_time_normal = pinn.solver_params.non_dim_scaler.fromNondim(
        {"t": pinn.train_state.X_test}, "t"
    )

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

        if showPINN:
            # pinn_nondim_vals
            items[i + 1]["cases"].append(
                # PINN
                {
                    "x": num.t,
                    "y": pinn_vals[i],
                    "color": pinn_colors[1],
                    "l": "--",
                }
            )

            derivatives[i + 1]["cases"].append(
                # PINN
                {
                    "x": num.t,
                    # "y": pinn_nondim_derivative_vals[i],
                    "y": pinn_derivative_vals[i],
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
            suptitle=f"Nondim derivatives ({pinn.model_name})",
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

    # -----------------
    # LOSS
    # -----------------
    fig = plt.figure()
    plt.plot(
        pinn.loss_history.steps,
        np.sum(pinn.loss_history.loss_train, axis=1),
        linestyle="solid",
        color=pinn_colors[-3],
        label="LoT",
    )
    plt.plot(
        pinn.loss_history.steps,
        np.sum(pinn.loss_history.loss_test, axis=1),
        linestyle="dotted",
        color=pinn_colors[-1],
        label="LoV",
    )
    plt.yscale("log")
    plt.xlabel("i")
    plt.ylabel("Loss")
    plt.title("Loss x i")
    plt.legend()
    if folder_to_save:
        file_path = os.path.join(folder_to_save, f"LOSS-{pinn.model_name}.png")
    # Save the figure
    # plt.savefig(file_path)
    plt.savefig(file_path, bbox_inches="tight", dpi=600)
    plt.close(fig)

    # ------------------------
    # POINTS USED IN SIMULATION
    if create_time_points_plot:
        fig = plt.figure()
        plt.plot(
            t_num_normal,
            t_num_normal,
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
        plt.savefig(file_path, bbox_inches="tight", dpi=600)
        plt.close(fig)
