# python -m data.plot.plot_3d

import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./main/plotting/plot_styles.mplstyle")


def plot3D(
    data: dict,
):
    """A caller for multiple 3D plots.

    Args:
        data (dict): Dict with items and meta_params.

            - items (dict): Dict with plots.

            - meta_params (dict): Parameters like resolution tilte and labels.
    """
    # TODO implementar multiplot num exemplo simples. Como??
    meta_params_default = data["meta_params"]
    standalone_plots = data["standalone_plots"]
    plot_angles_default = meta_params_default.get("plot_angles", (30, 45, 0))
    filename_base_default = meta_params_default.get("filename_base", "")
    figsize_default = meta_params_default["figsize"]

    for pkey, standalone_plot in standalone_plots.items():
        standalone_plot_params = standalone_plot.get("meta_params", {})
        suptitle = standalone_plot_params.get("suptitle", None)
        plot_angles = standalone_plot_params.get("plot_angles", plot_angles_default)
        filename_base = standalone_plot_params.get(
            "filename_base", filename_base_default
        )
        figsize = standalone_plot_params.get("figsize", figsize_default)

        # Uma fig pra cada standalone plot
        fig = plt.figure(figsize=figsize)  # constrained_layout=True)
        gridspec_create = standalone_plot_params.get("gridspec_create", None)
        gs = None
        if gridspec_create:
            gs = gridspec_create(fig)

        internal_figures = standalone_plot["internal_figures"]

        axes = []
        for figkey, internal_figure in internal_figures.items():
            # Um .ax pra cada internal_figures
            if gs:
                subplot_gridspec_callback = internal_figure["subplot_gridspec_callback"]
                ax = fig.add_subplot(subplot_gridspec_callback(gs), projection="3d")
            else:
                ax = fig.add_subplot(projection="3d")
            axes.append(ax)

            items = internal_figure["items"]
            for itemkey, item in items.items():
                plot_type = item["plot_type"]  # scatter // surface

                if plot_type == "scatter":
                    _plot3DScatter(ax=ax, item=item)
                pass

            title = internal_figure.get("title", None)
            xlabel = standalone_plot_params.get("xlabel", None)
            ylabel = standalone_plot_params.get("ylabel", None)
            zlabel = standalone_plot_params.get("zlabel", None)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if zlabel:
                ax.set_zlabel(zlabel)

        if suptitle:
            fig.suptitle(suptitle)

        # Só exibe as legendas se tiver especificado ncols delas
        path_to_save = standalone_plot_params.get("path_to_save", None)
        label_ncols = meta_params_default.get("label_ncols", None)
        if label_ncols:
            # ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
            plt.legend(
                bbox_to_anchor=(0, -1.3),
                ncol=label_ncols,
                loc="upper center",  # (0.5,-0.2)
            )
        # from https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
        # plt.subplots_adjust(bottom=0.1, top=0.9)

        for plot_angle in plot_angles:
            elev, azim, roll = plot_angle
            for ax in axes:
                ax.view_init(elev=elev, azim=azim, roll=roll)
                ax.legend()
            if path_to_save:
                fig_path = os.path.join(
                    path_to_save, f"{filename_base} a{elev}-{azim}-roll.png"
                )
                plt.savefig(
                    fig_path,
                    bbox_inches="tight",
                    dpi=600,
                )
            else:
                plt.show()
            pass
        axes.clear()

        plt.close(fig)


def _plot3DScatter(ax, item):
    # TODO faz um scatter simples e já começa a pensar em quais vão ser os parâmetros
    # e em como permitir plotar o scatter e a surface juntos...
    plot_data = item["plot_data"]
    xs, ys, zs = plot_data["xs"], plot_data["ys"], plot_data["zs"]
    label = item.get("label", None)

    plot_style = item.get("plot_style", {})

    s = plot_style.get("s", 12)
    c = plot_style.get("c", None)
    marker = plot_style.get("marker", None)
    cmap = plot_style.get("cmap", None)
    norm = plot_style.get("norm", None)
    linewidths = plot_style.get("linewidths", None)
    vmin = plot_style.get("vmin", None)
    vmax = plot_style.get("vmax", None)
    alpha = plot_style.get("alpha", None)
    edgecolors = plot_style.get("edgecolors", None)

    if label:
        ax.scatter(
            xs,
            ys,
            zs,
            label=label,
            s=s,
            c=c,
            marker=marker,
            cmap=cmap,
            norm=norm,
            linewidths=linewidths,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            edgecolors=edgecolors,
        )
    else:
        ax.scatter(
            xs,
            ys,
            zs,
            label=label,
            s=s,
            c=c,
            marker=marker,
            cmap=cmap,
            norm=norm,
            linewidths=linewidths,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            edgecolors=edgecolors,
        )
    pass


# from https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin) * np.random.rand(n) + vmin


def test():
    current_directory_path = os.getcwd()
    path_to_save = os.path.join(
        current_directory_path, "data", "plot_caller", "plot_caller_output"
    )

    # default arguments:
    meta_params = {
        "suptitle": "Test plot",
        "path_to_save": path_to_save,
        "xlabel": "NL",
        "ylabel": "HL",
        "zlabel": "LoV",
        "figsize": (7, 7),
        "label_ncols": 4,  # Nº cols of the legend
        # path_to_save (str, optional): if None, the plots will be shown instead of
        # saved. Defaults to None.
        "plot_angles": [
            # (elev, azim, roll)
            (35, 45, 0),
            # (35, 78, 0),
            # (35, 115, 0),
            (35, 135, 0),
            # (35, 180, 0),
            (35, 225, 0),
            (35, 315, 0),
        ],
    }

    # Conjunto
    # Figuras avulsas
    # - Lista de figuras internas
    # - Cada uma pode ter um ou + data_points e tipo de plots então são 4 níveis...

    # Conjunto
    data = {
        # -----
        # Figuras avulsas cada uma um id
        "meta_params": meta_params,
        "standalone_plots": {
            1: {
                "meta_params": {
                    "suptitle": "Figure Suptitle TESTING",
                    "figsize": (7, 7),
                    "filename_base": "MY CRAZY FIG",
                    "path_to_save": path_to_save,
                    "gridspec_create": lambda fig: fig.add_gridspec(2, 2),
                },
                "internal_figures": {
                    # Todos esses plots vão num gráfico só!
                    "loss v4": {
                        "title": "Part 4",
                        "subplot_gridspec_callback": lambda gs: gs[1, 1],
                        "items": {
                            1: {
                                "plot_type": "scatter",
                                "label": "LoV part4",
                                "plot_data": {
                                    "xs": randrange(25, -50, 100),
                                    "ys": randrange(25, -2, 100),
                                    "zs": randrange(25, 5, 25),
                                },
                            }
                        },
                    },
                    "loss v3": {
                        "title": "Part 3",
                        "subplot_gridspec_callback": lambda gs: gs[0, 0],
                        "items": {
                            1: {
                                "plot_type": "scatter",
                                "label": "LoV part3",
                                "plot_data": {
                                    "xs": randrange(25, -50, 100),
                                    "ys": randrange(25, -2, 100),
                                    "zs": randrange(25, 5, 25),
                                },
                            }
                        },
                    },
                    "loss v2": {
                        "title": "Part 2",
                        "subplot_gridspec_callback": lambda gs: gs[1, 0],
                        "items": {
                            1: {
                                "plot_type": "scatter",
                                "label": "LoV part2",
                                "plot_data": {
                                    "xs": randrange(25, 20, 200),
                                    "ys": randrange(25, -20, 12),
                                    "zs": randrange(25, 72, 83),
                                },
                                "plot_style": {"c": "pink"},
                            }
                        },
                    },
                    "loss v1": {
                        "title": "FIGURE 3D COOL",
                        "subplot_gridspec_callback": lambda gs: gs[0, 1],
                        # Lista de figuras internas
                        "items": {
                            1: {
                                "plot_type": "scatter",
                                "label": "LoV1",
                                "plot_data": {
                                    "xs": randrange(25, 50, 50),
                                    "ys": randrange(25, -20, 12),
                                    "zs": randrange(25, 72, 83),
                                },
                                "plot_style": {
                                    "s": randrange(
                                        25, 6, 150
                                    ),  # [], #array de mesmo tamanho que xs OU escalar simples
                                    "c": "grey",
                                    "marker": "^",
                                    # "cmap": None,
                                    # "norm": None,
                                    # "vmin": None,
                                    "linewidths": None,
                                    # "vmax": None,
                                    # "alpha": None,
                                    # "edgecolors": None,
                                },
                            },
                            2: {
                                "plot_type": "scatter",
                                "label": "LoV2",
                                "plot_data": {
                                    "xs": randrange(25, -30, 0),
                                    "ys": randrange(25, -20, 12),
                                    "zs": randrange(25, 72, 83),
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    plot3D(data=data)
    pass


if __name__ == "__main__":
    test()
