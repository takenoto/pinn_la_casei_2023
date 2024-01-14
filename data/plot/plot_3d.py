# python -m data.plot.plot_3d

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from data.sci_tick_formatter import SciFormatter

plt.style.use("./main/plotting/plot_styles.mplstyle")


plotTickFormatter = mtick.FuncFormatter(SciFormatter)


def plot3D(
    data: dict,
):
    """A caller for multiple 3D plots.

    Args:
        data (dict): Dict with items and meta_params.

            - items (dict): Dict with plots.

            - meta_params (dict): Parameters like resolution tilte and labels.
    """

    meta_params_default = data["meta_params"]
    standalone_plots = data["standalone_plots"]
    plot_angles_default = meta_params_default.get("plot_angles", (30, 45, 0))
    filename_base_default = meta_params_default.get("filename_base", "")
    figsize_default = meta_params_default["figsize"]
    figdpi_default = meta_params_default.get("figdpi", 600)

    for pkey, standalone_plot in standalone_plots.items():
        standalone_plot_params = standalone_plot.get("meta_params", {})
        plot_angles = standalone_plot_params.get("plot_angles", plot_angles_default)
        filename_base = standalone_plot_params.get(
            "filename_base", filename_base_default
        )
        figsize = standalone_plot_params.get("figsize", figsize_default)
        figdpi = standalone_plot_params.get("figdpi", figdpi_default)

        # Uma fig pra cada standalone plot
        fig = plt.figure(figsize=figsize)  # constrained_layout=True)

        # GRIDSPEC
        gridspec_create = standalone_plot_params.get("gridspec_create", None)
        gs = None
        if gridspec_create:
            gs = gridspec_create(fig)

        # FIGURE LABELS
        internal_figures_label_fontsize = standalone_plot_params.get(
            "internal_figures_label_font_size", 8
        )
        suptitle = standalone_plot_params.get("suptitle", None)
        supxlabel = standalone_plot_params.get("supxlabel", None)
        supylabel = standalone_plot_params.get("supylabel", None)
        fig.supylabel(supylabel, x=0.1)
        if suptitle:
            fig.suptitle(suptitle, y=1.10)
        supxlabel_delta = 0
        if supxlabel:
            supxlabel_delta = -0.1
            fig.supxlabel(supxlabel, y=supxlabel_delta)

        # Internal figures:
        internal_figures = standalone_plot["internal_figures"]

        axes = []
        for figkey, internal_figure in internal_figures.items():
            projection = internal_figure.get("projection", None)

            # Um .ax pra cada internal_figures
            if gs:
                subplot_gridspec_callback = internal_figure["subplot_gridspec_callback"]
                ax = fig.add_subplot(
                    subplot_gridspec_callback(gs), projection=projection
                )
            else:
                ax = fig.add_subplot(projection=projection)
            axes.append(ax)

            items = internal_figure["items"]
            for itemkey, item in items.items():
                plot_type = item["plot_type"]  # scatter // surface

                if plot_type == "scatter":
                    _plot3DScatter(ax=ax, item=item)
                    ax.tick_params(axis="x", labelsize=internal_figures_label_fontsize)
                    ax.tick_params(axis="y", labelsize=internal_figures_label_fontsize)
                    ax.tick_params(axis="z", labelsize=internal_figures_label_fontsize)
                elif plot_type == "contourf":
                    _plotContourf(ax=ax, item=item, fig=fig)
                    ax.tick_params(axis="x", labelsize=internal_figures_label_fontsize)
                    ax.tick_params(axis="y", labelsize=internal_figures_label_fontsize)
                elif plot_type == "heatmap":
                    _plot_heatmap(ax=ax, item=item, fig=fig)
                    ax.tick_params(axis="x", labelsize=internal_figures_label_fontsize)
                    ax.tick_params(axis="y", labelsize=internal_figures_label_fontsize)
                elif plot_type == "imshow":
                    _plot_imshow(ax=ax, item=item, fig=fig)
                    ax.tick_params(axis="x", labelsize=internal_figures_label_fontsize)
                    ax.tick_params(axis="y", labelsize=internal_figures_label_fontsize)

            title = internal_figure.get("title", None)
            xlabel = internal_figure.get("xlabel", None)
            ylabel = internal_figure.get("ylabel", None)
            zlabel = internal_figure.get("zlabel", None)
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=internal_figures_label_fontsize)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=internal_figures_label_fontsize)
            if zlabel:
                ax.set_zlabel(zlabel)

            # FIXME TODO
            ax.xaxis.set_major_formatter(plotTickFormatter)
            ax.yaxis.set_major_formatter(plotTickFormatter)
            if projection == "3d":
                ax.zaxis.set_major_formatter(plotTickFormatter)

        # Só exibe as legendas se tiver especificado ncols delas
        path_to_save = standalone_plot_params.get("path_to_save", None)
        label_ncols = meta_params_default.get("label_ncols", None)

        # from https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
        # plt.subplots_adjust(bottom=0.1, top=0.9)

        if label_ncols:
            # ref: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
            lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            # ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
            fig.legend(
                lines,
                labels,
                bbox_to_anchor=(0.5, supxlabel_delta - 0.01),
                ncol=label_ncols,
                loc="upper center",  # (0.5,-0.2))
            )

        # Variação por ângulo pra salvar
        for plot_angle in plot_angles:
            elev, azim, roll = plot_angle
            for ax in axes:
                # Apenas seta ângulos para gráficos 3D.
                # ref: https://stackoverflow.com/questions/43563244/python-check-if-figure-is-2d-or-3d
                if ax.name == "3d":
                    ax.view_init(elev=elev, azim=azim, roll=roll)

            if path_to_save:
                fig_path = os.path.join(
                    path_to_save, f"{filename_base} a{elev}-{azim}-roll.png"
                )
                plt.savefig(
                    fig_path,
                    bbox_inches="tight",
                    dpi=figdpi,
                )
            else:
                plt.show()
            pass
        axes.clear()

        plt.close(fig)


def _plot3DScatter(ax, item):
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


def _plotContourf(ax, item, fig):
    # ref: https://aleksandarhaber.com/explanation-of-pythons-meshgrid-function-numpy-and-3d-plotting-in-python/

    plot_data = item["plot_data"]
    x = plot_data["x"]
    y = plot_data["y"]
    z = plot_data["z"]

    plot_style = item.get("plot_style", {})

    colors = plot_style.get("colors", None)
    alpha = plot_style.get("alpha", None)
    cmap = plot_style.get("cmap", None)
    norm = plot_style.get("norm", None)
    vmin = plot_style.get("vmin", None)
    vmax = plot_style.get("vmax", None)
    origin = plot_style.get("origin", None)
    linewidths = plot_style.get("linewidths", None)

    contourf_plot = ax.contourf(
        x,
        y,
        z,
        colors=colors,
        alpha=alpha,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        linewidths=linewidths,
    )

    show_colorbar = plot_style.get("show_colorbar", True)
    if show_colorbar:
        fig.colorbar(contourf_plot, ax=ax)

    pass


def _plot_heatmap(ax, item, fig):
    # ref: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    plot_data = item["plot_data"]

    x_ticks = plot_data.get("x_ticks", None)
    if x_ticks:
        ax.set_x_ticks(np.arrange(x_ticks), labels=x_ticks)

    y_ticks = plot_data.get("y_ticks", None)
    if y_ticks:
        ax.set_x_ticks(np.arrange(y_ticks), labels=y_ticks)

    _kwargs = plot_data["kwargs"]

    img = ax.show(plot_data["data"], **_kwargs)
    # TODO
    pass


def _plot_imshow(ax, item, fig):
    plot_data = item["plot_data"]
    grid = plot_data["grid"]
    # methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    #        'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    #        'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    plot_style = item.get("plot_style", {})
    interpolation = plot_style.get("interpolation", None)
    cmap = plot_style.get("cmap", "viridis")

    imshow_plot = ax.imshow(grid, interpolation=interpolation, cmap=cmap)
    show_colorbar = plot_style.get("show_colorbar", True)
    if show_colorbar:
        fig.colorbar(imshow_plot, ax=ax)
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

    # FIXME remove:
    x = np.linspace(0, 3, 3)
    y = np.linspace(-6, -1, 3)
    x_contourf, y_contourf = np.meshgrid(x, y)
    z_contourf = x_contourf + y_contourf
    # print(x)
    # print(y)
    # print(xs)
    # print(ys)
    # print(z)
    # return

    # De forma geral é melhor aumentar a figura do que somente no dpi
    # Porque senão fica muito apertado e ruim de ler

    # default arguments:
    meta_params = {
        "suptitle": "Test plot",
        "path_to_save": path_to_save,
        # "supxlabel": "NL",
        # "supylabel": "HL",
        # "zlabel": "LoV",
        "figsize": (14, 14),
        "figdpi": 300,
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
                    "figsize": (20, 10),
                    "filename_base": "MY CRAZY FIG",
                    "path_to_save": path_to_save,
                    "gridspec_create": lambda fig: fig.add_gridspec(
                        2, 3, wspace=0.12, hspace=0.3
                    ),
                    "supxlabel": "NL",
                    "supylabel": "HL",
                    "internal_figures_label_font_size": 8,
                },
                "internal_figures": {
                    # Todos esses plots vão num gráfico só!
                    "imshow_test": {
                        "title": "Try imshow now!",
                        "subplot_gridspec_callback": lambda gs: gs[0, 2],
                        "xlabel": "Limão",
                        "ylabel": "Beterraba",
                        "items": {
                            1: {
                                "plot_type": "imshow",
                                "plot_data": {"grid": np.random.rand(4, 4)},
                                "plot_style": {
                                    "cmap": "viridis",
                                    "interpolation": None, #"hanning",
                                },
                            }
                        },
                    },
                    "contourf_test": {
                        "title": "Plot filled contourns",
                        "projection": None,
                        "subplot_gridspec_callback": lambda gs: gs[1, 2],
                        "xlabel": "Laranjas",
                        "ylabel": "Bananas",
                        "items": {
                            1: {
                                "plot_type": "contourf",
                                # Contourf não tem label
                                # "label": "iroiro",
                                "plot_data": {
                                    "x": x_contourf,
                                    "y": y_contourf,
                                    "z": z_contourf,
                                },
                                "plot_style": {"cmap": "plasma"},
                            }
                        },
                    },
                    "loss v4": {
                        "title": "Part 4",
                        "projection": "3d",
                        "xlabel": "Coisa1",
                        "ylabel": "Coisa2",
                        "zlabel": "Coisa3",
                        # subplot_gridspec_callback é pra dizer quais
                        # cols/rows irá ocupar
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
                            },
                            2: {
                                "plot_type": "scatter",
                                "label": "LoV surface",
                                "plot_data": {
                                    "xs": randrange(25, -50, 100),
                                    "ys": randrange(25, -2, 100),
                                    "zs": randrange(25, 5, 25),
                                },
                            },
                        },
                    },
                    "loss v3": {
                        "title": "Part 3",
                        "projection": "3d",
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
                        "projection": "3d",
                        "subplot_gridspec_callback": lambda gs: gs[1, 0],
                        "items": {
                            1: {
                                "plot_type": "scatter",
                                "label": "LoV part2",
                                "plot_data": {
                                    "xs": randrange(25, 20, 200000),
                                    "ys": randrange(25, -20, 12),
                                    "zs": randrange(25, 72, 83),
                                },
                                "plot_style": {"c": "pink"},
                            }
                        },
                    },
                    "loss v1": {
                        "title": "FIGURE 3D COOL",
                        "projection": "3d",
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
