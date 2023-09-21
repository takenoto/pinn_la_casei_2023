import os
import numpy as np
from textwrap import wrap
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from data.sci_tick_formatter import SciFormatter


plotTickFormatter = FuncFormatter(SciFormatter)


def plot_comparer_multiple_grid(
    nrows,
    ncols,
    items={},
    suptitle=None,
    x="x",
    y_keys=["y"],
    title_key="title",
    color_key="color",
    title_for_each=True,
    gridspec_kw={"hspace": 0.3, "wspace": 0.09},
    supxlabel=None,
    supylabel=None,
    sharey=True,
    sharex=True,
    yscale="log",
    figsize=(5.5, 3.5),
    labels=None,  # lista estilo ['a', 'b', 'c']
    folder_to_save=None,
    filename=None,
    showPlot=False,
    legend_bbox_to_anchor=(0.5, -0.07),
):
    """
    x and y are the keys to access the x and y values in the dictionary

    suptitle is a string
    -
    Plot comparer for 4 items
    items = {
        1: {x, y, color},
        2: {x, y, color},
        #etc
    }
    Tem que ser uma lista pra que eu saiba a ordem
    Pode ter uma cor pra cada valor de y, mas no caso a lista tem que ter a mesma
    dimensão do vetor y ou apenas 1.
    Não é possível especificar cor só pra 2 de 3 por exemplo.

    yscale pode ser 'log' ou 'linear' por exemplo. Mais em # https://matplotlib.org/3.1.3/gallery/pyplots/pyplot_scales.html


    filename
    Se for != None já vai salvar no caminho especificado ao invés de exibir

    showPlot
    caso filename seja None, SEMPRE IRÁ PLOTAR independentemente da configuração
    """
    i = items

    fig, axes_normal = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        gridspec_kw=gridspec_kw,
        sharex=sharex,
        sharey=sharey,
        layout="constrained",
    )

    if suptitle:
        sssuptitle = "\n".join(wrap(suptitle, 80))
        fig.suptitle("\n\n" + sssuptitle, y=-0.07)  # , y=-0.1, wrap=False, )#, y=0.14)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)

    # from https://stackoverflow.com/questions/37967786/axes-from-plt-subplots-is-a-numpy-ndarray-object-and-has-no-attribute-plot

    axes = axes_normal.flat  # Axes flattened in a list
    for s in range(len(axes)):
        ax = axes[s]
        has_title = i[s + 1].get(title_key, None)
        if has_title:
            # ref https://stackoverflow.com/questions/15740682/wrapping-long-y-labels-in-matplotlib-tight-layout-using-setp
            tttitle = "\n".join(wrap(i[s + 1][title_key], 16))
            ax.set_title(tttitle)

        # Se tiver a key 'cases', então os ys e xs estão vindo em pares
        # (o x não é o mesmo pra todos)

        # Lims Y for current axis
        lowestY = None
        biggestY = None

        if "cases" in i[s + 1].keys():
            for d in i[s + 1]["cases"]:
                # Itera lista de cases, plotando x, y e color
                ___color = d.get("color", "b")
                ___x = d["x"]
                ___y = d["y"]

                if ___y is not None:
                    if lowestY is None:
                        lowestY = np.min(___y)
                    else:
                        newLowY = np.min(___y)
                        lowestY = np.min([lowestY, newLowY])

                if ___x is not None:
                    if biggestY is None:
                        biggestY = np.max(___y)
                    else:
                        newBigY = np.max(___y)
                        biggestY = np.max([biggestY, newBigY])

                ___line_args = d.get("l", "None")
                ___marker = d.get("marker", None)
                ___axvspan = d.get("axvspan", None)
                if ___x is None or ___y is None:
                    pass
                else:
                    ax.plot(
                        ___x,
                        ___y,
                        linestyle=___line_args,
                        color=___color,
                        marker=___marker,
                        markersize=3,
                    )

                if ___axvspan is not None:
                    ax.axvspan(
                        ___axvspan["from"],
                        ___axvspan["to"],
                        edgecolor=___axvspan.get("edgecolor", None),
                        facecolor=___axvspan.get("facecolor", None),
                        hatch=___axvspan.get("hatch", "X"),
                    )
                pass
            pass

        ax_y_label = i[s + 1].get("y_label", None)
        ax_x_label = i[s + 1].get("x_label", None)
        ax_yscale = i[s + 1].get("ax_yscale", None)
        ax_nbinsx = i[s + 1].get("nbinxs", None)
        ax_nbinsy = i[s + 1].get("nbinsy", None)
        y_majlocator = i[s + 1].get("y_majlocator", None)
        y_minlocator = i[s + 1].get("y_minlocator", None)
        if ax_y_label:
            ax.set_ylabel(ax_y_label)
        if ax_x_label:
            ax.set_xlabel(ax_x_label)
        if ax_yscale:
            ax.set_yscale(ax_yscale)
        if ax_nbinsx:
            ax.locator_params(nbins=ax_nbinsx, axis="x")
        if ax_nbinsy:
            ax.locator_params(nbins=ax_nbinsy, axis="y")
        if y_majlocator:
            ax.yaxis.set_major_locator(y_majlocator)
        if y_minlocator:
            ax.yaxis.set_minor_locator(y_minlocator)

        # Set lims Y na força
        diffY = biggestY - lowestY
        average = (biggestY + lowestY) / 2
        diffYPerc = (biggestY - lowestY) / (biggestY)

        # Se diff < 1% força pra não ficar tão ruim de ler,
        # ou se a dif absoluta for menor que 0.001
        if diffY < 0.005:
            ax.set_ylim(top=average + 0.005, bottom=average - 0.005)
        if diffYPerc <= 0.03:
            ax.set_ylim(top=biggestY  +  0.03*average, bottom=lowestY - average*0.03)
            if np.abs(lowestY < 0.01) or np.abs(biggestY) < 0.01:
                ax.yaxis.set_major_formatter(plotTickFormatter)

        pass

    if yscale:
        plt.yscale(yscale)
    if labels:
        fig.legend(
            labels,
            loc="lower center",
            # loc="best",#"upper right",#"upper center",
            bbox_to_anchor=legend_bbox_to_anchor,  # bbox_to_anchor=(1,-0.1),
            ncol=len(labels),
            # bbox_transform=fig.transFigure,
        )

    # plt.tight_layout()
    if filename:
        file_path = filename
        if folder_to_save:
            file_path = os.path.join(folder_to_save, filename)
        # Save the figure
        # plt.savefig(file_path)
        plt.savefig(file_path, bbox_inches="tight", dpi=600)
        plt.close(fig)
        if showPlot:
            plt.show()
    else:
        plt.show()
    return
