# python -m data.plot.plot_3d

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

plt.style.use("./main/plotting/plot_styles.mplstyle")


def plot3D(
    items: dict,
    meta_params: dict,
):
    """A caller for multiple 3D plots.

    Args:
        items (dict): Dict with plots.

        meta_params(dict): Parameters like resolution tilte and labels.
    """
    figsize=meta_params["figsize"]
    fig = plt.figure(figsize=figsize)  # constrained_layout=True)
    ax = fig.add_subplot(projection="3d")

    for item_key in items:
        item = items[item_key]

        plot_type = item["plot_type"]  # scatter // surface

        if plot_type == "scatter":
            _plot3DScatter(ax=ax, item=item)
        pass

    title = meta_params.get("title", None)
    xlabel = meta_params.get("xlabel", None)
    ylabel = meta_params.get("ylabel", None)
    zlabel = meta_params.get("zlabel", None)
    label_ncols = meta_params.get("label_ncols", None)
    path_to_save = meta_params.get("path_to_save", None)

    if title:
        fig.suptitle(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)

    # ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
    plt.legend(
        bbox_to_anchor=(0.5, -0.0), ncol=label_ncols, loc="upper center"  # (0.5,-0.2)
    )
    # from https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
    plt.subplots_adjust(bottom=0.1, top=0.9)

    plot_angles = meta_params.get("plot_angles", (30, 45, 0))
    for plot_angle in plot_angles:
        elev, azim, roll = plot_angle
        ax.view_init(elev=elev, azim=azim, roll=roll)
        if path_to_save:
            plt.savefig(path_to_save + f"a{elev}-{azim}-roll" + ".png", bbox_inches="tight", dpi=600)
        else:
            plt.show()
        pass
        
        plt.close(fig)


def _plot3DScatter(ax, item):
    # TODO faz um scatter simples e já começa a pensar em quais vão ser os parâmetros
    # e em como permitir plotar o scatter e a surface juntos...
    plot_data = item["plot_data"]
    xs, ys, zs = plot_data["xs"], plot_data["ys"], plot_data["zs"]
    label = item.get("label", None)

    if label:
        ax.scatter(xs, ys, zs, label=label)
    else:
        ax.scatter(xs, ys, zs)
    pass


# from https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin) * np.random.rand(n) + vmin


def test():
    meta_params = {
        "title": "Test plot",
        "xlabel": "NL",
        "ylabel": "HL",
        "zlabel": "LoV",
        "figsize": (7, 7),
        "plot_angles": [
          # (elev, azim, roll)  
          (30, 45, 0),
          (90, 45, 0),
          (90, 125, 0),
        ],
        "label_ncols": 2,  # Nº cols of the legend
        "path_to_save": None
        # path_to_save (str, optional): if None, the plots will be shown instead of
        # saved. Defaults to None.
    }

    items = {
        1: {
            "plot_type": "scatter",
            "label": "LoV1",
            "plot_data": {
                "xs": randrange(25, 50, 50),
                "ys": randrange(25, -20, 12),
                "zs": randrange(25, 72, 83),
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
    }
    plot3D(meta_params=meta_params, items=items)
    pass


if __name__ == "__main__":
    test()
