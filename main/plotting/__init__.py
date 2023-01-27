# ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

import matplotlib.pyplot as plt


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
    gridspec_kw={"hspace": 0.45, "wspace": 0.3},
    supxlabel=None,
    supylabel=None,
    sharey=True,
    sharex=True):
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
    Pode ter uma cor pra cada valor de y, mas no caso a lista tem que ter a mesma dimensão do vetor y ou apenas 1.
    Não é possível especificar cor só pra 2 de 3 por exemplo.
    """
    i = items

    fig, axes_normal = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw=gridspec_kw, sharex=sharex, sharey=sharey)

    if suptitle:
        fig.suptitle(suptitle, y=0.94)
        plt.subplots_adjust(top=0.8)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)

    # from https://stackoverflow.com/questions/37967786/axes-from-plt-subplots-is-a-numpy-ndarray-object-and-has-no-attribute-plot

    axes = axes_normal.flat  # Axes flattened in a list
    for s in range(len(axes)):
        ax = axes[s]
        ax.set_title(i[s + 1][title_key])
        _y_count = 0
        for y in y_keys:
            if isinstance(i[s + 1][color_key], list):
                if len(i[s + 1][color_key]) > 1:
                    color = i[s + 1][color_key][_y_count]
                else:
                    color = i[s + 1][color_key][0]
            else:
                color = i[s + 1][color_key]
            if(color is not None):
                ax.plot(i[s + 1][x], i[s + 1][y], color)
            else:
                ax.plot(i[s + 1][x], i[s + 1][y])
            _y_count += 1
    plt.show()
    return


if __name__ == "__main__":
    # Exemplo funcionando:
    plot_comparer_multiple_grid(
        nrows=2,
        ncols=2,
        items={
            1: {
                "x": [1, 2, 3],
                "y": [4, 5, 1],
                "color": "tab:blue",
                "title": "blue bar",
            },
            2: {
                "x": [1, 9, 3],
                "y": [3, -5, 6],
                "color": "tab:orange",
                "title": "super_laranja",
            },
            3: {
                "x": [1, -4, -10],
                "y": [4, 5, 6],
                "color": None,#"tab:red",
                "title": "meu título",
            },
            4: {
                "x": [1, 2, 1],
                "y": [7, 10, 2],
                "color": 'tab:green',
                "title": "verde",
            },
        },
        suptitle=None,  #'Super colors',
        title_for_each=True,
        gridspec_kw={"hspace": 0.4, "wspace": 0.25},
        supxlabel="Bananas",
        supylabel="Oranges",
    )
