# ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# marker style ref : https://matplotlib.org/stable/api/markers_api.html

import matplotlib.pyplot as plt


# TODO passar fig_size
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
    yscale='log',
    figsize=(5.5, 3.5),
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
    Pode ter uma cor pra cada valor de y, mas no caso a lista tem que ter a mesma dimensão do vetor y ou apenas 1.
    Não é possível especificar cor só pra 2 de 3 por exemplo.

    yscale pode ser 'log' ou 'linear' por exemplo. Mais em # https://matplotlib.org/3.1.3/gallery/pyplots/pyplot_scales.html
    
    """
    i = items

    fig, axes_normal = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, gridspec_kw=gridspec_kw, sharex=sharex, sharey=sharey)

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

        # Se tiver a key 'cases', então os ys e xs estão vindo em pares (o x não é o mesmo pra todos)
        if 'cases' in i[s+1].keys():
            for d in i[s+1]['cases']:
                # Itera lista de cases, plotando x, y e color
                ___color = d.get('color', 'b')
                ___x = d['x']
                ___y = d['y']
                ___line_args = d.get('l', 'None')
                ___marker = d.get('marker', None)
                ax.plot(___x, ___y, linestyle=___line_args, color=___color, marker=___marker, markersize=3)

        else:
            _y_count = 0
            for y in y_keys:
                # Determina a color:
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
    if yscale:
        plt.yscale(yscale)
    plt.show()
    return


if __name__ == "__main__":
    plt.style.use("./main/plotting/plot_styles.mplstyle")

    # Exemplo funcionando:
    plot_comparer_multiple_grid(
        yscale='linear',
        nrows=2,
        ncols=2,
        items={
            # Caso tenha vários cases, pode fazer assim:
            1: {
                'title':'multiple ys',
                'cases':
                [
                    # "l" são os line_args, pra dizer se é tracejado -- linha - pontilhado : etc
                    {'x': [-2, 2, 3], 'y':[4, 5, 1.5], 'color':'tab:orange', 'l':'None', 'marker':'D'},
                    {'x': [1, 0, 5], 'y':[3.5, 4, 0.1], 'color':'green', 'l':'--'},
                    {'x': [1, 2, 6], 'y':[3, 2, 3], 'color':'r', 'l':'-'},
                ]
            },
            # Caso a keyword 'case' não exista, segue normalmente:
            # 1: {
            #     "x": [1, 2, 3],
            #     "y": [4, 5, 1],
            #     # "color": "tab:blue",
            #     'color':'o r',
            #     "title": "blue bar",
            # },
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
        gridspec_kw={"hspace": 0.3, "wspace": 0.09},
        supxlabel="Bananas",
        supylabel="Oranges",
    )
