# ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# marker style ref : https://matplotlib.org/stable/api/markers_api.html
import os
import matplotlib.pyplot as plt
from textwrap import wrap

from data.plot.plot_comparer_multiple_grid import plot_comparer_multiple_grid




if __name__ == "__main__":
    plt.style.use("./main/plotting/plot_styles.mplstyle")

    # Exemplo funcionando:
    plot_comparer_multiple_grid(
        figsize=(7, 6.5),
        labels=["a", "b", "c", "d", "e", "f", "g"],
        yscale="linear",
        sharey=False,
        sharex=False,
        nrows=2,
        ncols=2,
        items={
            # Caso tenha vários cases, pode fazer assim:
            1: {
                "nbinsy": 20,
                "y_label": "y turbo colorido",
                "title": "multiple ys",
                "cases": [
                    # "l" são os line_args, pra dizer se é tracejado -- linha - pontilhado : etc
                    {
                        "x": [-2, 2, 3],
                        "y": [4, 5, 1.5],
                        "color": "tab:orange",
                        "l": "None",
                        "marker": "D",
                    },
                    {"x": [1, 0, 5], "y": [3.5, 4, 0.1], "color": "green", "l": "--"},
                    {"x": [1, 2, 6], "y": [3, 2, 3], "color": "r", "l": "-"},
                ],
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
                "nbinsy": 20,
                "y_label": "y turbo colorido",
                "title": "multiple ys",
                "cases": [
                    # "l" são os line_args, pra dizer se é tracejado -- linha - pontilhado : etc
                    {
                        "x": [-2, 2, 3],
                        "y": [4, 4, 4],
                        "color": "tab:orange",
                        "l": "None",
                        "marker": "D",
                    },
                    {"x": [1, 0, 5], "y": [3.99999, 4, 4.000001], "color": "green", "l": "-"},
                ],
            },
            3: {
                "x": [1, -4, -10],
                "y": [4, 5, 6],
                "color": None,  # "tab:red",
                "title": "meu título",
            },
            4: {
                "x": [1, 2, 1],
                "y": [7, 10, 2],
                "color": "tab:green",
                "title": "verde",
            },
        },
        suptitle=None,  #'Super colors',
        title_for_each=True,
        gridspec_kw={"hspace": 0.3, "wspace": 0.09},
        supxlabel="Bananas",
        supylabel="Oranges",
    )
