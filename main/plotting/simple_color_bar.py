# from https://matplotlib.org/stable/gallery/axes_grid1/simple_colorbar.html#sphx-glr-gallery-axes-grid1-simple-colorbar-py
# https://matplotlib.org/stable/tutorials/colors/colormapnorms.html#sphx-glr-tutorials-colors-colormapnorms-py

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_simple_color_bar(
    x, y, z, title=None, x_label=None, y_label=None, z_label=None
):
    fig, ax = plt.subplots()

    cmap = 'Reds'

    # SHADING
    #"gouraud",  Faz um degradê interpolando pontos intermediários
    #"flat" -> flat normal, mas perde um ponto em cada dimensão. Não pergunte o pq.
    # Não consegui contornar.
    shading = 'nearest'
    shading = 'gouraud'
    shading = 'auto'
    shading = 'flat'

    if shading == 'flat':
        x = np.append(x, x[-1])
        y = np.append(y, y[-1])
        # z = z[:-1, :-1]
    
    c = ax.pcolormesh(
        x,
        y,
        z,
        cmap=cmap,
        shading=shading,
        vmin=np.min(z),
        vmax=np.max(z),
        # TODO quando os valores divergem muito, aplicar log ajuda:
        # Aí tem que tirar esse vmin e v max de cima ^
        # norm=colors.LogNorm(vmin=z.min(), vmax=z.max()
        # ),
    )
    if title:
        ax.set_title("test")
    fig.colorbar(c, ax=ax)
    
    if(np.min(x) != np.max(x)):
        plt.xlim(np.min(x), np.max(x))
    if(np.min(y) != np.max(y)):
        plt.ylim(np.min(y), np.max(y))

    if title:
        plt.title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if z_label:
        c.label = z_label

    plt.show()
