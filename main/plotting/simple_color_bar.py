# from https://matplotlib.org/stable/gallery/axes_grid1/simple_colorbar.html#sphx-glr-gallery-axes-grid1-simple-colorbar-py


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def plot_simple_color_bar(title=None, x_label=None, y_label=None):
    ax = plt.subplot()
    im = ax.imshow(np.arange(100).reshape((10, 10)))
    print('IM!!!!!!!!!!')
    print(im)

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    plt.colorbar(im)
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    # plt.colorbar(im, cax=cax)

    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.show()