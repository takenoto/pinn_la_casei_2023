# from https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from main.plotting.plot_pinn_3d_arg import PlotPINN3DArg
from main.plotting.pinn_conversor import get_ts_step_loss_as_xyz, get_x, get_y, get_z_value


def plot_3d_lines(
    pinns,
):
    """
    Plot only the lines, not the surface...
    """
    # X,Y,Z = get_ts_step_loss_as_xyz(pinns)
    X = [get_x(p, as_string=False) for p in pinns]
    Y = [get_y(p, as_string=False) for p in pinns]
    Z = [get_z_value(p, as_string=False) for p in pinns]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X, Y, Z, c="red", marker="o", linewidth=1.0, markersize=2)

    plt.title("t_s, error and number of steps")
    ax.set_xlabel("t_s")
    ax.set_ylabel("steps")
    ax.set_zlabel("loss")
    plt.show()


def plot_3d_surface_ts_step_error(pinns=None):
    ts,step,Z = get_ts_step_loss_as_xyz(pinns)
    # ts =[get_x(p, as_string=False) for p in pinns]
    # step = [get_y(p, as_string=False) for p in pinns]
    # Z = [get_z_value(p, as_string=False) for p in pinns]

    plot_3d_surface(
        X=ts,
        Y=step,
        Z=Z,
        title="t_s, error and number of steps",
        x_label="t_s",
        y_label="steps",
        z_label="loss",
    )


def plot_3d_surface(X, Y, Z, title=None, x_label=None, y_label=None, z_label=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(X, Y)
    
    # cmap = 'Paired'
    # cmap = 'terrain'
    # cmap = 'Accent'
    cmap = 'viridis_r'
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
    # Inveret eixo x:
    # ax.set_xlim(np.max(X)*1.1, np.min(X)/1.1)
    # Inveret eixo y:
    ax.set_ylim(np.max(Y)*1.1, np.min(Y)/1.1)

    if title:
        plt.title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if z_label:
        ax.set_zlabel(z_label)

    plt.show()
