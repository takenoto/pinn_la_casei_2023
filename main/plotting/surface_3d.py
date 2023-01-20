# from https://matplotlib.org/stable/gallery/mplot3d/surface3d.html

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

def get_y(p, as_string=True):
    # _y = p.solver_params.adam_epochs
    if as_string:
        return f'{(np.array([p.adam_epochs])).item()}'
    return p.adam_epochs
def get_x(p, as_string=True):
    # _x = p.solver_params.non_dim_scaler.t_not_tensor
    if as_string:
        return f'{(np.array([p.t_not_tensor])).item()}'
    return p.t_not_tensor
def get_z_value(p, as_string=False):
    # z = p.best_loss_test
    if as_string:
        return f'{(np.array([p.best_loss_test])).item()}'
    return p.best_loss_test
    
class TestPINNS:
    """
    Classe feita com o único intuito de poder testar essas funções.
    """
    def __init__(self, adam_epochs, best_loss_test, t_not_tensor):
        self.adam_epochs = adam_epochs
        self.best_loss_test = best_loss_test
        self.t_not_tensor = t_not_tensor

def plot_3d_lines(pinns):
    """
    Plot only the lines, not the surface...
    """
    X = [get_x(p, as_string=False) for p in pinns]
    Y = [get_y(p, as_string=False) for p in pinns]
    Z = [get_z_value(p, as_string=False) for p in pinns]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X,Y,Z, c='red', marker='o', linewidth=1.0, markersize=2)
    plt.show()

def plot_3d_surface_ts_step_error():
    
    pinns = [TestPINNS(adam_epochs=i, best_loss_test=t*1/i, t_not_tensor=t) for i in range(100, 10000, 20) for t in (1, 40, 1) ]
    

    #------------------------------
    unique_xs = []
    unique_ys = []
    unique_xs_nums = []
    unique_ys_nums = []
    z_dict = {}

    # 1º determina todos os uniques x e y
    for p in pinns:
        _y = get_y(p)
        # _x = p.solver_params.non_dim_scaler.t_not_tensor
        _x = get_x(p)
        # z = p.best_loss_test
        z_value_of_interest = get_z_value(p)

        if _y not in unique_ys:
            unique_ys = np.append(unique_ys, _y)
            unique_ys_nums = np.append(unique_ys_nums, get_y(p, as_string=False))
            z_dict[_y] = {}
        if _x in unique_xs:
            for y_key in z_dict:
                z_dict[y_key][_x] = np.append(z_dict[y_key].get(_x,[]), z_value_of_interest)
        if _x not in unique_xs:
            unique_xs = np.append(unique_xs, _x)
            unique_xs_nums = np.append(unique_xs_nums, get_x(p, as_string=False))
            for y_key in z_dict:
                # z[y][x] = valor que quero
                z_dict[y_key] = {_x:[z_value_of_interest]}


    # Determina o z para todos os dados, já agrupados
    zs = np.ones((len(unique_ys), len(unique_xs)))
    for y_index in range(len(unique_ys)):
        _y = unique_ys[y_index]
        for x_index in range(len(unique_xs)):
            _x = unique_xs[x_index]
            zs[y_index][x_index] = z_dict[_y][_x][0]


    plot_3d_surface(
        X=unique_xs_nums, Y=unique_ys_nums, Z=zs,
        title="t_s, error and number of steps",
        x_label="t_s",
        y_label="steps",
        z_label="loss",
    )

def plot_3d_surface(X, Y, Z, 
    title=None, x_label=None, y_label=None, z_label=None):
    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    # Make data.
    X, Y = np.meshgrid(X, Y)
    # X, Y = np.meshgrid([1, 2, 3], [1, 2, 3, 4 ])
    # A ordem é y, x
    # Z = np.ones((4, 3))
    # Z[y][x] = 
    # x = 0
    # y= 1 # Em X = 2 e y = 1, zera
    # Z[y][x] = -1
    # Z[0][0] = 2
    # Z[0][1] = 1
    # Z[0][2] = 0.4

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    if title:
        plt.title(title)
    if x_label:
        ax.set_xlabel(x_label)
        # plt.xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
        # plt.ylabel(y_label)
    if z_label:
        ax.set_zlabel(z_label)
        # plt.zlabel(z_label)

    plt.show()