# Plot linhas de loss por nº steps pra cada valor de t_s,
# e vice-versa

import numpy as np
import matplotlib.pyplot as plt


def plot_lines_error_compare(
    pinns,
    label_constructor=lambda value, group_name: f'{group_name} = {"{0:.1f}".format(float(value))}',
    group_by="t_s",
    y_name="best_loss_test",
    x_name="adam_epochs",
):
    """
    Group retorna o valor do agrupamento. Por exemplo, se for "t_s", vai retornar o próprio valor;
    name é o nome fornecido "t_s" por exemplo. Então fica "t_s" e um valor como 1.213123
    """
    values_dict = {}

    for i in range(len(pinns)):
        p = pinns[i]
        _y = None
        _x = None
        key = None

        # Determinar y, x e key pra essa iteração
        if y_name == "best_loss_test":
            _y = p.best_loss_test
        if x_name == "adam_epochs":
            _x = p.solver_params.adam_epochs
        if group_by == "t_s":
            key = f"{p.solver_params.non_dim_scaler.t_not_tensor}"

        # Cadastrar no dict
        if key not in values_dict:
            # 0 é x, 1 é y
            values_dict[key] = np.array([[], []])
        if key in values_dict:
            x_array = np.append(values_dict[key][0], _x)
            y_array = np.append(values_dict[key][1], _y)
            values_dict[key] = [x_array, y_array]

    fig, ax = plt.subplots()
    for key in values_dict:
        value = values_dict[key]
        x = value[0]
        y = value[1]
        ax.plot(x, y, label=label_constructor(value=key, group_name=group_by), alpha=0.5, linewidth=3.0)

    plt.title('loss vs epochs for each step')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.show()
