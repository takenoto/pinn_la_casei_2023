import numpy as np

from main.plotting.plot_pinn_3d_arg import PlotPINN3DArg

from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults
from domain.params.solver_params import SolverParams
from domain.optimization.non_dim_scaler import NonDimScaler


def get_y(
    p: PINNReactorModelResults,
    as_string=True,
    getter=lambda p: p.solver_params.adam_epochs,
):
    # _y = p.solver_params.adam_epochs
    # _y = p.solver_params.adam_epochs
    if as_string:
        return f"{(np.array([getter(p)])).item()}"
    return getter(p)


def get_x(
    p: PINNReactorModelResults,
    as_string=True,
    getter=lambda p: p.solver_params.non_dim_scaler.t_not_tensor,
):
    # _x = p.solver_params.non_dim_scaler.t_not_tensor
    if as_string:
        return f"{(np.array([getter(p)])).item()}"
    return getter(p)


def get_z_value(
    p: PINNReactorModelResults, as_string=False, getter=lambda p: p.best_loss_test
):
    # z = p.best_loss_test
    if as_string:
        return f"{(np.array([getter(p)])).item()}"
    return getter(p)


def get_ts_step_loss_as_xyz(pinns: PINNReactorModelResults = None):
    """
    If {pinns} is None, will return a default value.
    """

    if pinns is None:
        pinns = [
            PINNReactorModelResults(
                solver_params=SolverParams(
                    adam_epochs=i, non_dim_scaler=NonDimScaler(t=t)
                ),
                best_loss_test=t * 1 / i,
            )
            # for i in range(100, 10000, 20)
            for i in range(100, 10000, 500)
            for t in (0.1, 40, 3)
        ]

    # ------------------------------
    unique_xs = []
    unique_ys = []
    unique_xs_nums = []
    unique_ys_nums = []
    z_dict = {}

    # 1º determina todos os uniques x e y
    for p in pinns:
        _y = get_y(p)
        _x = get_x(p)
        z_value_of_interest = get_z_value(p)

        if _y not in unique_ys:
            unique_ys = np.append(unique_ys, _y)
            unique_ys_nums = np.append(unique_ys_nums, get_y(p, as_string=False))
            z_dict[_y] = {}

        if _x not in unique_xs:
            unique_xs = np.append(unique_xs, _x)
            unique_xs_nums = np.append(unique_xs_nums, get_x(p, as_string=False))
        
        z_dict[_y][_x] = np.append(z_dict[_y].get(_x, []),z_value_of_interest)


    # Determina o z para todos os dados, já agrupados
    zs = np.ones((len(unique_ys), len(unique_xs)))
    for y_index in range(len(unique_ys)):
        _y = unique_ys[y_index]
        for x_index in range(len(unique_xs)):
            _x = unique_xs[x_index]
            zs[y_index][x_index] = z_dict[_y][_x][0]


    return unique_xs_nums, unique_ys_nums, zs
