import matplotlib.pyplot as plt


class XPSVPlotArg:
    def __init__(self, ls="-", color=None, linewidth=1.0, alpha=1):
        self.ls = ls
        self.color = color
        self.linewidth = linewidth
        self.alpha = alpha


def multiplot_xpsv(
    t: list,
    suffix: list,
    X=None,
    P=None,
    S=None,
    V=None,
    scaler=None,
    plot_args: list = None,
    title=None,
    x_label=None,
    y_label=None,
):
    """
    Suffix é uma lista com suffixe, deve ter as mesas dimensões de X, P, S e V
    É o nome que será colocado após cada variável para comparação
    -
    lss são os "ls" a se passar pro matplot (line_style)

    """
    for i in range(len(t)):
        plot_xpsv(
            t=t[i],
            X=X[i] if X else None,
            P=P[i] if P else None,
            S=S[i] if S else None,
            V=V[i] if V else None,
            scaler=scaler[i] if scaler else None,
            suffix=suffix[i],
            show=False,
            plot_arg=plot_args[i] if plot_args else None,
        )
    plt.legend()
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    ax = plt.subplot()
    ax.legend()
    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.05),
    #     ncol=3,
    # )
    plt.show()
    pass


def plot_xpsv(
    t, X, P, S, V, scaler, suffix: str, show=True, plot_arg: XPSVPlotArg = None
):

    alpha = plot_arg.alpha if plot_arg else 1

    if scaler is None:
        if X is not None:
            plt.plot(
                t,
                X,
                label=f"X_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
        if P is not None:
            plt.plot(
                t,
                P,
                label=f"P_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
        if S is not None:
            plt.plot(
                t,
                S,
                label=f"S_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
        if V is not None:
            plt.plot(
                t,
                V,
                label=f"V_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
    else:
        if X is not None:
            plt.plot(
                t * scaler.t_not_tensor,
                X * scaler.X_not_tensor,
                label=f"X_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
        if P is not None:
            plt.plot(
                t * scaler.t_not_tensor,
                P * scaler.P_not_tensor,
                label=f"P_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
        if S is not None:
            plt.plot(
                t * scaler.t_not_tensor,
                S * scaler.S_not_tensor,
                label=f"S_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
        if V is not None:
            plt.plot(
                t * scaler.t_not_tensor,
                V * scaler.V_not_tensor,
                label=f"V_{suffix}",
                ls=plot_arg.ls,
                color=plot_arg.color,
                linewidth=plot_arg.linewidth,
                alpha=alpha,
            )
    if show:
        plt.legend()
        plt.show()
