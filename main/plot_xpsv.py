import matplotlib.pyplot as plt


def multiplot_xpsv(
    t: list,
    suffix: list,
    X=None,
    P=None,
    S=None,
    V=None,
    scaler=None,
):
    """
    Suffix é uma lista com suffixe, deve ter as mesas dimensões de X, P, S e V
    É o nome que será colocado após cada variável para comparação
    """
    for i in range(len(t)):
        plot_xpsv(
            t[i],
            X[i] if X else None,
            P[i] if P else None,
            S[i] if S else None,
            V[i] if V else None,
            scaler[i] if scaler else None,
            suffix=suffix[i],
            show=False,
        )
    plt.legend()
    plt.show()
    pass


def plot_xpsv(t, X, P, S, V, scaler, suffix: str, show=True):
    if scaler is None:
        if X is not None:
            plt.plot(t, X, label=f"X_{suffix}")
        if P is not None:
            plt.plot(t, P, label=f"P_{suffix}")
        if S is not None:
            plt.plot(t, S, label=f"S_{suffix}")
        if V is not None:
            plt.plot(t, V, label=f"V_{suffix}")
    else:
        if X is not None:
            plt.plot(
                t * scaler.t_not_tensor, X * scaler.X_not_tensor, label=f"X_{suffix}"
            )
        if P is not None:
            plt.plot(
                t * scaler.t_not_tensor, P * scaler.P_not_tensor, label=f"P_{suffix}"
            )
        if S is not None:
            plt.plot(
                t * scaler.t_not_tensor, S * scaler.S_not_tensor, label=f"S_{suffix}"
            )
        if V is not None:
            plt.plot(
                t * scaler.t_not_tensor, V * scaler.V_not_tensor, label=f"V_{suffix}"
            )
    if show:
        plt.legend()
        plt.show()
