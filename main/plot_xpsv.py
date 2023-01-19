import matplotlib.pyplot as plt

def plot_xpsv(t, X, P, S, V, scaler):
    if scaler is None:
        if X is not None:
            plt.plot(t, X, label="X")
        if P is not None:
            plt.plot(t, P, label="P")
        if S is not None:
            plt.plot(t, S, label="S")
        if V is not None:
            plt.plot(t, V, label="Volume(L)")
    else:
        if X is not None:
            plt.plot(t * scaler.t_not_tensor, X*scaler.X_not_tensor, label="X")
        if P is not None:
            plt.plot(t * scaler.t_not_tensor, P*scaler.P_not_tensor, label="P")
        if S is not None:
            plt.plot(t * scaler.t_not_tensor, S*scaler.S_not_tensor, label="S")
        if V is not None:
            plt.plot(t * scaler.t_not_tensor, V*scaler.V_not_tensor, label="Volume(L)")
    plt.legend()
    plt.show()
