from numpy import float32
import tensorflow as tf
import deepxde as dde

def lossV7(o, args):
    (
        X,
        P,
        S,
        V,
        dXdt,
        dPdt,
        dSdt,
        dVdt,
        rX,
        rP,
        rS,
        inlet,
        f_in,
        f_out,
        Xm,
        Pm,
        initial_state,
        process_params,
        t,
        dXdt_2,
        dPdt_2,
        dSdt_2,
        dVdt_2,
    ) = args

    # ----------------------
    # calc loss maxmin
    # ----------------------
    minmax_dict = {
        # value itself, min, max
        "X": (X, 0, Xm[0]),
        "P": (P, 0, Pm[0]),
        "S": (S, 0, initial_state.S[0]),
        "V": (V, 0, process_params.max_reactor_volume),
    }

    N, Nmin, Nmax = minmax_dict[o]

    # Sempre vai sair absoluta se o limite inferior for 0
    loss_minmax = tf.where(
        tf.less(N, Nmin),
        Nmin - N,
        tf.where(
            tf.greater(N, Nmax),
            N - Nmax,
            tf.zeros_like(N),
        ),
    )

    # -------------------------------

    if o in ["X", "P", "S"]:
        # ----------------------
        # calc loss derivative
        # ----------------------
        if o == "X":
            dNdt, rN, inletN, concN = dXdt, rX, inlet.X, X
        elif o == "P":
            dNdt, rN, inletN, concN = dPdt, rP, inlet.P, P
        elif o == "S":
            dNdt, rN, inletN, concN = dSdt, rS, inlet.S, S

        # Se V for zero, usar um valor limite de 1e-7
        volume_threshold = tf.constant(1e-5)
        V_th = tf.where(
            tf.math.equal(V, tf.zeros_like(V)), tf.ones_like(V) * volume_threshold, V
        )

        # V como divisor, e não multiplicando:
        dNdt_calc = rN + (f_in * inletN - f_out * concN) / V_th

    elif o == "V":
        dNdt = dVdt
        dNdt_calc = f_in - f_out

    #
    # ----------------------
    # calc loss derivative
    # ----------------------
    loss_derivative = dNdt - dNdt_calc
    loss_derivative_abs = tf.abs(loss_derivative)
    #
    # ----------------------
    # calc loss derivative signal
    # ----------------------
    sign_deriv_pred = tf.math.sign(dNdt)
    sign_deriv_calc = tf.cast(tf.math.sign(dNdt_calc), dtype=float32)
    loss_multiplier = tf.abs(sign_deriv_pred - sign_deriv_calc)
    loss = (1 + loss_multiplier/10) * (loss_derivative_abs + loss_minmax)
    # ----------------------

    return loss
