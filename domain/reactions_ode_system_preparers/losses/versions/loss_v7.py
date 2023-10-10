from numpy import float32
import tensorflow as tf
import deepxde as dde


def sign_dif_abs(x, y):
    return tf.abs(tf.sign(x) - tf.sign(y))


def lossV7(o, args, loss_version):
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
        nn_input,  # esse "nn_input" são todas as variáveis de entrada!!!
        # É chamado de "x" na deepxde
        inputSimulationType,
    ) = args

    # ----------------------
    # calc loss maxmin
    # ----------------------
    minmax_dict = {
        # value itself, min, max
        "X": (X, 0, Xm),
        "P": (P, 0, Pm),
        "S": (S, 0, process_params.Smax if process_params.Smax else initial_state.S),
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
            tf.zeros_like(N, dtype=float32),
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

        # Se V for zero, usar um valor limite de 1e-10
        volume_threshold = tf.constant(1e-10)
        V_th = tf.where(
            tf.math.equal(V, tf.zeros_like(V)),
            tf.ones_like(V) * volume_threshold,
            V,
        )

        # V como divisor, e não multiplicando:
        dNdt_calc = rN + (f_in * inletN - f_out * concN) / V_th

    elif o == "V":
        dNdt = dVdt
        dNdt_calc = f_in - f_out

    # ----------------------
    # Second derivative
    dNdt_2 = dde.grad.hessian(N, nn_input, j=inputSimulationType.t_index, grad_y=dNdt)
    dNdt_2_calc = dde.grad.jacobian(
        tf.convert_to_tensor(dNdt_calc * tf.ones_like(N)),
        nn_input,
        j=inputSimulationType.t_index,
        i=0,
    )
    #
    # ----------------------
    # ---      LOSS      ---
    # ----------------------
    #
    # ----------------------
    # calc loss derivative
    loss_d1 = tf.abs(dNdt - dNdt_calc)
    #
    # ----------------------
    # calc loss second derivative
    loss_d2 = tf.abs(dNdt_2 - dNdt_2_calc)
    #
    # ----------------------
    #
    # ----------------------
    # calc loss
    # ----------------------
    # ----------------------

    match loss_version:
        case "7A":
            return loss_d1
        case "7B":
            return loss_d1 + loss_minmax
        case "7C":
            return loss_d2
        case "7D":
            return loss_d2 + loss_minmax
        case "7E":
            return tf.add(1.0, sign_dif_abs(dNdt, dNdt_calc)) * loss_d1
        case "7F":
            return tf.add(1.0, sign_dif_abs(dNdt_2, dNdt_2_calc)) * loss_d2
        case "7G":
            # Tudo
            return (
                sign_dif_abs(dNdt, dNdt_calc) + sign_dif_abs(dNdt_2, dNdt_2_calc) + 1.0
            ) * (loss_d1 + loss_d2 + loss_minmax)
        case "7H":
            # Sign d2 na loss d1
            return tf.add(1.0, sign_dif_abs(dNdt_2, dNdt_2_calc)) * loss_d1
        case "7I":
            # Sign d1 na loss d2
            return tf.add(1.0, sign_dif_abs(dNdt, dNdt_calc)) * loss_d2
        case "7J":
            # Tudo
            return loss_d1 + loss_d2 + loss_minmax
