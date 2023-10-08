import tensorflow as tf
import deepxde as dde


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
        "S": (S, 0, initial_state.S),
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
    # calc loss 1 derivative signal
    sign_dif_d1 = tf.abs(tf.math.sign(dNdt) - tf.math.sign(dNdt_calc))
    #
    # ----------------------
    # calc loss second derivative
    loss_d2 = tf.abs(dNdt_2 - dNdt_2_calc)
    #
    # ----------------------
    # calc loss 2 derivative signal
    sign_dif_d2 = tf.abs(tf.math.sign(dNdt_2) - tf.math.sign(dNdt_2_calc))
    #
    # ----------------------
    # calc loss
    # ----------------------
    # ----------------------

    match loss_version:
        case "7A":
            loss = loss_d1
        case "7B":
            loss = loss_d1 + loss_minmax
        case "7C":
            loss = loss_d2
        case "7D":
            loss = loss_d2 + loss_minmax
        case "7E":
            loss = tf.add(1, sign_dif_d1) * loss_d1
        case "7F":
            loss = tf.add(1, sign_dif_d2) * loss_d2
        case "7G":
            # (1 + sing1 + sing2) * sum loss
            loss = tf.add(tf.add(1, sign_dif_d1), sign_dif_d2) * (
                loss_d1 + loss_minmax + loss_d2
            )

    return loss
