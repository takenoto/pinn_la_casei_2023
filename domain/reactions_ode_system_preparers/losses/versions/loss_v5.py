import tensorflow as tf

def lossV5(o, args):
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
    ) = args
    if o == "X":
        deriv_calc = V * rX + (f_in * inlet.X - f_out * X)
        loss_derivative = dXdt * V - deriv_calc
        loss_maxmin = tf.where(
            tf.less(X, 0),
            tf.math.pow(X, 3),
            tf.where(
                tf.greater(X, tf.ones_like(X) * Xm),
                X - Xm,
                tf.zeros_like(X),
            ),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_X = loss_derivative_abs + loss_maxmin_abs
        return loss_X

    elif o == "P":
        deriv_calc = V * rP + (f_in * inlet.P - f_out * P)
        loss_derivative = dPdt * V - deriv_calc
        loss_maxmin = tf.where(
            tf.less(P, 0),
            tf.math.pow(P, 3),
            tf.where(
                tf.greater(P, Pm),
                P - tf.ones_like(P) * Pm,
                tf.zeros_like(P),
            ),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_P = loss_derivative_abs + loss_maxmin_abs
        return loss_P

    elif o == "S":
        deriv_calc = V * rS + (f_in * inlet.S - f_out * S)
        loss_derivative = dSdt * V - deriv_calc
        loss_maxmin = tf.where(
            tf.less(S, 0),
            tf.math.pow(S, 3),
            tf.where(
                tf.greater(S, tf.ones_like(S) * initial_state.S[0]),
                S - initial_state.S[0],
                tf.zeros_like(S),
            ),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_S = loss_derivative_abs + loss_maxmin_abs
        return loss_S

    elif o == "V":
        dVdt_calc = f_in - f_out
        loss_derivative = dVdt - dVdt_calc
        loss_maxmin = tf.where(
            tf.less(V, 0),
            tf.math.pow(V, 3),
            tf.zeros_like(V),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_V = loss_derivative_abs + loss_maxmin_abs
        return loss_V

    else:
        assert False, "Key outside of XPSV"
