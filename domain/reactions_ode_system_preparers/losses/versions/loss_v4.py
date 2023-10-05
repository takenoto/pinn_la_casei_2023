import tensorflow as tf 

def lossV4(o, args):
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
        # if X<0, return X
        # else if X>Xm, return X
        # else return loss_X
        loss_maxmin = tf.where(
            tf.less(X, 0),
            X,
            tf.where(tf.greater(X, Xm), X - Xm, tf.zeros_like(X)),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_X = loss_derivative_abs + loss_maxmin_abs
        return loss_X

    if o == "P":
        deriv_calc = V * rP + (f_in * inlet.P - f_out * P)
        loss_derivative = dPdt * V - deriv_calc
        loss_maxmin = tf.where(
            tf.less(P, 0),
            P,
            tf.where(tf.greater(P, Pm), P - Pm, tf.zeros_like(P)),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_P = loss_derivative_abs + loss_maxmin_abs
        return loss_P

    if o == "S":
        deriv_calc = V * rS + (f_in * inlet.S - f_out * S)
        loss_derivative = dSdt * V - deriv_calc
        loss_maxmin = tf.where(
            tf.less(S, 0),
            S,
            tf.where(
                tf.greater(S, initial_state.S[0]),
                S - initial_state.S[0],
                tf.zeros_like(S),
            ),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_S = loss_derivative_abs + loss_maxmin_abs
        return loss_S

    if o == "V":
        dVdt_calc = f_in - f_out
        loss_derivative = dVdt - dVdt_calc
        loss_maxmin = tf.where(
            tf.less(V, 0),
            V,
            tf.zeros_like(V),
        )
        loss_derivative_abs = tf.abs(loss_derivative)
        loss_maxmin_abs = tf.abs(loss_maxmin)
        loss_V = loss_derivative_abs + loss_maxmin_abs
        return loss_V
    pass
