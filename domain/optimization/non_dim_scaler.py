import tensorflow as tf

class NonDimScaler:
    """
    Conversor para fácil adimensionalização
    Cada letra representa o fator de conversão.
    Por exemplo, "t" é o fator de conversão/scale do tempo (t)

    X --> cell
    P --> product (lactic acid)
    S --> substrate (lactose)
    t --> time
    V --> Volume

    Auto cast the values to tensor flow 32. If you need the values as numbers,
    use the _not_tensor subscript
    """
    def __init__(self, X=1, P=1, S=1, V=1, t=1):
        self.X = tf.cast(X,tf.float32)
        self.P = tf.cast(P,tf.float32)
        self.S = tf.cast(S,tf.float32)
        self.V = tf.cast(V,tf.float32)
        self.t = tf.cast(t,tf.float32)

        self.X_not_tensor = X
        self.P_not_tensor = P
        self.S_not_tensor = S
        self.V_not_tensor = V
        self.t_not_tensor = t

