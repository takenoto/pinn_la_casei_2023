from typing import Self
from matplotlib.transforms import ScaledTranslation
import numpy as np
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

    def __init__(self, X=1, P=1, S=1, V=1, t=1, toNondim=None, fromNondim=None):
        self.X = tf.cast(X, tf.float32)
        self.P = tf.cast(P, tf.float32)
        self.S = tf.cast(S, tf.float32)
        self.V = tf.cast(V, tf.float32)
        self.t = tf.cast(t, tf.float32)

        self.X_not_tensor = X
        self.P_not_tensor = P
        self.S_not_tensor = S
        self.V_not_tensor = V
        self.t_not_tensor = t

        self._toNondim = toNondim
        """
        _toNondim = _fromNondim(N, type, scaler)
        deve receber o próprio scaler pra poder usá-lo
        """
        self._fromNondim = fromNondim
        """
        _fromNondim = _fromNondim(N, type, scaler)
        deve receber o próprio scaler pra poder usá-lo
        """

        self.scalers = {"X": X, "P": P, "S": S, "V": V, "t": t}

        self.derivativesTypeMatcher = {
            "dXdt": "X",
            "dPdt": "P",
            "dSdt": "S",
            "dVdt": "V",
        }
        """
        Existe com o único intuito de permitir menos código ao fazer as conversões das derivadas
        """

    def fromNondim(self, N, type):
        if self._fromNondim:
            return self._fromNondim(N, type, self)
        else:
            return self.fromNondimLinearScaler(N, type)

    def toNondim(self, N, type):
        if self._toNondim:
            return self._toNondim(N, type, self)
        else:
            return self.toNondimLinearScaler(N, type)

    def fromNondimLinearScaler(self, N, type):
        assert self.scalers[type] is not None
        return N * self.scalers[type]

    def toNondimLinearScaler(self, N, type):
        assert self.scalers[type] is not None
        return N / self.scalers[type]

    # def fromNondimDerivative(self, N, type):
    #     return self.fromNondim(N, type, self)

    # def toNondimDerivative(self, N, type):
    #     return self.toNondim(N, type, self)
    
    #------------------------------------------------------------------------
    # Adimensionalização proposta pelo fernando, baseada num desvio
    # N = Ns (1 - N_nd)
    # N_nd = 1 - N/Ns
    # dN/dt = - Ns * dN_nd/dt
    # dN_nd/dt = - (dN/dt) / Ns

    # Onde nd => nondim // s => scaler // N => variável de interesse
    def toNondimDesvio(self, N, type, scaler):
        if type in scaler.scalers:
            return 1 - N / scaler.scalers[type]
        # Se não for XPSV, já assume que é uma das derivadas
        else:
            assert (
                scaler.derivativesTypeMatcher[type] is not None
            ), "type must be declared"
            # Pega o tipo da derivada. Ex: "X" de "dXdt"
            variableType = scaler.derivativesTypeMatcher[type]
            return -N / scaler.scalers[variableType]

    def fromNondimDesvio(self, N, type, scaler):
        if type in scaler.scalers:
            return scaler.scalers[type] * (1 - N)
        # Se não for XPSV, já assume que é uma das derivadas
        else:
            assert (
                scaler.derivativesTypeMatcher[type] is not None
            ), "type must be declared"
            # Pega o tipo da derivada. Ex: "X" de "dXdt"
            variableType = scaler.derivativesTypeMatcher[type]
            return -scaler.scalers[variableType] * N


def test():
    scaler = NonDimScaler()
    assert scaler.X == 1
    assert scaler.P == 1
    assert scaler.S == 1
    assert scaler.V == 1

    original_value = 5
    
    #------------------------------------------------------------------------
    #------------------------LINEAR------------------------------------------
    #------------------------------------------------------------------------

    # Testando a linear
    nondim = scaler.toNondimLinearScaler(original_value, "V")
    normal = scaler.fromNondimLinearScaler(nondim, "V")
    assert np.isclose(normal, original_value)

    # Testando o padrão (que é a linear)
    nondim = scaler.toNondim(original_value, "V")
    normal = scaler.fromNondim(nondim, "V")
    assert np.isclose(normal, original_value)


    
    
    #------------------------------------------------------------------------
    #------------------------DESVIO------------------------------------------
    #------------------------------------------------------------------------
    # Testando a adimensionalização proposta pelo fernando, baseada num desvio     
    # Valores referência pra comparar:
    All_s = 5 # Scalers de todos
    All_normal = 10
    All_nondim = -1
    scaler = NonDimScaler(
        X=All_s, P=All_s, S=All_s, V=All_s, t=All_s, toNondim=scaler.toNondimDesvio, fromNondim=scaler.fromNondimDesvio
    )
    for N_type in scaler.scalers:
        type = N_type 
        N_nondim_calc = scaler.toNondim(All_normal, type)
        assert np.isclose(N_nondim_calc, All_nondim), "Conversão de normal pra nondim"
        N_normal_calc = scaler.fromNondim(All_nondim, type)
        assert np.isclose(N_normal_calc, All_normal), "Conversão de nondim pra normal"
    
    # TODO agora testar a derivada

    # Testando adimensionalização por raiz quadrada do valor após padronização
    # N = sqrt[(N_ND*N_M)²]

    print("SUCCESS")


if __name__ == "__main__":
    test()
