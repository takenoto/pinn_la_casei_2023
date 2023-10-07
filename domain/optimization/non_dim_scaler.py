import numpy as np


class NonDimScaler:
    """
    Conversor para fácil adimensionalização
    Cada letra representa o fator de conversão.
    Por exemplo, "t" é o fator de conversão/scale do tempo (t)

    t --> time
    X --> cell
    P --> product (lactic acid)
    S --> substrate (lactose)
    V --> Volume


    N --> Must be a dictionary with all necessary values packed as:
    {
        't': value_of_t,
        'X': value_of_X
        #etc
    }

    Auto cast the values to tensor flow 32. If you need the values as numbers,
    use the _not_tensor subscript
    """

    def __init__(
        self,
        X=1,
        P=1,
        S=1,
        V=1,
        t=1,
        toNondim=None,
        fromNondim=None,
        name="NULL!!!!",
        etc_params=None,
    ):
        self.X = X
        self.P = P
        self.S = S
        self.V = V
        self.t = t

        self.X_not_tensor = X
        self.P_not_tensor = P
        self.S_not_tensor = S
        self.V_not_tensor = V
        self.t_not_tensor = t

        self._toNondim = toNondim
        """
        _toNondim = _fromNondim(scaler ou self, N, type)
        deve receber o próprio scaler pra poder usá-lo
        """
        self._fromNondim = fromNondim
        """
        _fromNondim = _fromNondim(scaler ou self, N, type)
        deve receber o próprio scaler pra poder usá-lo
        """

        self.scalers = {"X": X, "P": P, "S": S, "V": V, "t": t}

        self.derivativesTypeMatcher = {
            "dXdt": "X",
            "dPdt": "P",
            "dSdt": "S",
            "dVdt": "V",
            "dt": "t",
        }
        """
        Existe com o único intuito de permitir menos código 
        ao fazer as conversões das derivadas
        """

        self.name = name

        self.etc_params = etc_params
        """Used when the nondim scalers functions need params, such as a lower boundary
        or upper boundary
        """

    def toDict(self):
        return {"name": self.name, "scalers": self.scalers}

    def toNondim(self, N, type):
        if self._toNondim:
            return self._toNondim(self=self, N=N, type=type)
        else:
            return self.toNondimLinearScaler(N, type)

    def fromNondim(self, N, type):
        if self._fromNondim:
            return self._fromNondim(self=self, N=N, type=type)
        else:
            return self.fromNondimLinearScaler(N, type)

    def toNondimLinearScaler(self, N, type):
        scaler = self
        if type in scaler.scalers:
            return N[type] / scaler.scalers[type]
        else:
            assert (
                scaler.derivativesTypeMatcher[type] is not None
            ), "type must be declared"
            variableType = scaler.derivativesTypeMatcher[type]
            if type == "dt":
                return N[type] / scaler.scalers[variableType]
            else:
                return N[type] * scaler.scalers["t"] / scaler.scalers[variableType]

    def fromNondimLinearScaler(self, N, type):
        scaler = self
        if type in scaler.scalers:
            return N[type] * scaler.scalers[type]
        else:
            assert (
                scaler.derivativesTypeMatcher[type] is not None
            ), "type must be declared"
            variableType = scaler.derivativesTypeMatcher[type]
            if type == "dt":
                return N[type] * scaler.scalers[variableType]
            else:
                return N[type] * scaler.scalers[variableType] / scaler.scalers["t"]

    # def fromNondimDerivative(self, N, type):
    #     return self.fromNondim(N, type, self)

    # def toNondimDerivative(self, N, type):
    #     return self.toNondim(N, type, self)

    # ------------------------------------------------------------------------
    # Adimensionalização proposta pelo fernando, baseada num desvio
    # N = Ns (1 - N_nd)
    # N_nd = 1 - N/Ns
    # dN/dt = - Ns * dN_nd/dt
    # dN_nd/dt = - (dN/dt) / Ns
    # O "t" pode ser convertido por fora
    # dt = d(ts[1-t_nd]) => dt = -ts*dt_nd
    # Daí dX/dt  = -Xs * dX_nd/dt = (-Xs/-ts )* dX_nd/dt_nd

    # Onde nd => nondim // s => scaler // N => variável de interesse
    def toNondimDesvio(self, N, type):
        scaler = self
        if type in scaler.scalers:
            return 1 - N[type] / scaler.scalers[type]
        # Se não for XPSV, já assume que é uma das derivadas
        else:
            assert (
                scaler.derivativesTypeMatcher[type] is not None
            ), "type must be declared"
            # Pega o tipo da derivada. Ex: "X" de "dXdt"
            variableType = scaler.derivativesTypeMatcher[type]
            if type == "dt":
                return -N[type] / scaler.scalers[variableType]
            return (scaler.scalers["t"] / scaler.scalers[variableType]) * N[type]

    def fromNondimDesvio(self, N, type):
        scaler = self
        if type in scaler.scalers:
            return scaler.scalers[type] * (1 - N[type])
        # Se não for XPSV, já assume que é uma das derivadas
        else:
            assert (
                scaler.derivativesTypeMatcher[type] is not None
            ), "type must be declared"
            # Pega o tipo da derivada. Ex: "X" de "dXdt"
            variableType = scaler.derivativesTypeMatcher[type]
            if type == "dt":
                return -N[type] * scaler.scalers[variableType]
            return (scaler.scalers[variableType] / scaler.scalers["t"]) * (N[type])

    def toNondimUpscale(self, N, type):
        """Scales the division between lowerbound and upperbound
        N_A = lower_bound + N/N_s

        Args:
            N (_type_): _description_
            type (_type_): _description_

        Returns:
            _type_: _description_
        """
        scaler = self
        if type in scaler.scalers:
            return (
                self.etc_params["upscale_lowerbound"] + N[type] / scaler.scalers[type]
            )

    def fromNondimUpscale(self, N, type):
        """Scales the division between lowerbound and upperbound
        N_A = lower_bound + N/N_s
        N = N_s*(N_a - lower_bound)

        Args:
            N (_type_): _description_
            type (_type_): _description_

        Returns:
            _type_: _description_
        """
        scaler = self
        if type in scaler.scalers:
            return scaler.scalers[type] * (
                N[type] - self.etc_params["upscale_lowerbound"]
            )


# -------------------------------------------------------------
# --------------------------TESTES-----------------------------
# -------------------------------------------------------------


def test():
    _test_linear()

    _test_desvio()

    test_upscale()

    print("SUCCESS")


def _test_linear():
    print("LINEAR TEST")
    # N padrão para testes:
    N = {
        "X": 10,
        "P": 15,
        "S": 7,
        "V": 5,
        "t": 1,
        "dXdt": 12,
        "dPdt": 13,
        "dSdt": 14,
        "dVdt": 15,
        "dt": 2,
    }

    scaler = NonDimScaler()
    assert scaler.X == 1
    assert scaler.P == 1
    assert scaler.S == 1
    assert scaler.V == 1
    assert scaler.t == 1

    # ------------------------------------------------------------------------
    # ------------------------LINEAR------------------------------------------
    # ------------------------------------------------------------------------

    # Convertendo dt e pegando de volta
    dt_nondim = scaler.toNondim({"dt": 5}, "dt")
    assert np.isclose(dt_nondim, 5 / scaler.t), "dt test"

    for N_type in scaler.scalers:
        # Testando a linear
        nondim = scaler.toNondimLinearScaler(N, N_type)
        normal = scaler.fromNondimLinearScaler({N_type: nondim}, N_type)
        assert np.isclose(normal, N[N_type])

        # Testando o padrão (que é a própria linear,
        # então tecnicamente estamos nos repetindo)
        nondim = scaler.toNondimLinearScaler(N, N_type)
        normal = scaler.fromNondimLinearScaler({N_type: nondim}, N_type)
        assert np.isclose(normal, N[N_type])

    # Teste simples: converter e voltar PARA AS DERIVADAS
    N_nondim_answer = {}
    for dN_type in scaler.derivativesTypeMatcher:
        N_nondim_answer[dN_type] = nondim = scaler.toNondim(N, dN_type)

    for dN_type in scaler.derivativesTypeMatcher:
        normal = scaler.fromNondim(N, dN_type)
        assert np.isclose(
            normal, N_nondim_answer[dN_type]
        ), "Conversion to nondim and back"


def _test_desvio():
    print("TESTE DESVIO")
    # ------------------------------------------------------------------------
    # ------------------------DESVIO------------------------------------------
    # ------------------------------------------------------------------------
    # Testando a adimensionalização proposta pelo fernando, baseada num desvio
    # Valores referência pra comparar:

    # ----------------------
    # UNIT
    # Se os scalers forem 1, nondim e dim deverão ser numericamente iguais
    scaler = NonDimScaler(
        X=1,
        P=1,
        S=1,
        V=1,
        t=1,
        toNondim=NonDimScaler.toNondimDesvio,
        fromNondim=NonDimScaler.fromNondimDesvio,
    )

    N = {
        "X": 15.5,
        "P": 9.99,
        "S": 1.111,
        "V": -19.5,
        "t": 5.3,
        "dXdt": -7,
        "dPdt": 8.55,
        "dSdt": 3.0,
        "dVdt": -119.123,
        "dt": 3 / 5,
    }

    for type in N:
        nondim = scaler.toNondim(N, type)
        normal = scaler.fromNondim({type: nondim}, type)
        assert np.isclose(normal, N[type]), "Converting to nondim and back"

    # ----------------------
    # FIXED VALUES
    All_s = 5  # Scalers de todos
    All_normal = 10
    All_nondim = -1
    scaler = NonDimScaler(
        X=All_s,
        P=All_s,
        S=All_s,
        V=All_s,
        t=All_s,
        toNondim=NonDimScaler.toNondimDesvio,
        fromNondim=NonDimScaler.fromNondimDesvio,
    )
    N = {}
    N_nondim_answer = {}

    for N_type in scaler.scalers:
        N[N_type] = All_normal
        N_nondim_answer[N_type] = All_nondim

    N_nondim_calc = {}

    for N_type in scaler.scalers:
        type = N_type
        N_nondim_calc[N_type] = scaler.toNondim(N, type)
        assert np.isclose(
            N_nondim_calc[N_type], N_nondim_answer[N_type]
        ), "Desvio: Conversão de normal pra nondim"
        N_normal_calc = scaler.fromNondim(N_nondim_calc, type)
        assert np.isclose(
            N_normal_calc, N[N_type]
        ), "Desvio: Conversão de nondim pra normal"

    # Teste de apenas uma derivada fixa, só pra garantir
    scaler = NonDimScaler(
        X=2,
        t=12,
        toNondim=NonDimScaler.toNondimDesvio,
        fromNondim=NonDimScaler.fromNondimDesvio,
    )
    N = {
        "X": 5,
        "dXdt": 7,
    }
    N_nondim_answer = {"X": 1 - 5 / 2, "dXdt": 7 * 12 / 2}

    for key in ["X", "dXdt"]:
        assert np.isclose(
            N_nondim_answer[key], scaler.toNondim(N, type=key)
        ), f"Testing {key} => nondim values previously calculated"
        assert np.isclose(
            N[key], scaler.fromNondim(N_nondim_answer, type=key)
        ), f"Testing nondim => {key} values previously calculated"

    # -----------------------
    # --- DYNAMIC VALUES ---
    # Tests if converting to nondim and then converting to normal gives the same value
    # since the operations should cancel each other
    scaler = NonDimScaler(
        X=2,
        P=4,
        S=6,
        V=8,
        t=12,
        toNondim=NonDimScaler.toNondimDesvio,
        fromNondim=NonDimScaler.fromNondimDesvio,
    )

    N = {
        "X": 4,
        "P": 2,
        "S": 3,
        "V": 5,
        "t": 1.5,
        "dXdt": 3,
        "dPdt": 5,
        "dSdt": 7,
        "dVdt": 9,
        "dt": 3,
    }

    N_nondim_answer = {}
    # First the answers of simple nondim variables:
    for N_type in scaler.scalers:
        N_nondim_answer[N_type] = 1 - N[N_type] / scaler.scalers[N_type]
        normal_value = scaler.fromNondim(N_nondim_answer, type=N_type)
        assert np.isclose(normal_value, N[N_type]), "Converting to nondim and back"

    ## DERIVATIVES
    for dN_type in scaler.derivativesTypeMatcher:
        # Excepcionalmente pro dt é diferente, porque é só "dt"
        if dN_type == "dt":
            N_nondim_answer[dN_type] = -N[dN_type] / scaler.scalers["t"]
        else:
            variable_type = scaler.derivativesTypeMatcher[dN_type]
            N_nondim_answer[dN_type] = (
                N[dN_type] * (scaler.scalers["t"]) / scaler.scalers[variable_type]
            )

        assert np.isclose(
            N_nondim_answer[dN_type], scaler.toNondim(N, dN_type)
        ), "Derivative to nondim"
        normal_from_nondim = scaler.fromNondim(N_nondim_answer, type=dN_type)

        assert np.isclose(
            normal_from_nondim, N[dN_type]
        ), "Derivative: Converting to nondim and back"


def test_upscale():
    print("UPSCALE TEST")
    LB = 99
    scaler = NonDimScaler(
        X=4,
        P=2,
        S=-593,
        V=5,
        t=6,
        toNondim=NonDimScaler.toNondimUpscale,
        fromNondim=NonDimScaler.fromNondimUpscale,
        etc_params={"upscale_lowerbound": LB},
    )

    N = {
        "X": 4,
        "P": 2,
        "S": -593,
        "V": 15,
        "t": 3,
    }

    toNondimAnswers = {"X": 100, "P": 100, "S": 100, "V": 102, "t": 99.5}

    # Converting forth and back:
    for N_type in scaler.scalers:
        normal_value = N[N_type]

        # Checks to nondim:
        nondimval = scaler.toNondim(N, type=N_type)
        answer = toNondimAnswers[N_type]
        assert np.isclose(answer, nondimval), "Calculating nondim"

        # Checks from nondim:
        normal_value_calc = scaler.fromNondim({N_type: nondimval}, type=N_type)
        assert np.isclose(
            normal_value, normal_value_calc
        ), "Converting to nondim and back"

    pass


if __name__ == "__main__":
    test()
