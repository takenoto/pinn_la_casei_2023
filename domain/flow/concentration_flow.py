class ConcentrationFlow:
    def __init__(self, volume, X, P, S):
        """
        X, P e S são as concs de células, produto e substrato, respectivamente.

        volume é a vazão volumétrica na unidade definida
        Ex: se for 5, é 5 L/h
        """
        self.volume = np.array([volume])
        self.X = np.array([X])
        self.P = np.array([P])
        self.S = np.array([S])
