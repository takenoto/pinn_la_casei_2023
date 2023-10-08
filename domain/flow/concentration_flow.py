class ConcentrationFlow:
    def __init__(self, volume, X, P, S):
        """
        X, P e S são as concs de células, produto e substrato, respectivamente.

        volume é a vazão volumétrica na unidade definida
        Ex: se for 5, é 5 L/h
        """
        self.volume = volume
        self.X = X
        self.P = P
        self.S = S

    def to_dict(self):
        return {"volume": self.volume, "X": self.X, "P": self.P, "S": self.S}
