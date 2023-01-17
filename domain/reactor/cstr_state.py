class CSTRState:
    def __init__(self, volume, X, P, S):
        """
        Define o volume da fase líquida e as concentrações
        de X, P e S no CSTR.
        """
        self.volume = np.array([volume])
        self.X = np.array([X])
        self.P = np.array([P])
        self.S = np.array([S])