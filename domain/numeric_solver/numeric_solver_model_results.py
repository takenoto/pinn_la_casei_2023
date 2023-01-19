class NumericSolverModelResults:
    def __init__(self, model_name, model, X, P, S, V, t, dt):
        """
        model_name é algo como "euler" ou "runge_kutta"
        O resto são os parâmetros de solução numérica
        """
        self.model_name = model_name
        self.model = model
        self.X = X
        self.P = P
        self.S = S
        self.V = V
        self.t = t
        self.dt = dt
