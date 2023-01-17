class NumericSolverModelResults:
    def __init__(self, model_name, model):
        """
        model_name é algo como "euler" ou "runge_kutta"
        O resto são os parâmetros de solução numérica
        """
        self.model_name = model_name
        self.model = model