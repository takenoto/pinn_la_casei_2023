class NumericSolverModelResults:
    def __init__(
        self,
        model_name,
        model,
        X,
        P,
        S,
        V,
        t,
        dt,
        non_dim_scaler,
        dX_dt,
        dP_dt,
        dS_dt,
        dV_dt,
    ):
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
        self.non_dim_scaler = non_dim_scaler
        self.dX_dt = dX_dt
        self.dP_dt = dP_dt
        self.dS_dt = dS_dt
        self.dV_dt = dV_dt
