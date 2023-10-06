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
        # 1 order derivatives
        dX_dt,
        dP_dt,
        dS_dt,
        dV_dt,
        # 2 order derivatives
        dX_dt_2,
        dP_dt_2,
        dS_dt_2,
        dV_dt_2,
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
        self.dX_dt_2 = dX_dt_2
        self.dP_dt_2 = dP_dt_2
        self.dS_dt_2 = dS_dt_2
        self.dV_dt_2 = dV_dt_2
