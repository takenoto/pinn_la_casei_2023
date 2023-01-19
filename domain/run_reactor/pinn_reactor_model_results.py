
class PINNReactorModelResults:
    def __init__(
        self,
        model_name,
        model,
        loss_history,
        train_state,
        solver_params,
        eq_params,
        process_params,
        initial_state,
        f_out_value_calc,
        t,
        X,
        P,
        S,
        V,
        best_step,
        best_loss_test,
        best_loss_train,
        best_y,
        best_ystd,
        best_metrics,
    ):
        self.model_name=model_name
        self.model = model
        self.loss_history = loss_history
        self.train_state = train_state
        self.solver_params = solver_params
        """
        The solver params applied to achieve this model
        """
        self.eq_params = eq_params
        self.process_params = process_params
        self.initial_state = initial_state
        self.f_out_value_calc = f_out_value_calc

        self.t = t
        self.X = X
        self.P = P
        self.S = S

        self.best_step = best_step
        self.best_loss_test = best_loss_test
        self.best_loss_train = best_loss_train
        self.best_y = best_y
        self.best_ystd = best_ystd
        self.best_metrics = best_metrics
        """
        Function used to calculate the reactor's outlet flow (volume/time) 
        """