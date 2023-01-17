
class PINNReactorModelResults:
    def __init__(
        self,
        model,
        loss_history,
        train_state,
        solver_params,
        eq_params,
        process_params,
        initial_state,
        f_out_value_calc,
    ):
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
        """
        Function used to calculate the reactor's outlet flow (volume/time) 
        """