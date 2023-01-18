from domain.optimization.non_dim_scaler import NonDimScaler


class SolverParams:
    def __init__(
        self,
        num_domain,
        num_boundary,
        num_test,
        adam_epochs,
        adam_display_every,
        adam_lr,
        layer_size,
        activation,
        initializer,
        w_X=1,
        w_P=1,
        w_S=1,
        w_volume=1,
        non_dim_scaler:NonDimScaler=None,
    ):
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_test = num_test
        self.adam_epochs = adam_epochs
        self.adam_display_every = adam_display_every
        self.adam_lr = adam_lr
        self.layer_size = layer_size
        self.activation = activation
        self.initializer = initializer
        self.w_X = w_X
        self.w_P = w_P
        self.w_S = w_S
        self.w_volume = w_volume
        self.non_dim_scaler = (
            non_dim_scaler if non_dim_scaler else NonDimScaler(X=1, P=1, S=1, V=1, t=1)
        )
