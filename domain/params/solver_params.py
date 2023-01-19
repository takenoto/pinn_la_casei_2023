from domain.optimization.non_dim_scaler import NonDimScaler


class SolverLBFGSParams:
    """
    Parâmetros especificamente relacionados à aplicação do solver L-BFGS
    do_pre_optimization = irá passar lbfgs antes de adam
    do_post_optimization = irá passar lbfgs após adam
    """
    def __init__(self, do_pre_optimization=True, do_post_optimization=False):
        self.do_pre_optimization = do_pre_optimization
        self.do_post_optimization  = do_post_optimization

class SolverParams:
    def __init__(
        self,
        num_domain,
        num_boundary,
        num_test,
        adam_epochs,
        adam_display_every,
        adam_lr,
        l_bfgs:SolverLBFGSParams,
        layer_size,
        activation,
        initializer,
        loss_weights=[1,1,1,1],
        non_dim_scaler:NonDimScaler=None,
    ):
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_test = num_test
        self.adam_epochs = adam_epochs
        self.adam_display_every = adam_display_every
        self.adam_lr = adam_lr
        self.l_bfgs = l_bfgs if l_bfgs else SolverLBFGSParams(pre_optimization=False, post_optimization=False)
        self.layer_size = layer_size
        self.activation = activation
        self.initializer = initializer
        self.loss_weights = loss_weights
        self.non_dim_scaler = (
            non_dim_scaler if non_dim_scaler else NonDimScaler(X=1, P=1, S=1, V=1, t=1)
        )
