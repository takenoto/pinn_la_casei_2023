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
        name=None,
        num_domain=None,
        num_boundary=None,
        num_test=None,
        adam_epochs=None,
        adam_display_every=None,
        sgd_epochs=None,
        adam_lr=None,
        l_bfgs:SolverLBFGSParams=SolverLBFGSParams(),
        layer_size=None,
        activation=None,
        initializer=None,
        loss_weights=[1,1,1,1],
        non_dim_scaler:NonDimScaler=None,
        mini_batch=None,
        hyperfolder=None
    ):
        self.name = name if name else None
        """
        A name that explains, defines or identify this solver_params objects
        """
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_test = num_test
        self.adam_epochs = adam_epochs
        self.adam_display_every = adam_display_every
        self.sgd_epochs = sgd_epochs
        self.adam_lr = adam_lr
        self.l_bfgs = l_bfgs if l_bfgs else SolverLBFGSParams(pre_optimization=False, post_optimization=False)
        self.layer_size = layer_size
        self.activation = activation
        self.initializer = initializer
        self.loss_weights = loss_weights
        self.non_dim_scaler = (
            non_dim_scaler if non_dim_scaler else NonDimScaler(X=1, P=1, S=1, V=1, t=1)
        )
        self.mini_batch=mini_batch
        # Hyperfolder é a pasta padrão
        # onde salvar os resultados daquele trambei
        self.hyperfolder = hyperfolder
