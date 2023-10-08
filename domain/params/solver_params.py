# python -m domain.params.solver_params

import numpy as np
from data.pinn_saver import PINNSaveCaller
from domain.optimization.non_dim_scaler import NonDimScaler


class SystemSimulationType:
    """
    Define as variáveis de saída OU  de entrada.

    Cada variável que for True deve ser calculada.
    As demais não.
    Fim.

    Os index são para ficar mais fácil de encontrar as posições
    de cada variável no ODE_preparer
    """

    def __init__(self, supported_variables=["X", "P", "S", "V"]):
        self.t = "t" in supported_variables
        self.X = "X" in supported_variables
        self.P = "P" in supported_variables
        self.S = "S" in supported_variables
        self.V = "V" in supported_variables

        # Só pra ficar mais fácil de digitar abaixo
        t = self.t
        X = self.X
        P = self.P
        S = self.S
        V = self.V

        # Estabelece o index de cada uma, sempre seguindo um padrão,
        # que independe de "supported_variables"
        self.X_index = None
        self.P_index = None
        self.S_index = None
        self.V_index = None
        self.t_index = None

        self.order = []
        if X:
            self.X_index = len(self.order)
            self.order.append("X")
        if P:
            self.P_index = len(self.order)
            self.order.append("P")
        if S:
            self.S_index = len(self.order)
            self.order.append("S")
        if V:
            self.V_index = len(self.order)
            self.order.append("V")
        if t:
            self.t_index = len(self.order)
            self.order.append("t")

    def get_index_for(self, N):
        if N in self.order:
            return self.order.index(N)
        else:
            return None


class SolverLBFGSParams:
    """
    Parâmetros especificamente relacionados à aplicação do solver L-BFGS
    do_pre_optimization = irá passar lbfgs antes de adam
    do_post_optimization = irá passar lbfgs após adam
    """

    def __init__(self, do_pre_optimization=True, do_post_optimization=False, LR=None):
        self.do_pre_optimization = do_pre_optimization
        self.do_post_optimization = do_post_optimization
        self.LR = LR


class SolverParams:
    def __init__(
        self,
        name=None,
        num_domain=None,
        num_boundary=None,
        num_init=None,
        num_test=None,
        adam_epochs=None,
        adam_display_every=None,
        sgd_epochs=None,
        adam_lr=None,
        l_bfgs: SolverLBFGSParams = SolverLBFGSParams(),
        layer_size=None,
        activation=None,
        initializer=None,
        loss_weights=None,
        input_non_dim_scaler: NonDimScaler = None,
        output_non_dim_scaler: NonDimScaler = None,
        mini_batch=None,
        hyperfolder=None,
        isplot=False,
        is_save_model=False,
        outputSimulationType: SystemSimulationType = SystemSimulationType(),
        inputSimulationType: SystemSimulationType = SystemSimulationType(),
        loss_version=1,
        custom_loss_version={},
        train_distribution="Hammersley",
        save_caller: PINNSaveCaller = None,
        train_input_range=None,
    ):
        self.name = name if name else None
        """
        A name that explains, defines or identify this solver_params objects
        """
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_init = num_init
        self.num_test = num_test
        self.adam_epochs = adam_epochs
        self.adam_display_every = adam_display_every
        self.sgd_epochs = sgd_epochs
        self.adam_lr = adam_lr
        self.l_bfgs = (
            l_bfgs
            if l_bfgs
            else SolverLBFGSParams(pre_optimization=False, post_optimization=False)
        )
        self.layer_size = layer_size
        self.activation = activation
        self.initializer = initializer
        self.loss_weights = loss_weights
        self.input_non_dim_scaler = input_non_dim_scaler
        self.output_non_dim_scaler = output_non_dim_scaler
        self.mini_batch = mini_batch
        # Hyperfolder é a pasta padrão
        # onde salvar os resultados daquele trambei
        self.hyperfolder = hyperfolder

        self.isplot = isplot
        "Se vai plotar usando o arg isplot em run_reactor (dde.save)"

        self.is_save_model = is_save_model
        "Se vai salver o modelo usando dde.save"

        self.outputSimulationType = outputSimulationType
        "Variáveis de saída (XPSV)"

        self.inputSimulationType = inputSimulationType
        "Variáveis de entrada (XPSV)"
        self.loss_version = loss_version
        """Versão do solver.
        """

        self.custom_loss_version = custom_loss_version
        """
        Deve ser um dictionary, como:
            {
                X: 1, # versão 1
                V: 3 #versão 3
            }
        """

        self.train_distribution = train_distribution
        self.save_caller = save_caller
        self.train_input_range = train_input_range

    def get_loss_version_for_type(self, type):
        """
        type é "X", "P", etc
        """

        loss_version = self.loss_version
        custom_loss_version = self.custom_loss_version.get(type, None)
        if custom_loss_version:
            loss_version = custom_loss_version
        return loss_version

    def to_dict(self) -> str:
        return {
            "num_init": self.num_init,
            "num_domain": self.num_domain,
            "num_boundary": self.num_boundary,
            "adam_epochs": self.adam_epochs,
            "adam_lr": self.adam_lr,
            "sgd-epochs": self.sgd_epochs,
            "l_bfgs": {
                "post": self.l_bfgs.do_post_optimization,
                "pre": self.l_bfgs.do_pre_optimization,
            },
            "layer_size": np.array(self.layer_size).tolist(),
            "activation": self.activation,
            "initializer": self.initializer,
            "loss_version": self.loss_version,
            "mini_batch": self.mini_batch,
            "nondim_scaler_input": self.input_non_dim_scaler.toDict(),
            "nondim_scaler_output": self.output_non_dim_scaler.toDict(),
            "train_input_range": np.array(self.train_input_range).tolist(),
            "train_distribution": self.train_distribution,
            "loss_weights": np.array(self.loss_weights).tolist(),
        }


# python -m domain.params.solver_params
if __name__ == "__main__":
    """
    Testes
    """

    normal = SystemSimulationType()
    assert normal.X is True, "as expected from default"
    assert normal.P is True, "as expected from default"
    assert normal.S is True, "as expected from default"
    assert normal.V is True, "as expected from default"
    assert normal.X_index == 0, "as expected from default"
    assert normal.P_index == 1, "as expected from default"
    assert normal.S_index == 2, "as expected from default"
    assert normal.V_index == 3, "as expected from default"
    assert normal.order[0] == "X"
    assert normal.order[1] == "P"
    assert normal.order[2] == "S"
    assert normal.order[3] == "V"
    assert len(normal.order) == 4

    xv = SystemSimulationType(["X", "V"])
    assert xv.X is True
    assert xv.P is False
    assert xv.S is False
    assert xv.V is True
    assert xv.X_index == 0
    assert xv.P_index is None
    assert xv.S_index is None
    assert xv.V_index == 1
    assert xv.order[0] == "X"
    assert xv.order[1] == "V"
    assert len(xv.order) == 2

    # Sistema com tempo
    xv = SystemSimulationType(["t", "V"])
    assert xv.t is True
    assert xv.X is False
    assert xv.P is False
    assert xv.S is False
    assert xv.V is True
    assert xv.V_index == 0
    assert xv.t_index == 1, "t_index must always be the last"
    assert xv.X_index is None
    assert xv.P_index is None
    assert xv.S_index is None
    assert xv.order[0] == "V"
    assert xv.order[1] == "t"
    assert len(xv.order) == 2

    # Testing custom loss for each item
    sp = SolverParams(loss_version=2, custom_loss_version={"X": 5})
    assert sp.get_loss_version_for_type("V") == 2
    assert sp.get_loss_version_for_type("X") == 5

    print("success")
