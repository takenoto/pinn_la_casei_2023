# python -m domain.params.solver_params

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
    def __init__(self, supported_variables=['X', 'P', 'S', 'V']):
        self.t = 't' in supported_variables
        self.X = 'X' in supported_variables
        self.P = 'P' in supported_variables
        self.S = 'S' in supported_variables
        self.V = 'V' in supported_variables

        # Só pra ficar mais fácil de digitar abaixo
        t= self.t
        X = self.X
        P = self.P
        S = self.S
        V = self.V

        # Estabelece o index de cada uma, sempre seguindo um padrão,
        # que independe de "supported_variables"
        # self.t_index = 0 if t else None
        # self.X_index = None if not X else (self.t_index+1 if t else 0)
        # self.P_index = None if not P else (self.X_index+1 if X else self.t_index+1 if t else 0) #1 if P and X else 1 if P else None
        # self.S_index = None if not S else (self.P_index+1 if P else self.X_index+1 if X else self.t_index+1 if t else 0) #2 if P and S else None # o S não pode ser calculado sem o P
        # self.V_index = None if not V else (self.S_index+1 if S else self.P_index+1 if P else self.X_index+1 if X else self.t_index+1 if t else 0)#3 if X and P and S else 2 if P else 1 if X else 0 if V else None;
        self.X_index = None
        self.P_index = None
        self.S_index = None
        self.V_index = None
        self.t_index = None

        self.order = []
        if(X):
            self.X_index = len(self.order)
            self.order.append('X')
        if(P):
            self.P_index = len(self.order)
            self.order.append('P')
        if(S):
            self.S_index = len(self.order)
            self.order.append('S')
        if(V):
            self.V_index = len(self.order)
            self.order.append('V')
        if(t):
            self.t_index = len(self.order)
            self.order.append('t')



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
        num_init=None,
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
        hyperfolder=None,
        isplot=False,
        outputSimulationType:SystemSimulationType=SystemSimulationType(),
        inputSimulationType:SystemSimulationType=SystemSimulationType()
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
        self.isplot=isplot
        "Se vai plotar usando o arg isplot em run_reactor (dde.save)"
        self.outputSimulationType = outputSimulationType
        "Variáveis de saída (XPSV)"
        self.inputSimulationType = inputSimulationType
        "Variáveis de entrada (XPSV)"


# python -m domain.params.solver_params
if __name__ == "__main__":
    """
    Testes
    """
    
    normal = SystemSimulationType()
    assert normal.X == True, "as expected from default"
    assert normal.P == True, "as expected from default"
    assert normal.S == True, "as expected from default"
    assert normal.V == True, "as expected from default"
    assert normal.X_index == 0, "as expected from default"
    assert normal.P_index == 1, "as expected from default"
    assert normal.S_index == 2, "as expected from default"
    assert normal.V_index == 3, "as expected from default"
    assert normal.order[0] == 'X'
    assert normal.order[1] == 'P'
    assert normal.order[2] == 'S'
    assert normal.order[3] == 'V'
    assert len(normal.order) == 4

    xv = SystemSimulationType(['X', 'V'])
    assert xv.X == True
    assert xv.P == False
    assert xv.S == False
    assert xv.V == True
    assert xv.X_index == 0
    assert xv.P_index == None
    assert xv.S_index == None
    assert xv.V_index == 1
    assert xv.order[0] == 'X'
    assert xv.order[1] == 'V'
    assert len(xv.order) == 2

    # Sistema com tempo
    xv = SystemSimulationType(['t', 'V'])
    assert xv.t == True
    assert xv.X == False
    assert xv.P == False
    assert xv.S == False
    assert xv.V == True
    assert xv.V_index == 0
    assert xv.t_index == 1, "t_index must always be the last"
    assert xv.X_index == None
    assert xv.P_index == None
    assert xv.S_index == None
    assert xv.order[0] == 'V'
    assert xv.order[1] == 't'
    assert len(xv.order) == 2

    print('success')
