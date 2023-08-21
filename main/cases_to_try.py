from domain.optimization.non_dim_scaler import NonDimScaler


def change_layer_fix_neurons_number(eq_params, process_params):
    # --------- LOSS FUNCTION ------------
    # 1 é a loss tradicional, #2 retorna o própria X/P/S/V caso seja menor que zero.
    # 3 é a que faz dMols/dt e não dConc/dt, pro volume ir multiplicando... E também
    # faz com que EFETIVAMENTE X NaN seja bloqueado
    # A loss v4 é a que faz com que EFETIVAMENTE retorne o próprio valor 
    # da coisa (XPSV) se for < 0 ou maior que o limite (Xm, Pm, So. Volume fica solto.)
    loss_version = 4
    # O objetivo da v2 é desincentivar valores negativos.

    # ---------------- NN ------------------
    func = "tanh"  #'tanh' #'swish'
    mini_batch = 100  # 100 #None  # 50 #200
    initializer = "Glorot normal"  #'Glorot normal' #'Glorot normal' #'Orthogonal'
    # GLOROT UNIFORM # Era Glorot Normal nos testes sem swish
    LR = 1e-4  # 1e-3
    lbfgs_post = 1
    ADAM_EPOCHS = 18000  # 45000
    SGD_EPOCHS = None  # 1000
    dictionary = {}
    neurons = [60, 30]
    layers = [4, 3]  # [4,3,2]
    cols = len(layers)
    rows = len(neurons)

    # Se irá aplicar a estratégia de adimensionalização padrão
    IS_NONDIM = True
    IS_LOSS_WEIGHT = False

    # Me parece que quando usa 2 var de entrada precisa de um NUM_DOMAIN
    # e teste bem maior pra ficar razoável
    NUM_DOMAIN = 500  # 300
    NUM_TEST = 500  # 300
    NUM_INIT = 80
    NUM_BOUNDARY = 0

    # Anota aqui as variáveis que vão ser suportadas nessa simulação
    # supported_variables = ['X', 'P', 'S', 'V']
    # Por padrão, t de entrada e XPSV de saída:
    output_variables = ["X", "P", "S", "V"]
    input_variables = ["t"]
    # Alternativamente, PSV de saída e tV de entrada
    # output_variables = ["X", "P", "S"]
    # input_variables = ["t", "V"]

    for n in neurons:
        for l in layers:
            key = f"{n}x{l} {func}"
            dictionary[key] = {
                "layer_size": [len(input_variables)]
                + [n] * l
                + [len(output_variables)],
                "adam_epochs": ADAM_EPOCHS,
                "sgd_epochs": SGD_EPOCHS,
            }
            dictionary[key]["scaler"] = (
                NonDimScaler(
                    X=eq_params.Xm,
                    P=eq_params.Pm,
                    S=eq_params.So,
                    V=process_params.max_reactor_volume,
                    t=1,
                    toNondim=NonDimScaler.toNondimDesvio,
                    fromNondim=NonDimScaler.fromNondimDesvio,
                )
                if IS_NONDIM
                else NonDimScaler()
            )

            if IS_LOSS_WEIGHT:
                dictionary[key]["w_X"] = 1 / 10  # 100 # 1 / 3
                dictionary[key]["w_P"] = 1  # 1000 #1 / 100
                dictionary[key]["w_S"] = 1 / 100  # 1/10 #1 / 1000
                dictionary[key]["w_V"] = 1

    for key in dictionary:
        dictionary[key]["activation"] = func
        if mini_batch:
            dictionary[key]["mini_batch"] = mini_batch
        dictionary[key]["num_domain"] = NUM_DOMAIN
        dictionary[key]["num_test"] = NUM_TEST
        dictionary[key]["num_init"] = NUM_INIT
        dictionary[key]["num_bound"] = NUM_BOUNDARY
        dictionary[key]["lbfgs_pre"] = 0
        dictionary[key]["lbfgs_post"] = lbfgs_post
        dictionary[key]["LR"] = LR
        dictionary[key][
            "hyperfolder"
        ] = f'batch t {"ND" if IS_NONDIM else ""} 2023_08_20'
        dictionary[key]["isplot"] = True
        dictionary[key]["initializer"] = initializer
        dictionary[key]["output_variables"] = output_variables
        dictionary[key]["input_variables"] = input_variables
        dictionary[key]["loss_version"] = loss_version

        dictionary

    return (dictionary, cols, rows)
