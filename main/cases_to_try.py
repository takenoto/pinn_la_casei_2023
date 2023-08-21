from domain.optimization.non_dim_scaler import NonDimScaler

# Opções de adimensionalização disponíveis:
# Key é o identificador
# abrv é a abreviação
NondimAvailableOptions = {
    "None": {"abrv": "N"},
    "Linear": {
        "abrv": "L",
        "to": NonDimScaler.toNondimLinearScaler,
        "from": NonDimScaler.fromNondimLinearScaler,
    },
    "Desvio": {
        "abrv": "D",
        "to": NonDimScaler.toNondimDesvio,
        "from": NonDimScaler.fromNondimDesvio,
    },
}


def change_layer_fix_neurons_number(eq_params, process_params):
    # --------- LOSS FUNCTION ------------
    # 1 é a loss tradicional, #2 retorna o própria X/P/S/V caso seja menor que zero.
    # 3 é a que faz dMols/dt e não dConc/dt, pro volume ir multiplicando... E também
    # faz com que EFETIVAMENTE X NaN seja bloqueado
    # A loss v4 é a que faz com que EFETIVAMENTE retorne o próprio valor
    # da coisa (XPSV) se for < 0 ou maior que o limite (Xm, Pm, So. Volume fica solto.)
    # E a loss v4 também é absoluta
    loss_version = 4

    # ---------------- NN ------------------
    func = "tanh"  #'tanh' #'swish'
    mini_batch = [None, 100]  # 100 #None  # 50 #200
    initializer = "Glorot normal"  #'Glorot normal' #'Glorot normal' #'Orthogonal'
    # GLOROT UNIFORM # Era Glorot Normal nos testes sem swish
    LR = 1e-3  # 1e-3
    lbfgs_post = 0  # 1
    ADAM_EPOCHS = 1  # 15000  # 45000
    SGD_EPOCHS = None  # 1000
    dictionary = {}
    neurons = [60, 30]
    layers = [4, 3]  # [4,3,2]

    # Se irá aplicar a estratégia de adimensionalização padrão
    NondimSelectedOptions = [
        NondimAvailableOptions["None"],
        # NondimAvailableOptions["Linear"],
        # NondimAvailableOptions["Desvio"],
    ]
    IS_LOSS_WEIGHT = False

    cols = len(layers * len(NondimSelectedOptions))
    rows = len(neurons)
    # Me parece que quando usa 2 var de entrada precisa de um NUM_DOMAIN
    # e teste bem maior pra ficar razoável

    NUM_DOMAIN = [300]
    NUM_TEST = [300]
    NUM_INIT = [80]
    NUM_BOUNDARY = 0

    # Anota aqui as variáveis que vão ser suportadas nessa simulação
    # supported_variables = ['X', 'P', 'S', 'V']
    # Por padrão, t de entrada e XPSV de saída:
    output_variables = ["X", "P", "S", "V"]
    input_variables = ["t"]
    # Alternativamente, PSV de saída e tV de entrada
    # output_variables = ["X", "P", "S"]
    # input_variables = ["t", "V"]

    # Globais
    input_str = "in_"
    for i in input_variables:
        input_str += i

    # Específicos
    for n_domain in NUM_DOMAIN:
        for n_test in NUM_TEST:
            for n_init in NUM_INIT:
                for NL in neurons:
                    for HL in layers:
                        for nd in NondimSelectedOptions:
                            for mb in mini_batch:
                                # Montando o nome:
                                minibatch_str = f"mb{mb}" if mb is not None else "mb-"
                                nondim_str = f'ND{ nd["abrv"] }'
                                key = (
                                    # Primeiro o "core"
                                    f"{NL}x{HL} {input_str} {func}"
                                    #Depois coisas l relacionadas ao treino
                                    + f" l{loss_version}"
                                    + f" {nondim_str} {minibatch_str}"
                                    + f" nd{n_domain} nt{n_test} ni{n_init}"
                                )

                                # Executando ações:
                                dictionary[key] = {
                                    "layer_size": [len(input_variables)]
                                    + [NL] * HL
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
                                        t=process_params.t_final,
                                        toNondim=nd["to"],
                                        fromNondim=nd["from"],
                                    )
                                    if nd["abrv"] != "N"
                                    else NonDimScaler()
                                )

                                if IS_LOSS_WEIGHT:
                                    dictionary[key]["w_X"] = 1 / 10  # 100 # 1 / 3
                                    dictionary[key]["w_P"] = 1  # 1000 #1 / 100
                                    dictionary[key]["w_S"] = 1 / 100  # 1/10 #1 / 1000
                                    dictionary[key]["w_V"] = 1

                                dictionary[key]["activation"] = func
                                if mini_batch:
                                    dictionary[key]["mini_batch"] = mb
                                dictionary[key]["num_domain"] = n_domain
                                dictionary[key]["num_test"] = n_test
                                dictionary[key]["num_init"] = n_init
                                dictionary[key]["num_bound"] = NUM_BOUNDARY
                                dictionary[key]["lbfgs_pre"] = 0
                                dictionary[key]["lbfgs_post"] = lbfgs_post
                                dictionary[key]["LR"] = LR
                                dictionary[key][
                                    "hyperfolder"
                                ] = f"{input_str} {nondim_str} 2023_08_21"
                                dictionary[key]["isplot"] = False
                                dictionary[key]["initializer"] = initializer
                                dictionary[key]["output_variables"] = output_variables
                                dictionary[key]["input_variables"] = input_variables
                                dictionary[key]["loss_version"] = loss_version
                                dictionary[key]["custom_loss_version"] = {
                                    # 'X':3,
                                    # 'V':3,
                                }

    return (dictionary, cols, rows)
