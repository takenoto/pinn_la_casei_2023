import math
from multiprocessing import process
from domain.optimization.non_dim_scaler import NonDimScaler

# Opções de adimensionalização disponíveis:
# Key é o identificador
# abrv é a abreviação
NondimAvailableOptions = {
    "None": {"abrv": "None"},
    "Linear": {
        "abrv": "Lin",
        "to": NonDimScaler.toNondimLinearScaler,
        "from": NonDimScaler.fromNondimLinearScaler,
    },
    "Desvio": {
        "abrv": "Desv",
        "to": NonDimScaler.toNondimDesvio,
        "from": NonDimScaler.fromNondimDesvio,
    },
}

LRs_dict = {
    f"{num}e-{exp}": num * (10**-exp) for num in range(1, 10) for exp in range(1, 10)
}

ADAM_EPOCHS_dict = {
    f"{num}{'k' if mult==1000 else ''}": num * mult
    for num in range(1, 101)
    for mult in [1, 1000]
}


def get_train_input_range_dict(
    process_params, percent_min_range=[0, 50], percent_max_range=[100]
):
    """
    percent_max_range is the maximum % that will be created to be used.
    If None, no percent max dictionary entries will be created.

    How are the keys written:
    Percent: {min_perc}-{max_perc}pa" - era パー mas deu erro ao salvar

    input2Multiplier
    Is the multiplier of the #2 part of the input. Can be X, P, S or V

    """

    # Full time discretization
    dict = {
        "full": [
            [0, process_params.t_final],
            # [0, input2Multiplier]
        ]
    }

    # Percentual discretization from 0 to "i"
    for max_perc in percent_max_range:
        for min_perc in percent_min_range:
            dict[f"{min_perc}-{max_perc}pa"] = [
                [
                    process_params.t_final * min_perc / 100,
                    process_params.t_final * max_perc / 100,
                ],
            ]

    return dict


def change_layer_fix_neurons_number(eq_params, process_params, hyperfolder=None):
    dictionary = {}
    # --------- LOSS FUNCTION -----------
    loss_version = 5  # 6 5 4 3 2

    output_variables = ["X", "P", "S"]  # "V"
    input_variables = ["t"]

    # -------------------------------
    # DATA SAMPLING
    # -------------------------------
    # List (each_input)
    # each_input = List (input1, input2, input3)
    # input...N = List(min, max)
    train_input_range_dict = get_train_input_range_dict(
        process_params,
        percent_min_range=[
            0,
        ],  # [0,50] => iria fazer modelos iniciando em 0 e em 50
        percent_max_range=[10, 25, 50, 60, 90, 100, 200]
    )

    train_input_range_list = [
        "0-60pa",
        "full",
        # "0-50pa",
        # "30-90pa",
        # "0-200pa",
    ]

    NUM_DOMAIN = [32] #32  # [1600] [800] [400] [300] [80] [40] [20]
    NUM_TEST = [32]  # [1600] [800] [400] [300] [80] [40] [20]
    NUM_INIT = [16] #16 # [10] [20] [60] [80] 20 era o valor dos primeiros testes
    NUM_BOUNDARY = 0

    # -------------------------------
    # ---------------- NN ------------------
    func = "tanh"  #'tanh' 'swish' 'selu' 'relu'
    mini_batch = [None]  # [None] [20] [40] [80] [2]
    initializer = "Glorot uniform"  #'Glorot normal' #'Glorot uniform' #'Orthogonal'
    train_distribution_list = ["LHS"]  # "LHS" "Hammersley" "uniform"
    # GLOROT UNIFORM # Era Glorot Normal nos testes sem swish
    # LR_list = ["1e-2", "5e-3", "1e-3", "5e-4", "5e-5", "1e-5", "1e-6"]
    # LR_list = ["5e-2", "1e-3", "3e-3", "5e-3", "8e-3", "1e-4", "3e-4", "5e-4", "1e-5"]
    LR_list = [
        # "1e-1",
        # "1e-2",
        "1e-3",
        "1e-4",
        "1e-5",
    ]  # ["8e-4", "5e-4", "1e-4", "7e-5"]  # ["1e-3", "1e-4", "1e-5"]

    lbfgs_pre = 0  # 0 1
    lbfgs_post = 1  # 0 1
    ADAM_EPOCHS_list = ["6k"]  # ["25k"]  # ["100", "1k", "10k"]
    SGD_EPOCHS = 0  # 1000
    neurons = [32, 16]  # [16]  # [16, 32, 60]  # [16, 32, 60] [80, 120] [10]
    layers = [3, 2]  # [1, 2, 3, 4, 5]  # [2, 3, 4]  # [2, 3, 4, 5] [6, 7, 8]

    # Se irá aplicar a estratégia de adimensionalização padrão
    NondimSelectedOptions = [
        NondimAvailableOptions["None"],
        NondimAvailableOptions["Linear"],
        # NondimAvailableOptions["Desvio"],
    ]

    # -------------------------------
    # NONDIMENSIONALIZER
    # -------------------------------
    # Multiplica o scaler/adimensionalizador
    scaler_modifier_default = 1 / 10  # 1 #1/10
    # Aqui coloca as customizações
    scaler_modifiers = {
        "t": 1 / process_params.t_final,
        # Isso tira o t nondim porque tá desfazendo ele
        "X": scaler_modifier_default,
        "P": scaler_modifier_default,
        "S": scaler_modifier_default,
        "V": scaler_modifier_default,  # 1
    }

    # Loss Weight
    IS_LOSS_WEIGHT = False

    cols = len(neurons) * len(train_distribution_list) * len(NondimSelectedOptions)
    rows = (
        len(layers)
        * len(NUM_TEST)
        * len(NUM_INIT)
        * len(NUM_DOMAIN)
        * len(mini_batch)
        * len(LR_list)
    )

    # Globais
    input_str = "in_"
    for i in input_variables:
        input_str += i
    output_str = "out_"
    for o in output_variables:
        output_str += o

    # Específicos
    for train_input_range_key in train_input_range_list:
        for adam_str in ADAM_EPOCHS_list:
            for LR_str in LR_list:
                for train_distribution in train_distribution_list:
                    for n_domain in NUM_DOMAIN:
                        for n_test in NUM_TEST:
                            for n_init in NUM_INIT:
                                for NL in neurons:
                                    for HL in layers:
                                        for nd in NondimSelectedOptions:
                                            for mb in mini_batch:
                                                train_input_range = (
                                                    train_input_range_dict[
                                                        train_input_range_key
                                                    ]
                                                )

                                                ADAM_EPOCHS = ADAM_EPOCHS_dict[adam_str]
                                                LR = LRs_dict[LR_str]
                                                # Montando o nome:
                                                minibatch_str = (
                                                    f"mb{mb}"
                                                    if mb is not None
                                                    else "mb-"
                                                )
                                                nondim_str = f'ND{ nd["abrv"] }'
                                                key = (
                                                    # Primeiro o "core"
                                                    f"{NL}x{HL} {input_str} {output_str} {func}"
                                                    # Depois coisas l relacionadas ao treino
                                                    + f" l{loss_version}"
                                                    + f" LR-{LR_str}"
                                                    + f" {nondim_str} {minibatch_str}"
                                                    + f" nd{n_domain} nt{n_test} ni{n_init}"
                                                    + f" TD-{train_distribution}"
                                                    + f" {adam_str}ep"
                                                    + f" tr-{train_input_range_key}"
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
                                                        name=nd["abrv"],
                                                        X=eq_params.Xm
                                                        * scaler_modifiers["X"],
                                                        P=eq_params.Pm
                                                        * scaler_modifiers["P"],
                                                        S=eq_params.So
                                                        * scaler_modifiers["S"],
                                                        V=process_params.max_reactor_volume
                                                        * scaler_modifiers["V"],
                                                        t=process_params.t_final
                                                        * scaler_modifiers["t"],
                                                        toNondim=nd["to"],
                                                        fromNondim=nd["from"],
                                                    )
                                                    if nd["abrv"] != "None"
                                                    else NonDimScaler(name=nd["abrv"])
                                                )

                                                if IS_LOSS_WEIGHT:
                                                    dictionary[key][
                                                        "w_X"
                                                    ] = 10  # 100 # 1 / 3
                                                    dictionary[key][
                                                        "w_P"
                                                    ] = 1  # 1000 #1 / 100
                                                    dictionary[key][
                                                        "w_S"
                                                    ] = 1  # 1/10 #1 / 1000
                                                    dictionary[key]["w_V"] = 10

                                                dictionary[key]["activation"] = func
                                                if mini_batch:
                                                    dictionary[key]["mini_batch"] = mb
                                                dictionary[key]["num_domain"] = n_domain
                                                dictionary[key]["num_test"] = n_test
                                                dictionary[key]["num_init"] = n_init
                                                dictionary[key][
                                                    "num_bound"
                                                ] = NUM_BOUNDARY
                                                dictionary[key]["lbfgs_pre"] = lbfgs_pre
                                                dictionary[key][
                                                    "lbfgs_post"
                                                ] = lbfgs_post
                                                dictionary[key]["LR"] = LR
                                                dictionary[key]["hyperfolder"] = (
                                                    hyperfolder
                                                    if hyperfolder is not None
                                                    else f"{input_str} {nondim_str}"
                                                )
                                                dictionary[key]["isplot"] = False
                                                dictionary[key][
                                                    "initializer"
                                                ] = initializer
                                                dictionary[key][
                                                    "output_variables"
                                                ] = output_variables
                                                dictionary[key][
                                                    "input_variables"
                                                ] = input_variables
                                                dictionary[key][
                                                    "loss_version"
                                                ] = loss_version
                                                dictionary[key][
                                                    "custom_loss_version"
                                                ] = {
                                                    # 'X':3,
                                                    # 'V':3,
                                                }
                                                dictionary[key][
                                                    "train_distribution"
                                                ] = train_distribution
                                                dictionary[key][
                                                    "train_input_range"
                                                ] = train_input_range

    return (dictionary, cols, rows)
