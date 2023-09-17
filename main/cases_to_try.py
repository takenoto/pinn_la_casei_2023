from domain.optimization.non_dim_scaler import NonDimScaler
from domain.params import altiok_2006_params
from domain.params.process_params import ProcessParams


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
        percent_max_range=[10, 25, 50, 60, 90, 100, 200],
    )

    train_input_range_list = [
        "full",
        # "0-60pa",
        "0-90pa",
        # "0-200pa",
    ]

    N_POINTS = [
        # test points, domain points
        (32, 32),
        # (100, 100),
        # (70, 70),
        # (10, 10),
        # (300, 300),
    ]  # [1600] [800] [400] [300] [80] [40] [20]
    NUM_INIT = [16,
                # 80
                ]  # 16 # [10] [20] [60] [80] 20 era o valor dos primeiros testes
    NUM_BOUNDARY = 0

    # -------------------------------
    # ---------------- NN ------------------
    func = "tanh"  #'tanh' 'swish' 'selu' 'relu'
    mini_batch = [None]  # [None] [20] [40] [80] [2]
    initializer = "Glorot uniform"  #'Glorot normal' #'Glorot uniform' #'Orthogonal'
    train_distribution_list = ["uniform"]  # "LHS" "Hammersley" "uniform"
    # GLOROT UNIFORM # Era Glorot Normal nos testes sem swish
    # LR_list = ["1e-2", "5e-3", "1e-3", "5e-4", "5e-5", "1e-5", "1e-6"]
    # LR_list = ["5e-2", "1e-3", "3e-3", "5e-3", "8e-3", "1e-4", "3e-4", "5e-4", "1e-5"]
    LR_list = [
        # "E-2_1",
        # "E-3_5",  # = 5e-3
        "E-3_1",  # = 1e-3
        # "E-4_7",
        "E-4_5",
        "E-4_1",  # = 1e-4
        # "E-5_5", # = 5e-5
        # "E-5_1", # = 1e-5
    ]

    lbfgs_pre = 0  # 0 1
    lbfgs_post = 1  # 0 1
    ADAM_EPOCHS_list = [
        "100",
        "10k",
        # "10k",
        # "60k"
        # "35k",
        # "25k",
        # "1k",
    ]
    SGD_EPOCHS = 0  # 1000
    neurons = [
        # 20,
        30,
        # 45,
        60,
        # 80,
    ]  # , 16, 8]  # [16]  # [16, 32, 60]  # [16, 32, 60] [80, 120] [10]
    layers = [
        2,
        3,
        # 4,
        # 5
    ]  # [1, 2, 3, 4, 5]  # [2, 3, 4]  # [2, 3, 4, 5] [6, 7, 8]

    # -------------------------------
    # NONDIMENSIONALIZER
    # -------------------------------
    # Se irá aplicar a estratégia de adimensionalização padrão
    NDList = [
        # strategy, tscode, scalers_code
        # Esse é o mesmo que ser sem adimensionalização
        ("None", "t1", "1"),
        #
        # ----------
        # # Linear comum: t normal o resto nondim
        # ("Lin", "t1", "F1"),
        #
        # ----------
        # # Escalados por 10
        # ("Lin", "t1", "F1x10"),
        #
        # ----------
        # # Divididos por 10
        # ("Lin", "t1", "F1d10"),
        #
        # ---------------------------
        # Apenas o tempo nondim:
        ("Lin", "t2", "1"),
        # ("Lin", "t3", "1"),
        # ("Lin", "t4", "1"),
        # ("Lin", "t5", "1"),
        # ("Lin", "t6", "1"),
        # ("Lin", "t7", "1"),
        # ("Lin", "t8", "1"),
        #
        # ---------------------------
        # Tudo nondim (XPSV) incluindo o tempo:
        # ("Lin", "t2", "F1")
        # ("Lin", "t3", "F1")
        # ("Lin", "t4", "F1")
        # ("Lin", "t5", "F1")
        # ("Lin", "t6", "F1")
        # ("Lin", "t7", "F1")
        # ("Lin", "t8", "F1")
    ]

    # Loss Weight
    IS_LOSS_WEIGHT = False

    cols = len(neurons) * len(train_distribution_list) * len(NDList)
    rows = len(layers) * len(N_POINTS) * len(NUM_INIT) * len(mini_batch) * len(LR_list)

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
                    for n_points in N_POINTS:
                        for n_init in NUM_INIT:
                            for NL in neurons:
                                for HL in layers:
                                    for mb in mini_batch:
                                        for nd in NDList:
                                            n_test, n_domain = n_points
                                            train_input_range = train_input_range_dict[
                                                train_input_range_key
                                            ]

                                            ADAM_EPOCHS = ADAM_EPOCHS_dict[adam_str]
                                            LR = LRs_dict[LR_str]
                                            (
                                                nd_strategy,
                                                nd_tscode,
                                                nd_scalers_code,
                                            ) = nd
                                            nondim_scaler = get_nondim_scaler(
                                                process_params=process_params,
                                                eq_params=eq_params,
                                                strategy=nd_strategy,
                                                ts_code=nd_tscode,
                                                scalers_code=nd_scalers_code,
                                            )
                                            # Montando o nome:
                                            minibatch_str = (
                                                f"m{mb}" if mb is not None else "m-"
                                            )

                                            key = (
                                                # Primeiro o "core"
                                                f"ND-{nondim_scaler.name}"
                                                + f" {input_str} {output_str} {func}"
                                                + f" tr-{train_input_range_key}"
                                                + f" L{loss_version}"
                                                + f" LR-{LR_str}"
                                                + f" {NL}x{HL}"
                                                + f" p{n_init}-{n_domain}-{n_test}"
                                                + f" {adam_str}ep"
                                                + f" lbfgs-{lbfgs_post}"
                                                + f" TD-{train_distribution}"
                                                + f" {minibatch_str}"
                                            )

                                            # Executando ações:
                                            dictionary[key] = {
                                                "layer_size": [len(input_variables)]
                                                + [NL] * HL
                                                + [len(output_variables)],
                                                "adam_epochs": ADAM_EPOCHS,
                                                "sgd_epochs": SGD_EPOCHS,
                                            }
                                            dictionary[key]["scaler"] = nondim_scaler

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
                                            dictionary[key]["num_bound"] = NUM_BOUNDARY
                                            dictionary[key]["lbfgs_pre"] = lbfgs_pre
                                            dictionary[key]["lbfgs_post"] = lbfgs_post
                                            dictionary[key]["LR"] = LR
                                            dictionary[key]["hyperfolder"] = (
                                                hyperfolder
                                                if hyperfolder is not None
                                                else f"{input_str} {nondim_scaler.name}"
                                            )
                                            dictionary[key]["isplot"] = False
                                            dictionary[key]["initializer"] = initializer
                                            dictionary[key][
                                                "output_variables"
                                            ] = output_variables
                                            dictionary[key][
                                                "input_variables"
                                            ] = input_variables
                                            dictionary[key][
                                                "loss_version"
                                            ] = loss_version
                                            dictionary[key]["custom_loss_version"] = {
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


def get_nondim_scaler(
    process_params: ProcessParams,
    eq_params: altiok_2006_params.Altiok2006Params,
    strategy,
    ts_code,
    scalers_code,
):
    """Returns the values of nondim scaler

    Args:
        process_params (ProcessParams): _description_

        strategy (str): Strategy of Nondimensionalization. Can be "None", "Lin" (linear)
        and "Desv" (Desvio)

        ts_code (str): Scalers string. For time only.

        scalers_code (str): Scalers string. Defines the scalers parameters.

        ts_code = "t1" => No time nondimensionalizations. The others are defined in the document.
    """
    t_S, X_S, P_S, S_S, V_S = get_nondim_scaler_values(
        process_params=process_params,
        eq_params=eq_params,
        ts_code=ts_code,
        scalers_code=scalers_code,
    )

    if strategy == "None":
        return NonDimScaler(
            name=f"N-{ts_code}-{scalers_code}", t=t_S, X=X_S, P=P_S, S=S_S, V=V_S
        )
    elif strategy == "Lin":
        return NonDimScaler(
            name=f"Lin-{ts_code}-{scalers_code}",
            t=t_S,
            X=X_S,
            P=P_S,
            S=S_S,
            V=V_S,
            toNondim=NonDimScaler.toNondimLinearScaler,
            fromNondim=NonDimScaler.fromNondimLinearScaler,
        )
    elif strategy == "Desv":
        return NonDimScaler(
            name=f"Desv-{ts_code}-{scalers_code}",
            t=t_S,
            X=X_S,
            P=P_S,
            S=S_S,
            V=V_S,
            toNondim=NonDimScaler.toNondimDesvio,
            fromNondim=NonDimScaler.fromNondimDesvio,
        )

    pass


def get_nondim_scaler_values(
    process_params: ProcessParams,
    eq_params: altiok_2006_params.Altiok2006Params,
    ts_code="t1",
    scalers_code="1",
):
    """Returns the values of nondim scaler. Always in the order: t, X, P, S, V

    Args:
        process_params (ProcessParams): Used for creating some of the scalers

    F = full, significa que todos os demais (menos tempo) foram adimensionalizados
    """
    ts, Xs, Ps, Ss, Vs = 1, 1, 1, 1, 1
    match ts_code:
        # Evaluate only the time nondim
        case "t1":
            ts = 1
        case "t2":
            ts = process_params.t_final
        case "t2x10":
            ts = process_params.t_final * 10
        case "t2d10":
            ts = process_params.t_final / 10
        case "t3":
            ts = 10 * process_params.t_final
        case "t4":
            ts = 0.1 * process_params.t_final
        case "t5":
            ts = (eq_params.K_S + process_params.So) / (
                eq_params.mu_max * process_params.So
            )
        case "t6":
            ts = 1 / eq_params.mu_max
        case "t7":
            ts = (
                eq_params.alpha
                * process_params.So
                / ((eq_params.K_S + process_params.So) * eq_params.mu_max)
            )
        case "t8":
            ts = (
                (1 / eq_params.Y_PS)
                * eq_params.alpha
                * (eq_params.K_S + process_params.So)
                / ((process_params.So) * eq_params.mu_max)
            )
        case _:
            assert False, "case must exist"

    match scalers_code:
        case "1":
            Xs = 1
            Ps = 1
            Ss = 1
            Vs = 1
        # É F1 porque escala entre os valores máximo e 0
        # então fica tudo entre 0 e 1 teoricamente
        case "F1":
            Xs = eq_params.Xm
            Ps = eq_params.Pm
            Ss = eq_params.So
            Vs = process_params.max_reactor_volume
        # "x" representa multiplicação
        case "F1x10":
            Xs = eq_params.Xm * 10
            Ps = eq_params.Pm * 10
            Ss = eq_params.So * 10
            Vs = process_params.max_reactor_volume * 10
        case "F1x100":
            Xs = eq_params.Xm * 100
            Ps = eq_params.Pm * 100
            Ss = eq_params.So * 100
            Vs = process_params.max_reactor_volume * 100
        # O "d" representa uma divisão
        case "F1d10":
            Xs = eq_params.Xm / 10
            Ps = eq_params.Pm / 10
            Ss = eq_params.So / 10
            Vs = process_params.max_reactor_volume / 10
        case "F1d100":
            Xs = eq_params.Xm / 100
            Ps = eq_params.Pm / 100
            Ss = eq_params.So / 100
            Vs = process_params.max_reactor_volume / 100
        case _:
            print(scalers_code)
            assert False, "case must exist"

    return (ts, Xs, Ps, Ss, Vs)


LRs_dict = {
    f"E-{exp}_{num}": num * (10**-exp) for num in range(1, 10) for exp in range(1, 10)
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
