import os
import numpy as np

from domain.optimization.non_dim_scaler import NonDimScaler
from domain.params import altiok_2006_params
from domain.params.process_params import ProcessParams

# ----------------------------
# CAUTION
# OS CASES T3 E T4 SÃO T2D10 E T2X10 JÁ!!!!!!!!!
# ----------------------------


def change_layer_fix_neurons_number(eq_params, process_params):
    dictionary = {}
    #
    # -------------------------------
    # ---------------- NN -----------
    # -------------------------------
    #
    #'tanh' 'swish' 'selu' 'relu'
    activation_functions = [
        "tanh",
        "swish",
        # "selu",
        # RELU IS NOT SECOND ORDER DIFFERENTIABLE
        # AND SHOULD NOT BE USED FOR THIS PROJECT
        # REF: https://github.com/lululxvi/deepxde/issues/80
    ]
    neurons = [
        # 2,
        # 4,
        # 6,
        # 8,
        10,
        # 12,
        # 16,
        # 20,
        30,
        # 32,
        # 45,
        # 60,
        # 80,
        # 100,
        # 160
    ]
    layers = [
        1,
        # 2,
        3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8
    ]

    # -------------------------------
    # NONDIMENSIONALIZER
    # -------------------------------
    # Se irá aplicar a estratégia de adimensionalização padrão
    # Testes com blocos facilitam a vida porque basta dizer quais terão
    # Blocos custom:
    NDList = [
        # Order:
        # (tscode, scalers_code, input strategy, output strategy)
        # ex: ("t2", "F1", "Lin", "Upx1"),
        # Bota a padrão quase sempre, que é pra ter uma base de comparação
        # ---------------------------------
        #
        # CURRENT ITERATION
        #
        # Add here the ones to try
        # -----------------
        # # NO NONDIM:
        ("t1", "1", "Lin", "Lin"),
        #
        # -----------------------------------
        # ---------------- t2 e F1, d10 e x10
        # # LIN-LIN
        # ("t2", "F1", "Lin", "Lin"),
        # ("t2", "F1d10", "Lin", "Lin"),
        # ("t2", "F1x10", "Lin", "Lin"),
        # ("t2x10", "F1", "Lin", "Lin"),
        # ("t2d10", "F1", "Lin", "Lin"),
        # # LIN-UPX1
        # ("t2", "F1", "Lin", "Upx1"),
        # ("t2", "F1d10", "Lin", "Upx1"),
        # ("t2", "F1x10", "Lin", "Upx1"),
        # ("t2x10", "F1", "Lin", "Upx1"),
        # ("t2d10", "F1", "Lin", "Upx1"),
        #
        # -----------------------------------
        # ---------------- var times lin-lin
        # F1
        # ("t5", "F1", "Lin", "Lin"),
        # ("t6", "F1", "Lin", "Lin"),
        # ("t7", "F1", "Lin", "Lin"),
        # ("t8", "F1", "Lin", "Lin"),
        # ("t9", "F1", "Lin", "Lin"),
        #
        # F1d10
        # ("t5", "F1d10", "Lin", "Lin"),
        # ("t6", "F1d10", "Lin", "Lin"),
        # ("t7", "F1d10", "Lin", "Lin"),
        # ("t8", "F1d10", "Lin", "Lin"),
        # ("t9", "F1d10", "Lin", "Lin"),
        #
        # F1x10 - por padrão nem faz porque já sei que fica ruim
        # ("t5", "F1x10", "Lin", "Lin"),
        # ("t6", "F1x10", "Lin", "Lin"),
        # ("t7", "F1x10", "Lin", "Lin"),
        # ("t8", "F1x10", "Lin", "Lin"),
        # ("t9", "F1x10", "Lin", "Lin"),
        #
        # -----------------------------------
        # ---------------- var times lin-upx1
        #
        # F1d10
        # ("t5", "F1d10", "Lin", "Upx1"),
        # ("t6", "F1d10", "Lin", "Upx1"),
        # ("t7", "F1d10", "Lin", "Upx1"),
        # ("t8", "F1d10", "Lin", "Upx1"),
        # ("t9", "F1d10", "Lin", "Upx1"),
    ]
    ndlist_additional_blocks = [
        # Using now:
        # faz um a um pq aí pode parar e continuar depois
        # "default", # só ("t1", "1", "Lin", "Lin"),
        # ----------------------
        # PARTE 1
        # "1",
        # ----------------------
        # PARTE 2
        # "2.1-LinUpx1",
        # "2.2-t2F1-Lin<>Upx1",
        # ----------------------
        # ----------------------
        # PARTE 3
        # "3.1LinLinF1",
        # "3.1LinLinF1d10",
        # "3.2LinUpx1F1",
        # "3.2LinUpx1F1d10",
        # ----------------------
        #
        # -- DOCs:
        # -----------------
        # "1" => Itera rapidamente por lin escalonando t e "N"s individualmente
        #
        # -----------------
        # "2.1-LinUpx1" => testa combinação lin => Upx com t2F1
        # "2.2-t2F1-Lin<>Upx1" => Fixa Lin=>Upx1 e muda escalonamento (x e d)
        # Guia sobre usar UPx na entrada, saída, ou ambos
        #
        # -----------------
        # "3.1LinLinF1" => itera todos os tempos por LINLIN usando F1 e cada tempo
        # "3.1LinLinF1d10" => memsoque 3.1F1 mas com F1d10
        # "3.2LinUpx1F1" => Mesma coisa que 3.1F1 mas usa Lin-Upx1
        # "3.2LinUpx1F1d10" => Mesma coisa que 3.2Upx1F1 mas usa Lin-Upx1
    ]
    nondimlist_add_from_blocks(NDList, ndlist_additional_blocks)

    #
    # --------------------------------
    # -------- TRAINING
    # --------------------------------
    #

    # --------- LOSS FUNCTION -----------
    # "7A-I" e 6 5 4 3 2
    # loss_version_list = ["7A", "7B", "7C", "7D", "7E", "7F", "7G", "7H", "7I"]
    loss_version_list = ["7B", "7D", "7G"]

    input_output_variables_list = [
        # ----------------------
        # Input, output
        # ----------------------
        #
        # t => XPSV
        # (["t"], ["X", "P", "S", "V"]),
        # t => V
        # (["t"], ["V"]),
        # # t => XPS
        (["t"], ["X", "P", "S"]),
        # # t, V => XPS
        # (["t", "V"], ["X", "P", "S"]),
    ]
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
        percent_max_range=[10, 15, 25, 35, 50, 60, 90, 100, 200],
    )

    train_input_range_list = [
        # "0-10pa",
        # "0-15pa",
        # "0-25pa",
        # "0-35pa",
        # "0-60pa",
        # "0-90pa",
        "0-100pa",
        # "0-200pa",
    ]

    N_POINTS = [
        # ORDER:
        ## initial points, domain points, test points
        #
        # --------------------
        # USING:
        (10, 20, 20),
        # (4, 20, 20),
        # (4, 20, 800),
        # (4, 200, 20),
        #
        # PADRÃO:
        # (16, 32, 32),
        # (8, 24, 24),
        #
        # ---------------
        # Ppontos que avaliam se compensa ou não aumentar npoints
        # (20, 20, 20),
        # (4, 20, 20),
        # (4, 20, 800),
        # (4, 200, 20),
        # ou
        # (16, 32, 32),
        # (100, 32, 32),
        # (16, 300, 300),
        # (300, 300, 300),
        #
        # ---------------
        # (16, 100, 100),
        # (16, 70, 70),
        # (16, 10, 10),
        # (16, 300, 300),
    ]  # [1600] [800] [400] [300] [80] [40] [20]

    NUM_BOUNDARY = 0

    mini_batch = [None]  # [None] [20] [40] [80] [2]
    # O padrão era Glorot Uniform
    initializer = "Glorot uniform"  #'Glorot normal' #'Glorot uniform' #'Orthogonal'
    # Quando for fazer hypercube acho que posso boar distribution
    # começando com _ underline e usar isso pra checar
    train_distribution_list = ["Hammersley"]  # "LHS" "Hammersley" "uniform"
    # GLOROT UNIFORM # Era Glorot Normal nos testes sem swish
    LR_list = [
        # Default
        "E-3_4",
        "E-3_1",
        "E-4_5",
        # --------------------
        # FULL LIST:
        # --------------------
        # "E-2_1",
        # "E-3_9",  # = 9e-3
        # "E-3_8",  # = 8e-3
        # "E-3_7",  # = 7e-3
        # "E-3_6",  # = 6e-3
        # "E-3_5",  # = 5e-3
        # "E-3_4",  # = 4e-3
        # "E-3_3",  # = 3e-3
        # "E-3_2",  # = 2e-3
        # "E-3_1",  # = 1e-3
        # "E-4_9",  # = 9e-4
        # "E-4_8",  # = 8e-4
        # "E-4_7",  # = 7e-4
        # "E-4_6",  # = 6e-4
        # "E-4_5",  # = 5e-4
        # "E-4_4",  # = 4e-4
        # "E-4_3",  # = 3e-4
        # "E-4_2",  # = 2e-4
        # "E-4_1",  # = 1e-4
        # "E-5_9",  # = 9e-5
        # "E-5_8",  # = 8e-5
        # "E-5_7",  # = 7e-5
        # "E-5_6",  # = 6e-5
        # "E-5_5",  # = 5e-5
        # "E-5_4",  # = 4e-5
        # "E-5_3",  # = 3e-5
        # "E-5_2",  # = 2e-5
        # "E-5_1",  # = 1e-5
        # "E-6_5",  # = 5e-6
        # "E-6_1",  # = 1e-6
        # "E-7_1", # = 1e-7
    ]

    lbfgs_pre = 0  # 0 1
    lbfgs_post = 1  # 0 1
    ADAM_EPOCHS_list = [
        # "100",
        # "1k",
        # "10k",
        # "25k",
        # "30k",
        "35k",
        # "45k",
        # "60k",
        # "90k",
        # "120k",
        # "150k",
    ]
    # SGD_EPOCHS = 0  # 1000

    # Loss Weight
    loss_weights_list = ["A1"]  # All weights = 0

    # Específicos
    for input_output_variables in input_output_variables_list:
        for func in activation_functions:
            for train_input_range_key in train_input_range_list:
                for adam_str in ADAM_EPOCHS_list:
                    for train_distribution in train_distribution_list:
                        for n_points in N_POINTS:
                            for nd in NDList:
                                for loss_version in loss_version_list:
                                    for NL in neurons:
                                        for HL in layers:
                                            for mb in mini_batch:
                                                for LR_str in LR_list:
                                                    for (
                                                        loss_weight_str
                                                    ) in loss_weights_list:
                                                        _insert_into_list(
                                                            dictionary,
                                                            args=(
                                                                process_params,
                                                                eq_params,
                                                                input_output_variables,
                                                                loss_version,
                                                                lbfgs_pre,
                                                                lbfgs_post,
                                                                loss_version,
                                                                func,
                                                                initializer,
                                                                train_input_range_key,
                                                                adam_str,
                                                                train_distribution,
                                                                train_input_range_dict,
                                                                n_points,
                                                                NUM_BOUNDARY,
                                                                NL,
                                                                HL,
                                                                mb,
                                                                nd,
                                                                LR_str,
                                                                loss_weight_str,
                                                            ),
                                                        )

    return dictionary


def _insert_into_list(dictionary, args):
    (
        process_params,
        eq_params,
        input_output_variables,
        loss_version,
        lbfgs_pre,
        lbfgs_post,
        loss_version,
        func,
        initializer,
        train_input_range_key,
        adam_str,
        train_distribution,
        train_input_range_dict,
        n_points,
        NUM_BOUNDARY,
        NL,
        HL,
        mb,
        nd,
        LR_str,
        loss_weight_str,
    ) = args

    input_variables, output_variables = input_output_variables
    # Globais
    input_str = "in_"
    for i in input_variables:
        input_str += i
    output_str = "out_"
    for o in output_variables:
        output_str += o

    n_init, n_domain, n_test = n_points
    train_input_range = train_input_range_dict[train_input_range_key]
    ADAM_EPOCHS = ADAM_EPOCHS_dict[adam_str]
    LR = LRs_dict[LR_str]
    (
        nd_tscode,
        nd_scalers_code,
        nd_in_strategy,
        nd_out_strategy,
    ) = nd
    input_nondim_scaler, output_nondim_scaler = get_nondim_scaler(
        process_params=process_params,
        eq_params=eq_params,
        input_strategy=nd_in_strategy,
        output_strategy=nd_out_strategy,
        ts_code=nd_tscode,
        scalers_code=nd_scalers_code,
    )
    # Montando o nome:
    minibatch_str = f"m{mb}" if mb is not None else "m-"

    name = (
        f"{NL}x{HL}"
        + f" {func}"
        + f" L{loss_version}"
        + f" LR-{LR_str}"
        + f" w{loss_weight_str}"
        + f" {input_nondim_scaler.strategy_str}-{output_nondim_scaler.strategy_str}"
        + f" p{n_init}-{n_domain}-{n_test}"
        + f" {adam_str}ep"
        + f" lbfgs-{lbfgs_pre}-{lbfgs_post}"
        + f" {minibatch_str}"
    )

    key = (
        f"{NL}x{HL}"
        + f" {func}"
        + f" L{loss_version}"
        + f" LR-{LR_str}"
        + f" w{loss_weight_str}"
        + f" p{n_init}-{n_domain}-{n_test}"
        + f" {adam_str}ep"
        + f" lbfgs-{lbfgs_post}"
        + f" {minibatch_str}"
        + f" {input_str} {output_str}"
        + f" tr-{train_input_range_key}"
        + f"ND-{input_nondim_scaler.strategy_str}-{output_nondim_scaler.strategy_str}"
        + f"-{nd_tscode}-{nd_scalers_code}"
        + f" p{n_init}-{n_domain}-{n_test}"
        + f" TD-{train_distribution}"
    )

    # Executando ações:
    internal_dict = {
        "name": name,
        "layer_size": [len(input_variables)] + [NL] * HL + [len(output_variables)],
        "adam_epochs": ADAM_EPOCHS,
        "input_str": input_str,
        "output_str": output_str,
        "input_scaler": input_nondim_scaler,
        "output_scaler": output_nondim_scaler,
        "loss_weights": loss_weights(config=loss_weight_str),
        "activation": func,
        "mini_batch": mb,
        "num_domain": n_domain,
        "num_test": n_test,
        "num_init": n_init,
        "num_bound": NUM_BOUNDARY,
        "lbfgs_pre": lbfgs_pre,
        "lbfgs_post": lbfgs_post,
        "LR": LR,
        "isplot": False,
        "initializer": initializer,
        "output_variables": output_variables,
        "input_variables": input_variables,
        "loss_version": loss_version,
        "custom_loss_version": {},  # 'X':3, 'V':3,
        "train_distribution": train_distribution,
        "train_input_range": train_input_range,
    }

    internal_dict["hyperfolder"] = os.path.join(
        f"{input_str}-{output_str} tr- {train_input_range_key}"
        + f" {initializer}-{train_distribution}",
        func,
        f"ND-{input_nondim_scaler.strategy_str}-{output_nondim_scaler.strategy_str}-{nd_tscode}-{nd_scalers_code}",
    )

    dictionary[key] = internal_dict


def nondimlist_add_from_blocks(NDList, blocks_to_add):
    registered_blocs = {
        "no nondim": [("t1", "1", "Lin", "Lin")],
        # --------------------------------------
        # --------------------------------------
        # 1º Bloco: LINLIN Noção geral do impacto de t e vars crescendo e diminuindo
        #
        # F1 e t d10 e X10
        "1": [
            # ("t1", "F1", "Lin", "Lin"), # NOW IS THE "DEFAULT"
            ("t1", "F1d10", "Lin", "Lin"),
            ("t1", "F1x10", "Lin", "Lin"),
            #
            # 2º t nondim, x10 e d10 LIN LIN
            ("t2", "1", "Lin", "Lin"),
            ("t2", "F1d10", "Lin", "Lin"),
            ("t2x10", "1", "Lin", "Lin"),
            ("t2d10", "1", "Lin", "Lin"),
        ],
        # --------------------------------------
        # --------------------------------------
        # 2º Bloco: UPX Noção geral do impacto de t e vars crescendo e diminuindo
        #
        "2.1-LinUpx1": [
            ("t1", "F1", "Lin", "Upx1"),
            ("t1", "F1d10", "Lin", "Upx1"),
            ("t1", "F1x10", "Lin", "Upx1"),
            #
            ("t2", "1", "Lin", "Upx1"),
            ("t2x10", "1", "Lin", "Upx1"),
            ("t2d10", "1", "Lin", "Upx1"),
        ],
        "2.2-t2F1-Lin<>Upx1": [
            ("t2", "F1", "Lin", "Upx1"),
            ("t2", "F1", "Upx1", "Upx1"),
            ("t2", "F1", "Upx1", "Lin"),
        ],
        # --------------------------------------
        # --------------------------------------
        # 3º Bloco: Testes de tempo nondim
        # 3.1 - LINLIN t e F1
        "3.1LinLinF1": [
            ("t1", "F1", "Lin", "Lin"),
            ("t2", "F1", "Lin", "Lin"),
            ("t3", "F1", "Lin", "Lin"),
            ("t4", "F1", "Lin", "Lin"),
            ("t5", "F1", "Lin", "Lin"),
            ("t6", "F1", "Lin", "Lin"),
            ("t7", "F1", "Lin", "Lin"),
            ("t8", "F1", "Lin", "Lin"),
            ("t9", "F1", "Lin", "Lin"),
        ],
        "3.1LinLinF1d10": [
            ("t1", "F1d10", "Lin", "Lin"),
            ("t2", "F1d10", "Lin", "Lin"),
            ("t3", "F1d10", "Lin", "Lin"),
            ("t4", "F1d10", "Lin", "Lin"),
            ("t5", "F1d10", "Lin", "Lin"),
            ("t6", "F1d10", "Lin", "Lin"),
            ("t7", "F1d10", "Lin", "Lin"),
            ("t8", "F1d10", "Lin", "Lin"),
            ("t9", "F1d10", "Lin", "Lin"),
        ],
        #
        # 3.2 - Lin Upx
        "3.2LinUpx1F1": [
            ("t1", "F1", "Lin", "Upx1"),
            ("t2", "F1", "Lin", "Upx1"),
            ("t3", "F1", "Lin", "Upx1"),
            ("t4", "F1", "Lin", "Upx1"),
            ("t5", "F1", "Lin", "Upx1"),
            ("t6", "F1", "Lin", "Upx1"),
            ("t7", "F1", "Lin", "Upx1"),
            ("t8", "F1", "Lin", "Upx1"),
            ("t9", "F1", "Lin", "Upx1"),
        ],
        "3.2LinUpx1F1d10": [
            ("t1", "F1d10", "Lin", "Upx1"),
            ("t2", "F1d10", "Lin", "Upx1"),
            ("t3", "F1d10", "Lin", "Upx1"),
            ("t4", "F1d10", "Lin", "Upx1"),
            ("t5", "F1d10", "Lin", "Upx1"),
            ("t6", "F1d10", "Lin", "Upx1"),
            ("t7", "F1d10", "Lin", "Upx1"),
            ("t8", "F1d10", "Lin", "Upx1"),
            ("t9", "F1d10", "Lin", "Upx1"),
        ],
    }

    for block in blocks_to_add:
        NDList.extend(registered_blocs[block])


def get_nondim_scaler(
    process_params: ProcessParams,
    eq_params: altiok_2006_params.Altiok2006Params,
    input_strategy,
    output_strategy,
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

        ts_code = "t1" => No time nondimensionalizations.
        The others are defined in the document.
    """
    t_S, X_S, P_S, S_S, V_S = get_nondim_scaler_values(
        process_params=process_params,
        eq_params=eq_params,
        ts_code=ts_code,
        scalers_code=scalers_code,
    )

    def create_nondim_scaler(strategy):
        to_nondim = None
        from_nondim = None
        etc_params = None
        match strategy:
            case "N":
                pass
            case "Lin":
                to_nondim = NonDimScaler.toNondimLinearScaler
                from_nondim = NonDimScaler.fromNondimLinearScaler
            case "Desv":
                to_nondim = NonDimScaler.toNondimDesvio
                from_nondim = NonDimScaler.fromNondimDesvio
            case "UPx1":
                to_nondim = NonDimScaler.toNondimUpscale
                from_nondim = NonDimScaler.fromNondimUpscale
                etc_params = {"upscale_lowerbound": 1}

        return NonDimScaler(
            name=f"{strategy}-{ts_code}-{scalers_code}",
            t=t_S,
            X=X_S,
            P=P_S,
            S=S_S,
            V=V_S,
            toNondim=to_nondim,
            fromNondim=from_nondim,
            etc_params=etc_params,
            strategy_str=strategy,
            ts_code=ts_code,
            scalers_code=scalers_code,
        )

    input_nondim_scaler = create_nondim_scaler(input_strategy)
    output_nondim_scaler = create_nondim_scaler(output_strategy)

    return input_nondim_scaler, output_nondim_scaler


def loss_weights(config: str):
    # Sempre cadastra XPSV mesmo que não vá usar todos
    # e lá por dentro eu me resolvo
    lw = []
    match config:
        # X P S V X0 P0 S0 V0
        case "A1":
            lw = [1, 1, 1, 1, 1, 1, 1, 1]
        case "X10":
            lw = [10, 1, 1, 1, 10, 1, 1, 1]
        case "P10":
            lw = [1, 10, 1, 1, 1, 10, 1, 1]
        case "S10":
            lw = [1, 1, 10, 1, 1, 1, 10, 1]
        case "V10":
            lw = [1, 1, 1, 10, 1, 1, 1, 10]
        case "B":  # Focus on X and V
            lw = [100, 1, 1, 100, 1, 1, 1, 1]
        case "B2":  # Focus on X and V including initial conditions
            lw = [100, 1, 1, 100, 10, 1, 1, 10]
        case "B3":  # Focus on XPS
            lw = [40, 40, 40, 1, 10, 10, 10, 1]

    # Converte para float
    lw = [float(_lossw) for _lossw in lw]

    return lw


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
        case "t2x5":
            ts = process_params.t_final * 5
        case "t2d5":
            ts = process_params.t_final / 5
        case "t2x100":
            ts = process_params.t_final * 100
        case "t2d100":
            ts = process_params.t_final / 100
        case "t2x10":
            ts = process_params.t_final * 10
        case "t2d10":
            ts = process_params.t_final / 10
        case "t3":
            ts = process_params.t_final * 10
        case "t4":
            ts = process_params.t_final / 10
        case "t5":
            ts = (eq_params.K_S + eq_params.So) / (eq_params.mu_max * eq_params.So)
        case "t6":
            ts = 1 / eq_params.mu_max
        case "t7":
            ts = (
                eq_params.alpha
                * eq_params.So
                / ((eq_params.K_S + eq_params.So) * eq_params.mu_max)
            )
        case "t8":
            ts = (
                (1 / eq_params.Y_PS)
                * eq_params.alpha
                * (eq_params.K_S + eq_params.So)
                / ((eq_params.So) * eq_params.mu_max)
            )
        case "t9":
            # Proíbe que volumes iguais a zero ou valores não numéricos sejam executados
            assert process_params.inlet.volume != 0
            assert 0 * process_params.inlet.volume == 0
            ts = process_params.max_reactor_volume / process_params.inlet.volume
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
        case "F1d5":
            Xs = eq_params.Xm / 5
            Ps = eq_params.Pm / 5
            Ss = eq_params.So / 5
            Vs = process_params.max_reactor_volume / 5
        case "F1d10":
            Xs = eq_params.Xm / 10
            Ps = eq_params.Pm / 10
            Ss = eq_params.So / 10
            Vs = process_params.max_reactor_volume / 10
        case "F1d20":
            Xs = eq_params.Xm / 20
            Ps = eq_params.Pm / 20
            Ss = eq_params.So / 20
            Vs = process_params.max_reactor_volume / 100
        case "F1d100":
            Xs = eq_params.Xm / 100
            Ps = eq_params.Pm / 100
            Ss = eq_params.So / 100
            Vs = process_params.max_reactor_volume / 100
        case _:
            print(scalers_code)
            assert False, "case must exist"

    vals = (ts, Xs, Ps, Ss, Vs)
    # Converte pra float antes de retornar
    return tuple(float(val) for val in vals)


LRs_dict = {
    f"E-{exp}_{num}": num * (10**-exp) for num in range(1, 10) for exp in range(1, 10)
}

ADAM_EPOCHS_dict = {
    f"{num}{'k' if mult==1000 else ''}": num * mult
    for num in range(1, 201)
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

    """

    # Full time discretization
    dict = {
        # "full": [
        #     [0, process_params.t_final],
        # ]
    }

    # Percentual discretization from 0 to "i"
    # O 1º pro tempo o último pras demais variáveis. As var SEMPRE É TUDO!!!
    for max_perc in percent_max_range:
        for min_perc in percent_min_range:
            dict[f"{min_perc}-{max_perc}pa"] = [
                [
                    min_perc / 100,
                    max_perc / 100,
                ],
            ]

    return dict
