"""
LAYER SIZE

# # Muito espalhadas
                # [1] + [8] * 22 + [4],
                # [1] + [4] * 40 + [4],
                # # Muito concentradas
                # [1] + [140] * 2 + [4],
                # [1] + [320] * 1 + [4],
                # Equilibradas
                # [1] + [22] * 3 + [4],
                [1]
                + [12] * 2
                + [4]
                # PRINCIPAL -->>>>>>>>  # [1] + [36] * 4 + [4],
                # [1] + [80] * 5 + [4],

"""

default_adam_epochs = 16000
# default_layer_size = [1] + [36] * 4 + [4]
default_layer_size = [1] + [22] * 4 + [4]
default_num_domain = 800
default_num_test = 1000

# TODO faz 3 configs boas repetindo
# Uma com tanh, uma com selu, uma com swish

#TODO usa isso dde.optimizers.set_LBFGS_options()
#pra ver as configs padrão do lbfgs que to usando e anotar na metodologia
def change_layer_fix_neurons_number(eq_params, process_params):
    # Usar SGD no lugar de adam
    # https://stats.stackexchange.com/questions/365778/what-should-i-do-when-my-neural-network-doesnt-generalize-well
    # https://arxiv.org/abs/1712.07628 
    # TODO refaz esse mesmo nondim e adam. E fim. Cabou-se. Credo.
    #1400 epochs 100n x3x8 800p domain 800p test horrivel erro >10
    #1400 epochs 160 x4x8 300p mb50 e sem mb ruinzão tb
    # Fazendo agora 90x8 que tinha dado certo antes
    # TODO roda um quadrado neuronios vs layers
    # Roda um adam, um sgd, um com nondim. E FIM!!!!
    # AÍ FAÇO OS GRÁFICOS 3D
    func = 'tanh' #'swish'#'tanh'
    # TODO antes de rodar o novo pelo amor passe só 700 pra testar e sem lbfgs
    # é pra demorar < 5 min que foi o quanto demorou pra maior qtde de neuronios
    n_epochs = 10 #100 #1000 #45000
    #neurons = 90
    #layer=4
    dictionary = {}
    layers = [3, 6, 8, 12]
    layers=[3, 6]
    # Vou ter que fazer em 2 partes...
    # neurons = [22, 45, 70,]
    neurons = [70, 22,]
    # neurons = [90, 130, 200,]
    # layers = [3, 12]
    # neurons = [22,70]
    # TODO sgd parece mais suscetível à quebra que adam???
    
    # Anota aqui as variáveis que vão ser suportadas nessa simulação
    # supported_variables = ['X', 'P', 'S', 'V']
    output_variables = ['X', 'P', 'S', 'V']
    input_variables = ['t']

    for n in neurons:
        for l in layers:
            dictionary[f'{n}x{l} {func} sgd'] = {
                # 'layer_size': [1] + [n] * l + [4],
                'layer_size': [len(input_variables)] + [n] * l + [len(output_variables)],
                "sgd_epochs": n_epochs,
            }

    # dictionary = { 
    #     f'{neurons}x{layer} {func} sgd nondim':{
    #         "sgd_epochs": n_epochs,
    #         'layer_size': [1] + [neurons] * layer + [4],
    #         'X_S': eq_params.Xm,
    #         "P_s": eq_params.Pm,
    #         "S_s":eq_params.So,
    #         "V_s": process_params.max_reactor_volume
    #     },
    #     f'{neurons}x{layer} {func} adam nondim':{
    #         "adam_epochs": n_epochs,
    #         'layer_size': [1] + [neurons] * layer + [4],
    #         'X_S': eq_params.Xm,
    #         "P_s": eq_params.Pm,
    #         "S_s":eq_params.So,
    #         "V_s": process_params.max_reactor_volume
    #     },
    #     f'{neurons}x{layer} {func} sgd adam nondim':{
    #         "adam_epochs": int(n_epochs/2),
    #         "sgd_epochs": int(n_epochs/2),
    #         'layer_size': [1] + [neurons] * layer + [4],
    #         'X_S': eq_params.Xm,
    #         "P_s": eq_params.Pm,
    #         "S_s":eq_params.So,
    #         "V_s": process_params.max_reactor_volume,
    #     },
    #     f'{neurons}x{layer} {func} adam nondim W':{
    #         "adam_epochs": n_epochs,
    #         'layer_size': [1] + [neurons] * layer + [4],
    #         'X_S': eq_params.Xm,
    #         "P_s": eq_params.Pm,
    #         "S_s":eq_params.So,
    #         "V_s": process_params.max_reactor_volume,
    #         'w_S':15,
    #         'w_P': 3,
    #     },
    # }


    for key in dictionary:
        dictionary[key]['activation'] = func
        # TODO fez foi piorar em relação a 300............
        # mas tb fiz poucas interações né a sei lá fim
        dictionary[key]['num_domain'] = 300 #600
        dictionary[key]['num_test'] = 300 #600
        dictionary[key]["lbfgs_pre"] = 0
        dictionary[key]["lbfgs_post"] = 0 #2 #0 #3
        dictionary[key]['LR'] = 0.001 #0.001 quebra no 70x3
        dictionary[key]['hyperfolder'] = f'batch 2023_07_24'#f'fb{neurons}n{func}'
        dictionary[key]['isplot'] = True
        dictionary[key]['initializer'] = 'Glorot normal' #GLOROT UNIFORM
        dictionary[key]['output_variables'] = output_variables
        dictionary[key]['input_variables'] = input_variables
        
        dictionary

    return dictionary


def batch_tests_fixed_neurons_number(eq_params, process_params):
    func = "tanh"
    # neurons total fixo em 360?
    dictionary = {
        # -----------------------
        # -----FIXED NEURONS-----
        # -----------------------
        # -------------
        # 720
        # -------------
        f"360x2 {func}": {
            "layer_size": [1] + [360] * 2 + [4],
        },
        f"180x4 {func}": {
            "layer_size": [1] + [180] * 4 + [4],
        },
        f"120x6 {func}": {
            "layer_size": [1] + [120] * 6 + [4],
        },
        f"90x8 {func}": {
            "layer_size": [1] + [90] * 8 + [4],
        },
        f"72x10 {func}": {
            "layer_size": [1] + [72] * 10 + [4],
        },
        # -------------
        # 360
        # -------------
        f"180x2 {func}": {
            "layer_size": [1] + [180] * 2 + [4],
        },
        f"90x4 {func}": {
            "layer_size": [1] + [90] * 4 + [4],
        },
        f"60x6 {func}": {
            "layer_size": [1] + [60] * 6 + [4],
        },
        f"45x8 {func}": {
            "layer_size": [1] + [45] * 8 + [4],
        },
        f"36x10 {func}": {
            "layer_size": [1] + [36] * 10 + [4],
        },
        # -------------
        # 180
        # -------------
        f"90x2 {func}": {
            "layer_size": [1] + [90] * 2 + [4],
        },
        f"45x4 {func}": {
            "layer_size": [1] + [45] * 4 + [4],
        },
        f"30x6 {func}": {
            "layer_size": [1] + [30] * 6 + [4],
        },
        f"23x8 {func}": {
            "layer_size": [1] + [23] * 8 + [4],
        },
        f"18x10 {func}": {
            "layer_size": [1] + [18] * 10 + [4],
        },
    }

    for key in dictionary:
        dictionary[key]["adam_epochs"] = 30000
        dictionary[key]["activation"] = func
        dictionary[key]["num_domain"] = 300
        dictionary[key]["num_test"] = 300
        dictionary[key]["lbfgs_pre"] = False
        dictionary[key]["lbfgs_post"] = True

    return dictionary


def batch_nondim_v2(eq_params, process_params):

    dictionary = {
        "24x3": {
            "layer_size": [1] + [24] * 3 + [4],
        },
        "24x3 X": {"layer_size": [1] + [24] * 3 + [4], "X_s": eq_params.Xo},
        "24x3 nondim": {
            "layer_size": [1] + [24] * 3 + [4],
            "X_s": eq_params.Xo,
            "P_s": eq_params.Po,
            "S_s": eq_params.So,
            "V_s": process_params.max_reactor_volume,
            # 't_s':process_params.t_final
        },
        "24x6": {
            "layer_size": [1] + [24] * 6 + [4],
        },
        "24x6 X": {"layer_size": [1] + [24] * 6 + [4], "X_s": eq_params.Xo},
        "24x6 nondim": {
            "layer_size": [1] + [24] * 6 + [4],
            "X_s": eq_params.Xo,
            "P_s": eq_params.Po,
            "S_s": eq_params.So,
            "V_s": process_params.max_reactor_volume,
            # 't_s':process_params.t_final
        },
        "24x6": {
            "layer_size": [1] + [24] * 9 + [4],
        },
        "24x6 X": {"layer_size": [1] + [24] * 9 + [4], "X_s": eq_params.Xo},
        "24x6 nondim": {
            "layer_size": [1] + [24] * 9 + [4],
            "X_s": eq_params.Xo,
            "P_s": eq_params.Po,
            "S_s": eq_params.So,
            "V_s": process_params.max_reactor_volume,
            # 't_s':process_params.t_final
        },
        # '60x3':{
        #     'layer_size': [1] + [60] * 3 + [4],
        # },
        # '60x3 X':{
        #     'layer_size': [1] + [60] * 3 + [4],
        #     'X_s':eq_params.Xo
        # },
        # '60x3 nondim':{
        #     'layer_size': [1] + [60] * 3 + [4],
        #     'X_s':eq_params.Xo,
        #     'P_s': eq_params.Po,
        #     'S_s': eq_params.So,
        #     'V_s':process_params.max_reactor_volume,
        #     # 't_s':process_params.t_final
        #     },
    }

    for key in dictionary:
        dictionary[key]["adam_epochs"] = 100  # 25000 #70000
        dictionary[key]["activation"] = "tanh"
        dictionary[key]["num_domain"] = 70
        dictionary[key]["num_test"] = 70
        dictionary[key]["lbfgs_pre"] = False
        dictionary[key]["lbfgs_post"] = False  # True

    return dictionary


def iterate_cstr_convergence(eq_params, process_params):

    dictionary = {
        "60x3": {
            "layer_size": [1] + [60] * 3 + [4],
        },
        "60x5": {
            "layer_size": [1] + [60] * 5 + [4],
        },
        "60x7": {
            "layer_size": [1] + [60] * 7 + [4],
        },
        "120x3": {
            "layer_size": [1] + [120] * 3 + [4],
        },
        "120x5": {
            "layer_size": [1] + [120] * 5 + [4],
        },
        "120x7": {
            "layer_size": [1] + [120] * 7 + [4],
        },
        "240x3": {
            "layer_size": [1] + [240] * 3 + [4],
        },
        "240x5": {
            "layer_size": [1] + [240] * 5 + [4],
        },
        "240x7": {
            "layer_size": [1] + [240] * 5 + [4],
        },
    }

    for key in dictionary:
        dictionary[key]["adam_epochs"] = 100  # 70000
        # dictionary[key]["num_domain"] = 1500 #2000
        # dictionary[key]["num_test"] = 3000 #2000
        dictionary[key]["activation"] = "tanh"
        dictionary[key]["num_domain"] = 70
        dictionary[key]["num_test"] = 70
        dictionary[key]["lbfgs_pre"] = False  # True
        dictionary[key]["lbfgs_post"] = True

    return dictionary


def iterate_layer_size_with_caset6(
    eq_params, process_params, use_lbfgs_pre=True, ts_case_num=5
):
    lbfgs_pre = use_lbfgs_pre
    _adam_epochs = 30000
    # _adam_epochs = 10

    dictionary = {
        # Neuronio = 8
        "t_lay1": {
            # 'layer_size': [1] + [4] * 16 + [4],
            "layer_size": [1]
            + [8] * 6
            + [4],
        },
        "t_lay2": {
            # 'layer_size': [1] + [6] * 12 + [4],
            "layer_size": [1]
            + [8] * 10
            + [4],
        },
        "t_lay3": {
            # 'layer_size': [1] + [8] * 10 + [4],
            "layer_size": [1]
            + [8] * 14
            + [4],
        },
        "t_lay4": {
            # 'layer_size': [1] + [42] * 2 + [4],
            "layer_size": [1]
            + [16] * 3
            + [4],
        },
        "t_lay5": {
            "layer_size": [1] + [16] * 4 + [4],
        },
        "t_lay6": {
            "layer_size": [1] + [16] * 6 + [4],
        },
        # Equilibradas: relação neurônio/camada >1 e <=20
        "t_lay7": {
            # 'layer_size': [1] + [22] * 3 + [4],
            "layer_size": [1]
            + [22] * 3
            + [4],
        },
        "t_lay8": {
            # 'layer_size': [1] + [30] * 3 + [4],
            "layer_size": [1]
            + [22] * 4
            + [4],
        },
        "t_lay9": {
            "layer_size": [1] + [22] * 5 + [4],
        },
        "t_lay10": {
            # 'layer_size': [1] + [22] * 3 + [4],
            "layer_size": [1]
            + [32] * 3
            + [4],
        },
        "t_lay11": {
            # 'layer_size': [1] + [30] * 3 + [4],
            "layer_size": [1]
            + [32] * 4
            + [4],
        },
        "t_lay12": {
            "layer_size": [1] + [32] * 5 + [4],
        },
    }

    for key in dictionary:
        dictionary[key]["adam_epochs"] = _adam_epochs
        dictionary[key]["num_domain"] = default_num_domain
        dictionary[key]["lbfgs_pre"] = lbfgs_pre
        dictionary[key]["lbfgs_post"] = False
        # Case 6:
        if ts_case_num == 6:
            dictionary[key]["t_s"] = (1 / eq_params.mu_max,)
        elif ts_case_num == 5:
            dictionary[key]["t_s"] = (1,)
        elif ts_case_num == 3:
            dictionary[key]["t_s"] = (
                eq_params.alpha
                * eq_params.So
                * (eq_params.K_S + eq_params.So)
                / eq_params.mu_max
            )

    return dictionary


def only_case_6_v3_for_ts_CSTR(eq_params):
    lbfgs_pre = False
    # _layer_size = [1] + [22] * 4 + [4]
    _layer_size = [1] + [22] * 3 + [4]
    _adam_epochs = 30000
    d = {
        # "t_6": {
        #     "t_s": 1 / eq_params.mu_max,
        #     'layer_size':_layer_size,
        #     'adam_epochs':_adam_epochs,
        #     'lbfgs_pre':lbfgs_pre,
        #     }
        "t_5": {"t_s": 1, "lbfgs_pre": lbfgs_pre, "adam_epochs": _adam_epochs}
    }


def only_case_6_v3_for_ts(eq_params):
    lbfgs_pre = False
    # _layer_size = [1] + [22] * 4 + [4]
    _layer_size = [1] + [22] * 3 + [4]
    _layer_size = [1] + [32] * 5 + [4]
    _adam_epochs = 30000
    # _adam_epochs = 120000
    # _adam_epochs = 60000
    # _adam_epochs = 10
    d = {
        # "t_6": {
        #     "t_s": 1 / eq_params.mu_max,
        #     'layer_size':_layer_size,
        #     'adam_epochs':_adam_epochs,
        #     'lbfgs_pre':lbfgs_pre,
        #     },
        "t_5": {
            "t_s": 1,
            "layer_size": _layer_size,
            "lbfgs_pre": lbfgs_pre,
            "adam_epochs": _adam_epochs,
        },
    }

    return d


def cases_non_dim(eq_params, process_params):
    lbfgs_pre = False
    # _layer_size = [1] + [32] * 5 + [4]
    # _adam_epochs = 50000
    _layer_size = [1] + [22] * 3 + [4]
    _adam_epochs = 80000

    dic = {
        "n1": {
            "V_s": 1,
            "X_s": 1,
            "P_s": 1,
            "S_s": 1,
        },  # É tudo 1
        "n_2": {
            "V_s": process_params.max_reactor_volume,
        },
        "n_2": {
            "V_s": process_params.inlet.volume
            if process_params.inlet.volume > 0
            else 1,
            "X_s": 1,
            "P_s": 1,
            "S_s": 1,
        },
        "n_3": {
            "V_s": process_params.max_reactor_volume,
            "X_s": eq_params.Xo,
            "P_s": eq_params.Po,
            "S_s": eq_params.So,
        },
        "n_4": {
            "V_s": process_params.max_reactor_volume,
            "X_s": eq_params.Xm,
            "P_s": eq_params.Pm,
            "S_s": eq_params.So,
        },
        "n_5": {
            "V_s": 1,
            "X_s": 1,  # eq_params.Xm, se ligar dá erro, vai tudo pra NaN no Fedbatch
            "P_s": eq_params.Pm,
            "S_s": eq_params.So,
        },
        "n_6": {
            "V_s": process_params.max_reactor_volume,
            "X_s": eq_params.Xm,
            "P_s": 1,
            "S_s": 1,
        },
    }

    for key in dic:
        dic[key]["adam_epochs"] = _adam_epochs
        dic[key]["layer_size"] = _layer_size
        dic[key]["num_domain"] = default_num_domain
        dic[key]["lbfgs_pre"] = lbfgs_pre
        dic[key]["lbfgs_post"] = False
    return dic


def cases_to_try_batch_vary_ts(eq_params, process_params):

    """
    Testa diferentes t_s para layer_size fixa e num_domain=800
    """

    lbfgs_pre = False
    # _layer_size = [1] + [22] * 4 + [4]
    _layer_size = _layer_size = [1] + [22] * 3 + [4]
    _adam_epochs = 30000

    dictionary = {
        "t_1": {
            "t_s": process_params.t_final,
        },
        "t_2": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So))
        },
        "t_3": {
            "t_s": eq_params.alpha
            * eq_params.So
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max
        },
        "t_4": {
            "t_s": (1 / eq_params.Y_PS)
            * eq_params.alpha
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
        },
        "t_5": {
            "t_s": 1
            # process_params.max_reactor_volume / process_params.inlet.volume
            # if process_params.inlet.volume > 0
            # else 1,
        },
        "t_6": {"t_s": 1 / eq_params.mu_max},
    }

    # Now put the default variables in each case:
    for key in dictionary:
        dictionary[key]["adam_epochs"] = _adam_epochs
        dictionary[key]["layer_size"] = _layer_size
        dictionary[key]["num_domain"] = default_num_domain
        dictionary[key]["lbfgs_pre"] = lbfgs_pre
        dictionary[key]["lbfgs_post"] = False

    return dictionary


def NEW_cases_to_try_WEIGHTS():
    lbfgs_pre = False
    # _layer_size = [1] + [22] * 3 + [4]
    _layer_size = [1] + [32] * 5 + [4]
    # _adam_epochs = 30000
    _adam_epochs = 45000

    dictionary = {
        "W1": {
            # TUDO 1
            "w_V": 1,
        },
        "W2": {
            "w_X": 3,
        },
        "W3": {
            "w_P": 3,
        },
        "W4": {
            "w_S": 3,
        },
        "W5": {
            "w_V": 3,
        },
        "W6": {
            "w_X": 3,
            "w_P": 1,
            "w_S": 1,
            "w_V": 3,
        },
        "W7": {
            "w_X": 1,
            "w_P": 3,
            "w_S": 3,
            "w_V": 1,
        },
        "W8": {
            "w_X": 2,
            "w_P": 1,
            "w_S": 3,
            "w_V": 5,
        },
        "W9": {
            "w_X": 5,
            "w_P": 1,
            "w_S": 1,
            "w_V": 10,
        },
    }

    for key in dictionary:
        dictionary[key]["adam_epochs"] = _adam_epochs
        dictionary[key]["layer_size"] = _layer_size
        dictionary[key]["num_domain"] = default_num_domain
        dictionary[key]["lbfgs_pre"] = lbfgs_pre
        dictionary[key]["lbfgs_post"] = False

    return dictionary


def cases_to_try_vs(eq_params, process_params):
    """
    Variando V_s e epochs
    """
    return {
        "case 0": {
            "adam_epochs": 18000,
            "t_s": process_params.t_final,
            "V_s": 1,
        },
        "case 1": {
            "adam_epochs": 18000,
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "V_s": 1,
        },
        "case 2": {
            "adam_epochs": 18000,
            "t_s": eq_params.alpha
            * eq_params.So
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
            "V_s": 1,
        },
    }


def cases_to_try_fed_batch_vs(eq_params, process_params):
    return {
        "default": {
            "adam_epochs": 18000,
            "t_s": 1,
            "V_s": 1,
        },
        "V+ A+ t=": {
            "adam_epochs": 18000,
            "t_s": 1,
            "V_s": process_params.max_reactor_volume,
        },
        "V= A+ t+": {
            "adam_epochs": 18000,
            "t_s": 1,
            "V_s": process_params.max_reactor_volume,
        },
        "V+ A- t=": {
            "adam_epochs": 18000,
            "t_s": 1,
            "V_s": process_params.max_reactor_volume,
        },
    }


def cases_to_try_vs_simplex(eq_params, process_params):
    """
    Poucos parâmetros ts, pra ser mais rápido de rodar e checar problemas
    """
    return {
        "case 1-1": {"t_s": 1, "adam_epochs": 18000},
        "case 1-2": {"t_s": 1, "adam_epochs": 18000},
        "case 2-1": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 18000,
        },
        "case 2-2": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 18000,
        },
    }


def cases_to_try_ts(eq_params, process_params):
    """
    Apenas 2 ts, variando em várias faixas de adam
    """
    return {
        "case 1-1": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 18000,
        },
        "case 1-2": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 18000,
        },
        "case 2-1": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 8000,
        },
        "case 2-2": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 8000,
        },
        "case 3-1": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 800,
        },
        "case 3-2": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 800,
        },
        "case 4-1": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 2800,
        },
        "case 4-2": {
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
            "adam_epochs": 2800,
        },
    }


def cases_to_try_ts_adam_multistep(eq_params, process_params):
    """
    Variando apenas o ts e pochs
    Os 4 casos recomendados pelo amaro variando em 3 faixas de epocjs
    """
    return {
        # "default": {"t": 1},
        "case 0_ad18k": {"adam_epochs": 18000, "t_s": process_params.t_final},
        "case 1_ad18k": {
            "adam_epochs": 18000,
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
        },
        "case 2_ad18k": {
            "adam_epochs": 18000,
            "t_s": eq_params.alpha
            * eq_params.So
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
        },
        "case 3_ad18k": {
            "adam_epochs": 18000,
            "t_s": (1 / eq_params.Y_PS)
            * eq_params.alpha
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
        },
        "case 4_ad18k": {
            "adam_epochs": 18000,
            "t_s": process_params.max_reactor_volume / process_params.inlet.volume
            if process_params.inlet.volume > 0
            else 1,
        },
        "case 0_ad5k": {"adam_epochs": 5000, "t_s": process_params.t_final},
        "case 1_ad5k": {
            "adam_epochs": 5000,
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
        },
        "case 2_ad5k": {
            "adam_epochs": 5000,
            "t_s": eq_params.alpha
            * eq_params.So
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
        },
        "case 3_ad5k": {
            "adam_epochs": 5000,
            "t_s": (1 / eq_params.Y_PS)
            * eq_params.alpha
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
        },
        "case 4_ad5k": {
            "adam_epochs": 5000,
            "t_s": process_params.max_reactor_volume / process_params.inlet.volume
            if process_params.inlet.volume > 0
            else 1,
        },
        "case 0_ad1k": {"adam_epochs": 1000, "t_s": process_params.t_final},
        "case 1_ad1k": {
            "adam_epochs": 1000,
            "t_s": 1
            / (eq_params.mu_max * eq_params.So / (eq_params.K_S + eq_params.So)),
        },
        "case 2_ad1k": {
            "adam_epochs": 1000,
            "t_s": eq_params.alpha
            * eq_params.So
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
        },
        "case 3_ad1k": {
            "adam_epochs": 1000,
            "t_s": (1 / eq_params.Y_PS)
            * eq_params.alpha
            * (eq_params.K_S + eq_params.So)
            / eq_params.mu_max,
        },
        "case 4_ad1k": {
            "adam_epochs": 1000,
            "t_s": process_params.max_reactor_volume / process_params.inlet.volume
            if process_params.inlet.volume > 0
            else 1,
        },
    }
