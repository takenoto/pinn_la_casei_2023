
# TODO testar adimensionalização DO TEMPO APENAS novamente mds... Repete settings
# => Na verdade acho que faz duplicata. Um sem adim e um com. Mas comece com o com
# pq se quebrar já mudo as settings. Isso no batch mesmo...
# TODO CSTR? O que eu tava fazendo que esse 30x10 pareceu bom?????
# TODO comece fazendo um teste geral, com mais neurônios e camadas, pra investigar a bacia de atração do amaro
# Só depois caço essas outras marmotas, vejo o que se saiu melhor  pego pontos próximos...

def change_layer_fix_neurons_number(eq_params, process_params):
    # Parece ter algo MUITO bom na região próxima de 30x10. Vamos investigar ela agora.
    func = 'tanh' #'swish'
    mini_batch = None #50 #200
    initializer = 'Glorot normal' #'Glorot normal' #'Orthogonal' #GLOROT UNIFORM # Era Glorot Normal nos testes sem swish
    LR = 0.001 
    lbfgs_post = 0 #1 #4
    ADAM_EPOCHS = 100 #35000
    SGD_EPOCHS = None #1000
    dictionary = {}
    neurons =[80, 40]
    layers = [4, 3]
    cols = len(layers)
    rows = len(neurons)
   
    # Se irá aplicar a estratégia de adimensionalização padrão
    IS_NONDIM = False
    
    # Me parece que quando usa 2 var de entrada precisa de um NUM_DOMAIN e teste bem maior pra ficar razoável
    NUM_DOMAIN = 300 #800 #100 #900 #1000
    NUM_TEST = 300 #800 #100 #400 #1000
    NUM_INIT = 60
    NUM_BOUNDARY = 0 #2 #200 #100
 
    # Anota aqui as variáveis que vão ser suportadas nessa simulação
    # supported_variables = ['X', 'P', 'S', 'V']
    # Por padrão, t de entrada e XPSV de saída:
    output_variables = ['X', 'P', 'S', 'V']
    input_variables = ['t']
    # Alternativamente, PSV de saída e tX de entrada
    output_variables = ['P', 'S', 'V']
    input_variables = ['t', 'X']

    for n in neurons:
        for l in layers:
            key = f'{n}x{l} {func}'
            dictionary[key] = {
                'layer_size': [len(input_variables)] + [n] * l + [len(output_variables)],
                "adam_epochs": ADAM_EPOCHS,
                "sgd_epochs": SGD_EPOCHS,
            }
            if (IS_NONDIM):
                # dictionary[key]['t_s'] = process_params.t_final
                dictionary[key]["X_s"] = eq_params.Xm,
                dictionary[key]["P_s"] = eq_params.Pm,
                dictionary[key]["S_s"] = eq_params.So
                dictionary[key]["V_s"] = process_params.max_reactor_volume


    for key in dictionary:
        dictionary[key]['activation'] = func
        if mini_batch:
            dictionary[key]['mini_batch'] = mini_batch
        dictionary[key]['num_domain'] = NUM_DOMAIN
        dictionary[key]['num_test'] = NUM_TEST
        dictionary[key]['num_init'] = NUM_INIT
        dictionary[key]['num_bound'] = NUM_BOUNDARY
        dictionary[key]["lbfgs_pre"] = 0
        dictionary[key]["lbfgs_post"] = lbfgs_post
        dictionary[key]['LR'] = LR
        dictionary[key]['hyperfolder'] = f'cstr  {"ND" if IS_NONDIM else ""} 2023_08_07'
        dictionary[key]['isplot'] = True
        dictionary[key]['initializer'] = initializer
        dictionary[key]['output_variables'] = output_variables
        dictionary[key]['input_variables'] = input_variables
        
        dictionary

    return (dictionary, cols, rows)
