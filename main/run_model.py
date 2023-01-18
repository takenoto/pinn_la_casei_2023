import numpy as np

from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.params.solver_params import SolverParams
from domain.run_reactor.plot_params import PlotParams
from domain.optimization.ode_system_caller import RunReactorSystemCaller
from domain.optimization.grid_search import grid_search
from domain.optimization.non_dim_scaler import NonDimScaler
from domain.reactor.cstr_state import CSTRState
from domain.flow.concentration_flow import ConcentrationFlow

from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer

import tensorflow as tf


def run_model(
    solver_params_list=None,
    eq_params=None,
    process_params: ProcessParams = None,
    initial_state: CSTRState = None,
    plot_params: PlotParams = None,
    f_out_value_calc=None,
):
    """
    f_out_value_calc --> f_out_value_calc(max_reactor_volume, f_in_v, volume)
    """
    assert f_out_value_calc is not None, "f_out_value_calc is necessary"


    if solver_params_list is None:

        # Preenche com um único item caso nenhuma opção tenha sido passada
        # V1, funciona bem no normal mas V cai
        # quando faz a com normalização, dá tudo errado, vai pra erros NaN
        # o volume fica negativo, etc
        # Mudando esses comentados resolve. A malha tem que ser mais fina????
        solver_params = SolverParams(
            num_domain=600,
            num_boundary=10,
            num_test=1000,
            adam_epochs=8500,
            adam_display_every=300,
            adam_lr=0.0001,
            layer_size=[1] + [36] * 4 + [4],
            activation="tanh",
            initializer="Glorot uniform",
            # Só funciona com esse w volume 10+. Em 4 e 1 não funcionou, porque
            # o volume acaba caindo muito
            w_volume=10,
            # Faz com que não use o modo adimensional
            non_dim_scaler=NonDimScaler(t=1),
        )

        # Teste 1: Se for adimensinal, ainda tem essa dependência de peso pro volume?
        # Curiosamente as concs tem de 0.9 e de 50 mas não pesa tanto
        # e o volume, em 5, esculhamba tudo
        # w_V =1 e V_scale = process_params.max_reactor_volume o volume ainda cai sei controle
        # w_V =9 e V_scale = process_params.max_reactor_volume o volume ainda não tá bom
        # Aparentemente quanto maior o scale_V mais descarrilha e demora a convergir
        solver_params = SolverParams(
            num_domain=600,
            num_boundary=10,
            num_test=1000,
            adam_epochs=8500,
            adam_display_every=300,
            adam_lr=0.0001,
            layer_size=[1] + [36] * 4 + [4],
            activation="tanh",
            initializer="Glorot uniform",
            # Só funciona com esse w volume 10+. Em 4 e 1 não funcionou, porque
            # o volume acaba caindo muito
            w_volume=9,
            # Faz com que não use o modo adimensional
            non_dim_scaler=NonDimScaler(V=process_params.max_reactor_volume * 5, t=1),
        )
        
        solver_params_list = [solver_params]
        
        #---------------------------------------
        #---------------------------------------
        # NOVO
        # Fração pela qual os valores serão divididos dentro do loop. Precisa ser >= 1 e inteiro
        frac = 1 # 10
        w_frac = 1 # fração a se testar os weights do retorno da rede
        solver_params_list = [
            SolverParams(
                num_domain=600,
                num_boundary=10,
                num_test=1000,
                adam_epochs=8500,
                adam_display_every=3000,
                adam_lr=0.0001,
                layer_size=layer_size,
                activation="tanh",
                initializer="Glorot uniform",
                w_X=X_weight,
                w_P=P_weight,
                w_S=S_weight,
                w_volume=V_weight,
                non_dim_scaler=NonDimScaler(
                    X=eq_params.Xm*(X_scaler / frac),
                    P=eq_params.Pm*(P_scaler / frac),
                    S=eq_params.So*(S_scaler / frac),
                    V=process_params.max_reactor_volume * (V_scaler / frac),
                    t=process_params.t_final * (t_scaler / frac),
                ),
            )
            for layer_size in [
                # Muito espalhadas
                [1] + [8] * 22 + [4],
                [1] + [4] * 40 + [4],
                # Muito concentradas
                [1] + [140] * 2 + [4],
                [1] + [320] * 1 + [4],
                # Equilibradas
                [1] + [22] * 3 + [4],                
                [1] + [36] * 4 + [4],                
                [1] + [80] * 5 + [4],                
            ]
            for X_scaler in range(1, frac+1)
            for P_scaler in range(1, frac+1)
            for S_scaler in range(1, frac+1)
            for V_scaler in range(1, frac+1)
            for t_scaler in range(1, frac+1)
            for X_weight in range(1, w_frac+1)
            for P_weight in range(1, w_frac+1)
            for S_weight in range(1, w_frac+1)
            for V_weight in range(1, w_frac+1)
        ]

    # ---------------------------------------

    # Create a default initial state if it is not given
    initial_state = (
        initial_state
        if initial_state
        else CSTRState(
            volume=np.array([4]),
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )
    )

    if process_params is None:
        # If the inlet flow is not specified, it is assumed that it is 0 L/h
        inlet = ConcentrationFlow(
            volume=0,
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )
        process_params = ProcessParams(max_reactor_volume=5, inlet=inlet, t_final=16)
    
    # Se plot params for nulo cria um padrão
    plot_params = plot_params if plot_params else PlotParams(force_y_lim=True)

    ode_system_preparer = ODEPreparer(
        solver_params, eq_params, process_params, f_out_value_calc
    )

    # Define um caller para os parâmetros atuais (só necessita do solver depois)
    run_reactor_system_caller = RunReactorSystemCaller(
        ode_system_preparer=ode_system_preparer,
        eq_params=eq_params,
        process_params=process_params,
        initial_state=initial_state,
        plot_params=plot_params,  
        f_out_value_calc=f_out_value_calc,
    )

    return grid_search(
        pinn_system_caller=run_reactor_system_caller,
        solver_params_list=solver_params_list,
    )
