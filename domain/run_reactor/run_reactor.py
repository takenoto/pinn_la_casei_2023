# Foreign imports
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Local imports
from domain.params.solver_params import SolverParams
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.reactor.cstr_state import CSTRState
from domain.run_reactor.plot_params import PlotParams
from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer

from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults


def run_reactor(
    ode_system_preparer: ODEPreparer,
    solver_params: SolverParams,
    eq_params: Altiok2006Params,
    process_params: ProcessParams,
    initial_state: CSTRState,
    f_out_value_calc,
) -> PINNReactorModelResults:
    # NÃO! O objetivo dessa função deve ser só calcular. printar, salvar, tudo por fora.
    """
    Runs a reactor that supports both outlet and inlet.
    The outlet can be controlled using the f_out_value_calc(max_reactor_volume, f_in_v, volume)

    # O ode system preparer usa as constantes fornecidas para gerar uma função
    ode_system_preparer --> ode_system_preparer(solver_params: SolverParams, eq_params:Altiok2006Params, process_params: ProcessParams, f_out_value_calc)
    |-> retorna o ode-system (a função que foi gerada com as constantes)
    ode_system --> ode_system(x, y)

    Returns the trained model with its loss_history and train_data AND the parameters
    used to achieve these results.
    """

    inputSimulationType = solver_params.inputSimulationType
    outputSimulationType = solver_params.outputSimulationType

    # ---------------------------------------
    # ------------- Geometry ----------------
    # ---------------------------------------
    # TODO acho que é aqui que vou ter que declarar
    # X, P, S e/ou V como parâmetros de entrada...
    # como todos são obrigatórios, basta fazer o contrário dos params
    # o que não tiver lá vai aqui
    # TODO o "t" sempre é o zero, então os outros são o número+1!!!!!
    time_domain = dde.geometry.TimeDomain(
        0, process_params.t_final / solver_params.non_dim_scaler.t_not_tensor
    )

    # TODO agora faz isso do X pras outras
    # ref:
    # https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.html?highlight=geometry
    # ref2:
    # FAQ => SOLVE PARAMETRIC PDES
    # https://deepxde.readthedocs.io/en/latest/user/faq.html
    # aí roda um MUITO simples só pra ver se não vai crashar
    if len(inputSimulationType.order) == 1:
        geom = time_domain  # Isso deixa da forma como estava antes
    elif len(inputSimulationType.order) == 2:
        if inputSimulationType.X:
            print("!!!!!!!!!!!!!!!!!!!!")
            print(eq_params.Xm)
            print(solver_params.non_dim_scaler.X)
            print("\n\n\n")
            dimension_geom = dde.geometry.Interval(
                0, eq_params.Xm[0] / solver_params.non_dim_scaler.X_not_tensor
            )
            # ^ Não tinha dado certo pq eu tava usando Xm e não Xm[0]
            

        elif inputSimulationType.P:
            dimension_geom = dde.geometry.Interval(
                0, eq_params.Pm[0] / solver_params.non_dim_scaler.P_not_tensor
            )
        elif inputSimulationType.S:
            dimension_geom = dde.geometry.Interval(
                0, eq_params.So[0] / solver_params.non_dim_scaler.S_not_tensor
            )
        elif inputSimulationType.V:
            dimension_geom = dde.geometry.Interval(
                0, process_params.max_reactor_volume[0] / solver_params.non_dim_scaler.V_not_tensor
            )
            
        geom = dde.geometry.GeometryXTime(dimension_geom, time_domain)

    # geom = time_domain # Isso deixa da forma como estava antes
    # ---------------------------------------
    # --- Initial and Boundary Conditions ---
    # ---------------------------------------
    # Boundary : For time = 0, returns initial condition
    def boundary(_, on_initial):
        return on_initial

    ## X
    icX = dde.icbc.IC(
        geom,
        lambda x: initial_state.X[0] / solver_params.non_dim_scaler.X,
        boundary,
        component=outputSimulationType.X_index,  # 0
    )
    ## P
    icP = dde.icbc.IC(
        geom,
        lambda x: initial_state.P[0] / solver_params.non_dim_scaler.P,
        boundary,
        component=outputSimulationType.P_index,  # 1,
    )
    ## S
    icS = dde.icbc.IC(
        geom,
        lambda x: initial_state.S[0] / solver_params.non_dim_scaler.S,
        boundary,
        component=outputSimulationType.S_index,  # 2,
    )
    ## Volume
    icV = dde.icbc.IC(
        geom,
        lambda x: initial_state.volume[0] / solver_params.non_dim_scaler.V,
        boundary,
        component=outputSimulationType.V_index,  # 3,
    )

    # ---------------------------------------
    # --------- Solving the System ----------
    # ---------------------------------------
    # bcs=[ic0, ic1, ic2, ic3],
    # agora 0 -> x, 1 -> P, 2 -> S, 3 -> V
    ics = []

    if outputSimulationType.X:
        ics.append(icX)
    if outputSimulationType.P:
        ics.append(icP)
    if outputSimulationType.S:
        ics.append(icS)
    if outputSimulationType.V:
        ics.append(icV)

    if len(inputSimulationType.order) >= 2:
        # ics = dde.icbc.IC(
        #     geom, lambda x:  0*x[:, 0:1], lambda _, on_initial: on_initial, component=0
        #     )
        # TODO agora sim achei algo. Só printava 4 erros pra 3 saídas, agora printa 6, sempre o dobro como era antes.
        # A partir do momento que usei os ics de antes no lugar desse
        # bcs =  dde.icbc.DirichletBC(geom, lambda x:0*x[:, 0:1], lambda _, on_boundary: on_boundary);
        bcs = []
        o_index = 0
        for o in outputSimulationType.order:
            bc = dde.icbc.DirichletBC(
                geom, lambda x: 0, lambda _, on_boundary: on_boundary, component=o_index
            )
            bcs.append(bc)
            o_index += 1

        # FIXME isso aqui copiei e colei do modelo pra ver no que dava
        def fun_bc(x):
            # return 1 - x[:, 0:1]
            return 0 * x[:, 0:1]

        def fun_init(x):
            # return np.exp(-20 * x[:, 0:1])
            return 0 * x[:, 0:1]

        bcs = [
            dde.icbc.DirichletBC(
                geom, fun_bc, lambda _, on_boundary: on_boundary, component=c
            )
            for c in [0, 1, 2]
        ]
        # FIXME taquei zero aí pra ver se anda
        # ics = [
        #     dde.icbc.IC(geom, fun_init, lambda _, on_initial: on_initial, component=c)
        #     for c in [0, 1, 2]
        # ]
        # bc =  dde.icbc.DirichletBC(geom, lambda x:0*x[:, 0:1], lambda _, on_boundary: on_boundary, component=1);
        icsbcs = []
        icsbcs.extend(bcs)
        icsbcs.extend(ics)

        data = dde.data.TimePDE(
            geometryxtime=geom,
            pde=ode_system_preparer.prepare(),
            # talvez fosse aqui
            # ic_bcs=[bcs, ics],
            ic_bcs=icsbcs,
            num_domain=solver_params.num_domain,
            num_boundary=solver_params.num_boundary,
            num_test=solver_params.num_test,
            num_initial=20,
        )
    else:
        data = dde.data.PDE(
            geometry=geom,
            pde=ode_system_preparer.prepare(),
            bcs=ics,
            num_domain=solver_params.num_domain,
            num_boundary=solver_params.num_boundary,
            num_test=solver_params.num_test,
        )

    net = dde.nn.FNN(
        solver_params.layer_size, solver_params.activation, solver_params.initializer
    )
    ## SOLVING
    model = dde.Model(data, net)
    w = solver_params.loss_weights
    loss_weights = []
    # Os pesos vem primeiro todos na ordem depois repetem
    for i in [1, 2]:
        if outputSimulationType.X:
            loss_weights.append(w[0])
        if outputSimulationType.P:
            loss_weights.append(w[1])
        if outputSimulationType.S:
            loss_weights.append(w[2])
        if outputSimulationType.V:
            loss_weights.append(w[3])

    # loss_weights = [
    #     # Os 1ºs são da pde, os 3 útilmos do ajuste físico (proibir menor que zero)
    #     # Equece, não deu certo
    #     w[0], w[1], w[2], w[3], #w[0],# w[1], w[2], w[3],
    #     # Esses são do teste eu acho, e os de cima do train? embora não faça o menor sentido...
    #     w[0], w[1], w[2], w[3],]# solver_params.loss_weights

    # ------- CUSTOM LOSS --------------
    # REFS:
    # https://github.com/lululxvi/deepxde/issues/174
    # https://github.com/lululxvi/deepxde/issues/504
    # https://github.com/lululxvi/deepxde/issues/467
    loss = None
    loss_version = None
    mini_batch = solver_params.mini_batch  # None # Tamanho da mini-batch

    if loss_version is None:
        loss = "MSE"

    # Caminho pra pasta. Já vem com  a barra ou em branco caso não tenha hyperfolder
    # Pra facilitar a vida e poder botar ele direto
    hyperfolder_path = (
        f"./results/exported/{solver_params.hyperfolder}-"
        if solver_params.hyperfolder
        else ""
    )
    loss_history = None
    train_state = None

    start_time = timer()
    ### Step 1: Pre-solving by "L-BFGS"
    if solver_params.l_bfgs.do_pre_optimization >= 1:
        for i in range(solver_params.l_bfgs.do_pre_optimization):
            model.compile("L-BFGS", loss_weights=loss_weights, loss=loss)
            model.train()

    ### Step 2: Solving by "adam"
    pde_resampler = None
    if mini_batch:
        # FIXME a minha mini-batch esse tempo todo tava com period=10 mds mds mds mds mds mds
        # FIXME então nada tava certo AAAAAAAAAAA
        # pde_resampler = dde.callbacks.PDEPointResampler(period=10)
        pde_resampler = dde.callbacks.PDEPointResampler(period=mini_batch)

    # FIXME tirei loss_weights
    # ValueError: Dimensions must be equal, but are 5 and 6 for '{{node mul_36}} = Mul[T=DT_FLOAT](packed, mul_36/y)' with input shapes: [5], [6]
    loss_weights = None

    if solver_params.adam_epochs:
        model.compile(
            "adam", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss
        )
        loss_history, train_state = model.train(
            epochs=solver_params.adam_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            model_save_path=f"{hyperfolder_path}{solver_params.name}/adam"
            if solver_params.name
            else None,
        )

    if solver_params.sgd_epochs:
        model.compile(
            "sgd", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss
        )
        loss_history, train_state = model.train(
            epochs=solver_params.sgd_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            model_save_path=f"{hyperfolder_path}{solver_params.name}/sgd"
            if solver_params.name
            else None,
        )

    ### Step 3: Post optmization
    if solver_params.l_bfgs.do_post_optimization >= 1:
        for i in range(solver_params.l_bfgs.do_post_optimization):
            model.compile("L-BFGS", loss_weights=loss_weights, loss=loss)
            loss_history, train_state = model.train(
                model_save_path=f"{hyperfolder_path}{solver_params.name}/lbfgs post{i}"
                if solver_params.name
                else None,
                callbacks=[pde_resampler] if pde_resampler else None,
            )
    end_time = timer()
    total_training_time = end_time - start_time
    
    # Por algum motivo o plot não funciona nem aqui nem no saveplot de baixo aff
    if solver_params.isplot and False:
        print(loss_history)
        print(loss_history.loss_test)
        print(train_state.X_test)
        print(train_state.X_train)
        dde.saveplot(loss_history, train_state, issave=False, isplot=True)
    dde.saveplot(loss_history, train_state, issave=True, isplot=False, output_dir=f'{hyperfolder_path}{solver_params.name}/plot')
    # dde.saveplot(loss_history, train_state, issave=False, isplot=True)
    model.save(f"{hyperfolder_path}{solver_params.name}/model")

    # ---------------------------------------
    # ------------- FINISHING ---------------
    # ---------------------------------------

    return PINNReactorModelResults(
        model=model,
        model_name=solver_params.name,
        loss_history=loss_history,
        train_state=train_state,
        solver_params=solver_params,
        eq_params=eq_params,
        process_params=process_params,
        initial_state=initial_state,
        f_out_value_calc=f_out_value_calc,
        t=solver_params.non_dim_scaler.t_not_tensor * train_state.X_test,
        X=solver_params.non_dim_scaler.X_not_tensor
        * train_state.best_y[:, outputSimulationType.X_index]
        if outputSimulationType.X
        else None,
        P=solver_params.non_dim_scaler.P_not_tensor
        * train_state.best_y[:, outputSimulationType.P_index]
        if outputSimulationType.P
        else None,
        S=solver_params.non_dim_scaler.S_not_tensor
        * train_state.best_y[:, outputSimulationType.S_index]
        if outputSimulationType.S
        else None,
        V=solver_params.non_dim_scaler.V_not_tensor
        * train_state.best_y[:, outputSimulationType.V_index]
        if outputSimulationType.V
        else None,
        best_step=train_state.best_step,
        best_loss_test=train_state.best_loss_test,
        best_loss_train=train_state.best_loss_train,
        best_y=train_state.best_y,
        best_ystd=train_state.best_ystd,
        best_metrics=train_state.best_metrics,
        total_training_time=total_training_time,
    )
