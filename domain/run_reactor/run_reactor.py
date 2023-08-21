# LOG
# 2023-08-12 => Removi o save pra cada modelo individual de cada tipo de treinamento.
#               Agora salva só o final. Esses modelos pesam muito no HD, não vale à pena.


# Foreign imports
import deepxde as dde
from timeit import default_timer as timer

# Local imports
from domain.params.solver_params import SolverParams
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.reactor.cstr_state import CSTRState
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

    scaler = solver_params.non_dim_scaler

    # ---------------------------------------
    # ------------- Geometry ----------------
    # ---------------------------------------
    # X, P, S e/ou V como parâmetros de entrada...
    # como todos são obrigatórios, basta fazer o contrário dos params
    # o que não tiver lá vai aqui
    time_domain = dde.geometry.TimeDomain(
        0, scaler.toNondim({"t": process_params.t_final}, "t")
    )

    # ref:
    # https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.html?highlight=geometry
    # ref2:
    # FAQ => SOLVE PARAMETRIC PDES
    # https://deepxde.readthedocs.io/en/latest/user/faq.html
    # aí roda um MUITO simples só pra ver se não vai crashar

    if len(inputSimulationType.order) == 1:
        geom = time_domain  # Isso deixa da forma como estava antes
    elif len(inputSimulationType.order) == 2:
        # Valores adimensionalizados das variáveis em t0
        Nondim_boundaries = {
            "X": scaler.toNondim({"X": eq_params.Xm[0]}, "X"),
            "P": scaler.toNondim({"P": eq_params.Pm[0]}, "P"),
            "S": scaler.toNondim({"S": eq_params.So[0]}, "S"),
            "V": scaler.toNondim({"V": process_params.max_reactor_volume}, "V"),
        }

        if inputSimulationType.X:
            dimension_geom = dde.geometry.Interval(0, Nondim_boundaries["X"])

        elif inputSimulationType.P:
            dimension_geom = dde.geometry.Interval(0, Nondim_boundaries["P"])
        elif inputSimulationType.S:
            dimension_geom = dde.geometry.Interval(0, Nondim_boundaries["S"])
        elif inputSimulationType.V:
            dimension_geom = dde.geometry.Interval(0, Nondim_boundaries["V"])

        geom = dde.geometry.GeometryXTime(dimension_geom, time_domain)

    # ---------------------------------------
    # --- Initial and Boundary Conditions ---
    # ---------------------------------------

    # Boundary : For time = 0, returns initial condition
    def boundary(_, on_initial):
        return on_initial

    # Initial condition, with dimension
    N0_dim = {
        "X": initial_state.X[0],
        "P": initial_state.P[0],
        "S": initial_state.S[0],
        "V": initial_state.volume[0],
    }
    
    # Initial conditions (XPSV), without dimension
    N0_nondim = {type: scaler.toNondim(N0_dim, type) for type in N0_dim}


    ## X
    icX = dde.icbc.IC(
        geom,
        lambda x: N0_nondim["X"],
        boundary,
        component=outputSimulationType.X_index,  # 0
    )
    ## P
    icP = dde.icbc.IC(
        geom,
        lambda x: N0_nondim["P"],
        boundary,
        component=outputSimulationType.P_index,  # 1,
    )
    ## S
    icS = dde.icbc.IC(
        geom,
        lambda x: N0_nondim["S"],
        boundary,
        component=outputSimulationType.S_index,  # 2,
    )
    ## Volume
    icV = dde.icbc.IC(
        geom,
        lambda x: N0_nondim["V"],
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

    # Setting the data for the NN
    data = dde.data.PDE(
        geometry=geom,
        pde=ode_system_preparer.prepare(),
        bcs=ics,
        num_domain=solver_params.num_domain,
        num_boundary=solver_params.num_init,
        num_test=solver_params.num_test,
    )

    # Creating the model and the net
    net = dde.nn.FNN(
        solver_params.layer_size, solver_params.activation, solver_params.initializer
    )
    model = dde.Model(data, net)
    
    # Loss Weights
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

    #---------------------------
    ## SOLVING
    #---------------------------
    
    start_time = timer()
    ### Step 1: Pre-solving by "L-BFGS"
    if solver_params.l_bfgs.do_pre_optimization >= 1:
        for i in range(solver_params.l_bfgs.do_pre_optimization):
            model.compile("L-BFGS", loss_weights=loss_weights, loss=loss)
            model.train()

    ### Step 2: Solving by "adam"
    pde_resampler = None
    if mini_batch:
        pde_resampler = dde.callbacks.PDEPointResampler(period=mini_batch)

    if solver_params.adam_epochs:
        model.compile(
            "adam", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss
        )
        loss_history, train_state = model.train(
            epochs=solver_params.adam_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            # model_save_path=f"{hyperfolder_path}{solver_params.name}/adam"
            # if solver_params.name
            # else None,
        )

    if solver_params.sgd_epochs:
        model.compile(
            "sgd", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss
        )
        loss_history, train_state = model.train(
            epochs=solver_params.sgd_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            # model_save_path=f"{hyperfolder_path}{solver_params.name}/sgd"
            # if solver_params.name
            # else None,
        )

    ### Step 3: Post optmization
    if solver_params.l_bfgs.do_post_optimization >= 1:
        for i in range(solver_params.l_bfgs.do_post_optimization):
            model.compile("L-BFGS", loss_weights=loss_weights, loss=loss)
            loss_history, train_state = model.train(
                # model_save_path=f"{hyperfolder_path}{solver_params.name}/lbfgs post{i}"
                # if solver_params.name
                # else None,
                callbacks=[pde_resampler] if pde_resampler else None,
                display_every=solver_params.adam_display_every,
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
    model.save(f"{hyperfolder_path}{solver_params.name}/model")
    dde.saveplot(
        loss_history,
        train_state,
        issave=True,
        isplot=False,
        output_dir=f"{hyperfolder_path}{solver_params.name}/plot",
    )
    # dde.saveplot(loss_history, train_state, issave=False, isplot=True)

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


# ------------------------------------------------------------------------
# ---------------------------- BACKUP // OLD -----------------------------
# ------------------------------------------------------------------------


def run_reactor1Backup(
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
    print(f"loss version = {solver_params.loss_version}")

    inputSimulationType = solver_params.inputSimulationType
    outputSimulationType = solver_params.outputSimulationType

    # ---------------------------------------
    # ------------- Geometry ----------------
    # ---------------------------------------
    # X, P, S e/ou V como parâmetros de entrada...
    # como todos são obrigatórios, basta fazer o contrário dos params
    # o que não tiver lá vai aqui
    time_domain = dde.geometry.TimeDomain(
        0, process_params.t_final / solver_params.non_dim_scaler.t_not_tensor
    )

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
                0,
                process_params.max_reactor_volume
                / solver_params.non_dim_scaler.V_not_tensor,
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
        bcs = []
        o_index = 0
        for o in outputSimulationType.order:
            bc = dde.icbc.DirichletBC(
                geom, lambda x: 0, lambda _, on_boundary: on_boundary, component=o_index
            )
            bcs.append(bc)
            o_index += 1

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
        # bc =  dde.icbc.DirichletBC(geom, lambda x:0*x[:, 0:1], lambda _, on_boundary: on_boundary, component=1);
        icsbcs = []
        # Removi bcs. Não temos bcs de fato...
        # Porque o eixo é o próprio X, não faz sentido.
        # icsbcs.extend(bcs)
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
            num_initial=solver_params.num_init,
        )
    else:
        data = dde.data.PDE(
            geometry=geom,
            pde=ode_system_preparer.prepare(),
            bcs=ics,
            num_domain=solver_params.num_domain,
            num_boundary=solver_params.num_init,
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
        pde_resampler = dde.callbacks.PDEPointResampler(period=mini_batch)

    if solver_params.adam_epochs:
        model.compile(
            "adam", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss
        )
        loss_history, train_state = model.train(
            epochs=solver_params.adam_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            # model_save_path=f"{hyperfolder_path}{solver_params.name}/adam"
            # if solver_params.name
            # else None,
        )

    if solver_params.sgd_epochs:
        model.compile(
            "sgd", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss
        )
        loss_history, train_state = model.train(
            epochs=solver_params.sgd_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            # model_save_path=f"{hyperfolder_path}{solver_params.name}/sgd"
            # if solver_params.name
            # else None,
        )

    ### Step 3: Post optmization
    if solver_params.l_bfgs.do_post_optimization >= 1:
        for i in range(solver_params.l_bfgs.do_post_optimization):
            model.compile("L-BFGS", loss_weights=loss_weights, loss=loss)
            loss_history, train_state = model.train(
                # model_save_path=f"{hyperfolder_path}{solver_params.name}/lbfgs post{i}"
                # if solver_params.name
                # else None,
                callbacks=[pde_resampler] if pde_resampler else None,
                display_every=solver_params.adam_display_every,
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
    model.save(f"{hyperfolder_path}{solver_params.name}/model")
    dde.saveplot(
        loss_history,
        train_state,
        issave=True,
        isplot=False,
        output_dir=f"{hyperfolder_path}{solver_params.name}/plot",
    )
    # dde.saveplot(loss_history, train_state, issave=False, isplot=True)

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
