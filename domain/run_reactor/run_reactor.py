# LOG
# 2023-08-12 => Removi o save pra cada modelo individual de cada tipo de treinamento.
# Agora salva só o final. Esses modelos pesam muito no HD, não vale à pena.

# Foreign imports
import deepxde as dde
from timeit import default_timer as timer
import numpy as np


import tensorflow as tf

# Local imports
from domain.params.solver_params import SolverParams
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.reactor.reactor_state import ReactorState
from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer

from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults


def run_reactor(
    ode_system_preparer: ODEPreparer,
    solver_params: SolverParams,
    eq_params: Altiok2006Params,
    process_params: ProcessParams,
    initial_state: ReactorState,
    f_out_value_calc,
) -> PINNReactorModelResults:
    # NÃO! O objetivo dessa função deve ser só calcular. printar, salvar, tudo por fora.
    """
    Runs a reactor that supports both outlet and inlet.
    The outlet can be controlled using the
    f_out_value_calc(max_reactor_volume, f_in_v, volume)

    # O ode system preparer usa as constantes fornecidas para gerar uma função
    ode_system_preparer --> ode_system_preparer(solver_params: SolverParams,
    eq_params:Altiok2006Params, process_params: ProcessParams, f_out_value_calc)
    |-> retorna o ode-system (a função que foi gerada com as constantes)
    ode_system --> ode_system(x, y)

    Returns the trained model with its loss_history and train_data AND the parameters
    used to achieve these results.
    """

    inputSimulationType = solver_params.inputSimulationType
    _out = solver_params.outputSimulationType

    # ---------------------------------------
    # ------------- Geometry ----------------
    # ---------------------------------------
    # X, P, S e/ou V como parâmetros de entrada...
    # como todos são obrigatórios, basta fazer o contrário dos params
    # o que não tiver lá vai aqui
    time_domain = dde.geometry.TimeDomain(
        solver_params.train_input_range[0][0] * process_params.t_final,
        solver_params.train_input_range[0][-1] * process_params.t_final,
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
        XPSVboundsMultipliers = {
            "X": eq_params.Xm,
            "P": eq_params.Pm,
            "S": initial_state.S,
            "V": process_params.max_reactor_volume,
        }

        def getNondimBoundary(N):
            "N is the variable (X, P, S, V, etc)"
            minVal = 0.0
            maxVal = XPSVboundsMultipliers[N]
            return minVal, maxVal

        if inputSimulationType.X:
            minVal, maxVal = getNondimBoundary("X")

        elif inputSimulationType.P:
            minVal, maxVal = getNondimBoundary("P")

        elif inputSimulationType.S:
            minVal, maxVal = getNondimBoundary("S")

        elif inputSimulationType.V:
            minVal, maxVal = getNondimBoundary("V")

        dimension_geom = dde.geometry.Interval(minVal, maxVal)
        geom = dde.geometry.GeometryXTime(dimension_geom, time_domain)

    # ---------------------------------------
    # --- Initial and Boundary Conditions ---
    # ---------------------------------------

    # Boundary : For time = 0, returns initial condition
    def boundary(_, on_initial):
        return on_initial

    # Initial condition, with dimension
    N0_dim = {
        "X": initial_state.X,
        "P": initial_state.P,
        "S": initial_state.S,
        "V": initial_state.volume,
    }

    ## X
    icX = dde.icbc.IC(
        geom,
        lambda x: N0_dim["X"],
        boundary,
        component=_out.X_index,  # 0
    )
    ## P
    icP = dde.icbc.IC(
        geom,
        lambda x: N0_dim["P"],
        boundary,
        component=_out.P_index,  # 1,
    )
    ## S
    icS = dde.icbc.IC(
        geom,
        lambda x: N0_dim["S"],
        boundary,
        component=_out.S_index,  # 2,
    )
    ## Volume
    icV = dde.icbc.IC(
        geom,
        lambda x: N0_dim["V"],
        boundary,
        component=_out.V_index,  # 3,
    )

    # ---------------------------------------
    # --------- Solving the System ----------
    # ---------------------------------------
    # bcs=[ic0, ic1, ic2, ic3],
    # agora 0 -> x, 1 -> P, 2 -> S, 3 -> V
    ics = []

    if _out.X:
        ics.append(icX)
    if _out.P:
        ics.append(icP)
    if _out.S:
        ics.append(icS)
    if _out.V:
        ics.append(icV)

    # Setting the data for the NN

    if geom == time_domain:
        data = dde.data.PDE(
            geometry=geom,
            pde=ode_system_preparer.prepare(),
            bcs=ics,
            num_domain=solver_params.num_domain,
            num_boundary=solver_params.num_init,
            num_test=solver_params.num_test,
            train_distribution=solver_params.train_distribution,
        )
    else:
        data = dde.data.TimePDE(
            geometryxtime=geom,
            pde=ode_system_preparer.prepare(),
            ic_bcs=ics,
            num_domain=solver_params.num_domain,
            num_initial=solver_params.num_init,
            num_boundary=solver_params.num_boundary,
            num_test=solver_params.num_test,
            train_distribution=solver_params.train_distribution,
        )
    # ---------------------------------
    # Creating the model and the net
    # ---------------------------------
    net = dde.nn.FNN(
        solver_params.layer_size, solver_params.activation, solver_params.initializer
    )

    #
    # TRANSFORM --------------------
    #
    # INPUT
    def input_transform(x):
        transformed_inputs = []
        for N in inputSimulationType.order:
            N_input_index = inputSimulationType.get_index_for(N)
            if N_input_index is not None:
                N_dim = x[:, N_input_index : N_input_index + 1]
                N_val = solver_params.input_non_dim_scaler.toNondim({N: N_dim}, N)
                transformed_inputs.append(N_val)

        return tf.concat(transformed_inputs, axis=1)

    net.apply_feature_transform(input_transform)

    # OUTPUT
    def output_transform(x, y):
        transformed_outputs = []
        for N in _out.order:
            N_output_index = _out.get_index_for(N)
            # If it is not None, then it is valid
            # Era aqui o problema. Se for if is ignora o 0 também, não só nones
            # !!!!!!!!! que comportamento questionvel pelo amor viu.
            if N_output_index is not None:
                N_nondim = y[:, N_output_index : N_output_index + 1]
                N_val = solver_params.output_non_dim_scaler.fromNondim({N: N_nondim}, N)
                transformed_outputs.append(N_val)

        # --------------------------------
        # --------------------------------
        return tf.concat(transformed_outputs, axis=1)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    # ------- CUSTOM LOSS --------------
    # REFS:
    # https://github.com/lululxvi/deepxde/issues/174
    # https://github.com/lululxvi/deepxde/issues/504
    # https://github.com/lululxvi/deepxde/issues/467
    loss = "MSE"  # "MSE"

    # para definir uma loss custom, por exemplo, poderia ser feito algo do tipo:
    # def loss_test(y_true, y_pred):
    #     return tf.math.reduce_mean(tf.abs(y_true - y_pred))
    # loss = loss_test

    metrics = None  # ["l2 relative error"] # ["MSE"]
    resample_every = solver_params.resample_every  # None # Tamanho da mini-batch

    # Caminho pra pasta. Já vem com  a barra ou em branco caso não tenha hyperfolder
    # Pra facilitar a vida e poder botar ele direto
    hyperfolder_path = solver_params.hyperfolder
    loss_history = None
    train_state = None

    # ---------------------------
    ## SOLVING
    # ---------------------------
    # A lws SEMPRE terá todas, na ordem: XPSV
    # Loss Weights
    loss_weights = []
    lws = solver_params.loss_weights

    def refactor_loss_to_output_format():
        "Transforms the loss from 0-4 to 0-N where N is the number of output variables"
        loss_weights.clear()
        # Sempre é XPSV, e aqui converte pra as saídas que tem de fato:
        if _out.X:
            loss_weights.append(lws.values[0])
        if _out.P:
            loss_weights.append(lws.values[1])
        if _out.S:
            loss_weights.append(lws.values[2])
        if _out.V:
            loss_weights.append(lws.values[3])
        # Boundary conditions
        if _out.X:
            loss_weights.append(lws.values[0 + int(len(lws.values) / 2)])
        if _out.P:
            loss_weights.append(lws.values[1 + int(len(lws.values) / 2)])
        if _out.S:
            loss_weights.append(lws.values[2 + int(len(lws.values) / 2)])
        if _out.V:
            loss_weights.append(lws.values[3 + int(len(lws.values) / 2)])

    refactor_loss_to_output_format()

    # O número que tentaremos deixar todas as loss próximas escalonando
    scale_to = lws.settings["scale_to"]
    multi_exponent_op = lws.settings.get("multi_exponent_op", 1)

    def update_lw_values(loss_history, update_ICs=False):
        for N in ["X", "P", "S", "V"]:
            if N in _out.order:
                #
                # Outputs normal
                N_index = _out.get_index_for(N)
                ics_index = N_index + len(_out.order)
                loss_first_it = np.array(loss_history.loss_test)[
                    :, N_index : N_index + 1
                ].tolist()[-1][0]
                #
                # Prevent division by 0
                multiplier = scale_to
                if loss_first_it > 0:
                    multiplier = scale_to / loss_first_it
                #
                # -----------------------
                # ICs
                loss_first_it_IC = np.array(loss_history.loss_test)[
                    :, ics_index : ics_index + 1
                ].tolist()[-1][0]
                #
                # Prevent float division by zero
                multiplier_IC = scale_to
                if loss_first_it_IC > 0:
                    multiplier_IC = scale_to / loss_first_it_IC

                # Index de cada um no loss weights que tem tudo:
                base_index = {"X": 0, "P": 1, "S": 2, "V": 3}

                lws.values[base_index[N]] = multiplier**multi_exponent_op

                if update_ICs:
                    lws.values[base_index[N] + 4] = multiplier_IC**multi_exponent_op
                else:
                    pass  # do not update

    #
    # LWs Auto
    #
    # Os pesos vem primeiro todos na ordem depois repetem
    if lws.type == "auto":
        model.compile(
            "adam",
            lr=solver_params.adam_lr,
            loss_weights=loss_weights,
            loss=loss,
            metrics=metrics,
        )
        loss_history, train_state = model.train(
            iterations=1,
            display_every=1,
        )

        update_lw_values(loss_history=loss_history, update_ICs=True)
        # Agora ajusta pra usar daqui pra baixo novamente
        refactor_loss_to_output_format()

    #
    # Initial Conditions pre trainning
    if solver_params.pre_train_ics_epochs is not None:
        # Zera tudo que não for IC
        ics_only_lw = []
        for l_index in range(len(loss_weights)):
            if l_index < int(len(loss_weights) / 2):
                ics_only_lw.append(0)
            else:
                ics_only_lw.append(1)

        model.compile(
            "adam",
            lr=solver_params.adam_lr,
            loss_weights=ics_only_lw,
            loss=loss,
            metrics=metrics,
        )
        loss_history, train_state = model.train(
            iterations=solver_params.pre_train_ics_epochs,
            display_every=solver_params.adam_display_every,
        )

    #
    # LWs autic
    # autic => otimização é feita após pré-treino e NÃO leva em consideração ICs
    if lws.type == "autic":
        model.compile(
            "adam",
            lr=solver_params.adam_lr,
            loss_weights=loss_weights,
            loss=loss,
            metrics=metrics,
        )
        loss_history, train_state = model.train(
            iterations=1,
            display_every=1,
        )
        # Index a partir do qual começam as loss de ics e bcs
        update_lw_values(loss_history=loss_history, update_ICs=False)
        # Agora ajusta pra usar daqui pra baixo novamente
        refactor_loss_to_output_format()

    start_time = timer()
    ### Step 1: Pre-solving by "L-BFGS"
    if solver_params.l_bfgs.do_pre_optimization >= 1:
        for i in range(solver_params.l_bfgs.do_pre_optimization):
            model.compile(
                "L-BFGS", loss_weights=loss_weights, loss=loss, metrics=metrics
            )
            model.train()

    ### Step 2: Solving by "adam"
    pde_resampler = None
    if resample_every:
        pde_resampler = dde.callbacks.PDEPointResampler(period=resample_every)

    if solver_params.adam_epochs:
        model.compile(
            "adam",
            lr=solver_params.adam_lr,
            loss_weights=loss_weights,
            loss=loss,
            metrics=metrics,
        )
        loss_history, train_state = model.train(
            iterations=solver_params.adam_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            disregard_previous_best=True,
            # model_save_path=f"{hyperfolder_path}{solver_params.name}/adam"
            # if solver_params.name
            # else None,
        )

    if solver_params.sgd_epochs:
        model.compile(
            "sgd",
            lr=solver_params.adam_lr,
            loss_weights=loss_weights,
            loss=loss,
            metrics=metrics,
        )
        loss_history, train_state = model.train(
            iterations=solver_params.sgd_epochs,
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler] if pde_resampler else None,
            # model_save_path=f"{hyperfolder_path}{solver_params.name}/sgd"
            # if solver_params.name
            # else None,
        )

    ### Step 3: Post optmization
    if solver_params.l_bfgs.do_post_optimization >= 1:
        for i in range(solver_params.l_bfgs.do_post_optimization):
            model.compile(
                "L-BFGS", loss_weights=loss_weights, loss=loss, metrics=metrics
            )
            loss_history, train_state = model.train(
                callbacks=[pde_resampler] if pde_resampler else None,
                display_every=solver_params.adam_display_every,
            )
    end_time = timer()
    total_training_time = end_time - start_time

    # Por algum motivo o plot não funciona nem aqui nem no saveplot de baixo aff
    if solver_params.isplot:
        dde.saveplot(loss_history, train_state, issave=False, isplot=True)
    if solver_params.is_save_model:
        model.save(f"{hyperfolder_path}/models/{solver_params.name}")
        dde.saveplot(
            loss_history,
            train_state,
            issave=True,
            isplot=False,
            output_dir=f"{hyperfolder_path}/models/{solver_params.name}",
        )
    # dde.saveplot(loss_history, train_state, issave=False, isplot=True)

    # ---------------------------------------
    # ------------- FINISHING ---------------
    # ---------------------------------------

    pinn = PINNReactorModelResults(
        model=model,
        model_name=solver_params.name,
        loss_history=loss_history,
        train_state=train_state,
        solver_params=solver_params,
        eq_params=eq_params,
        process_params=process_params,
        initial_state=initial_state,
        f_out_value_calc=f_out_value_calc,
        best_step=train_state.best_step,
        best_loss_test=train_state.best_loss_test,
        best_loss_train=train_state.best_loss_train,
        best_y=train_state.best_y,
        best_ystd=train_state.best_ystd,
        best_metrics=train_state.best_metrics,
        total_training_time=total_training_time,
    )

    solver_params.save_caller.save_pinn(pinn=pinn, folder_to_save=hyperfolder_path)
    pinn.model = None  # to tentando esvaziar a memória
    # o consumo vai de  600 mb (vscode normal) para até ~9gb conforme mais modelos são
    # executados, o que não faz nenhum sentido????

    return pinn
