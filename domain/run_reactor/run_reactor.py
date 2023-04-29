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
)->PINNReactorModelResults:
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

    # ---------------------------------------
    # ------------- Geometry ----------------
    # ---------------------------------------
    geom = dde.geometry.TimeDomain(
        0, process_params.t_final / solver_params.non_dim_scaler.t_not_tensor
    )

    # ---------------------------------------
    # --- Initial and Boundary Conditions ---
    # ---------------------------------------
    # Boundary : For time = 0, returns initial condition
    def boundary(_, on_initial):
        return on_initial

    ## X
    ic0 = dde.icbc.IC(
        geom,
        lambda x: initial_state.X[0] / solver_params.non_dim_scaler.X,
        boundary,
        component=0,
    )
    ## P
    ic1 = dde.icbc.IC(
        geom,
        lambda x: initial_state.P[0] / solver_params.non_dim_scaler.P,
        boundary,
        component=1,
    )
    ## S
    ic2 = dde.icbc.IC(
        geom,
        lambda x: initial_state.S[0] / solver_params.non_dim_scaler.S,
        boundary,
        component=2,
    )
    ## Volume
    ic3 = dde.icbc.IC(
        geom,
        lambda x: initial_state.volume[0] / solver_params.non_dim_scaler.V,
        boundary,
        component=3,
    )

    # ---------------------------------------
    # --------- Solving the System ----------
    # ---------------------------------------
    data = dde.data.PDE(
        geometry=geom,
        pde=ode_system_preparer.prepare(),
        bcs=[ic0, ic1, ic2, ic3],
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
    loss_weights = [
        # Os 1ºs são da pde, os 3 útilmos do ajuste físico (proibir menor que zero)
        # Equece, não deu certo
        w[0], w[1], w[2], w[3], #w[0],# w[1], w[2], w[3],
        # Esses são do teste eu acho, e os de cima do train? embora não faça o menor sentido...
        w[0], w[1], w[2], w[3],]# solver_params.loss_weights
    
    #------- CUSTOM LOSS --------------
    # REFS:
    # https://github.com/lululxvi/deepxde/issues/174
    # https://github.com/lululxvi/deepxde/issues/504
    # https://github.com/lululxvi/deepxde/issues/467
    loss = None
    loss_version = None
    mini_batch = solver_params.mini_batch #None # Tamanho da mini-batch

    def X_cons(y_true, y_pred):
        print('XIS!')
        print(y_true)
        print(y_pred)
        if y_pred[0]<0:
            return y_pred[0]
        return 0

    if loss_version is None:
        loss = 'MSE'
    elif loss_version == 2:
        def lossv2(y_true, y_pred):
            # Ref: https://towardsdatascience.com/custom-loss-function-in-tensorflow-eebcd7fed17a
            mse = tf.reduce_mean(tf.square(y_pred - y_true))
            # Se menor que 0, retorna o proprio valor
            consservation = tf.reduce_mean(tf.minimum(y_pred, 0))
            # acho que posso fazer o if, só tenho que somar  ou passar o mean de todos
            # https://stackoverflow.com/questions/34875944/how-to-write-a-custom-loss-function-in-tensorflow
            # 
            # conservation = tf.cond(tf.less(y_pred, tf.multiply(y_pred, 0)), lambda: y_pred, lambda: tf.multiply(y_pred, 0))
            return mse + consservation
        loss = lossv2 #['MSE', X_cons] -> Acho que essa sintaxe é quando tá treinando várias coisas
    # TODO como faz???????

    # Caminho pra pasta. Já vem com  a barra ou em branco caso não tenha hyperfolder
    # Pra facilitar a vida e poder botar ele direto
    hyperfolder_path = f'./results/exported/{solver_params.hyperfolder}-' if solver_params.hyperfolder else ''
    loss_history = None
    train_state = None

    start_time = timer()
    ### Step 1: Pre-solving by "L-BFGS"
    if(solver_params.l_bfgs.do_pre_optimization):
        model.compile("L-BFGS", loss_weights=loss_weights, loss=loss)
        model.train()

    ### Step 2: Solving by "adam"
    pde_resampler = None
    if mini_batch:
        pde_resampler = dde.callbacks.PDEPointResampler(period=10)


    if solver_params.adam_epochs:
        model.compile("adam", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss)
        loss_history, train_state = model.train(
            epochs=solver_params.adam_epochs, 
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler]  if pde_resampler else None,
            model_save_path=f'{hyperfolder_path}{solver_params.name}/adam' if solver_params.name else None,
        )


    if solver_params.sgd_epochs:
        model.compile("sgd", lr=solver_params.adam_lr, loss_weights=loss_weights, loss=loss)
        loss_history, train_state = model.train(
            epochs=solver_params.sgd_epochs, 
            display_every=solver_params.adam_display_every,
            callbacks=[pde_resampler]  if pde_resampler else None,
            model_save_path=f'{hyperfolder_path}{solver_params.name}/sgd' if solver_params.name else None,
        )
    ### Step 3: Post optmization
    if(solver_params.l_bfgs.do_post_optimization):
        model.compile("L-BFGS", loss_weights=loss_weights, loss=loss)
        loss_history, train_state = model.train(
            model_save_path=f'{hyperfolder_path}{solver_params.name}/lbfgs post' if solver_params.name else None,
        )
    end_time = timer()
    total_training_time = end_time - start_time

    
    dde.saveplot(loss_history, train_state, issave=True, isplot=False, output_dir=f'{hyperfolder_path}{solver_params.name}')
    model.save(f'{hyperfolder_path}{solver_params.name}/model')

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
        t = solver_params.non_dim_scaler.t_not_tensor*train_state.X_test,
        X = solver_params.non_dim_scaler.X_not_tensor*train_state.best_y[:,0],
        P = solver_params.non_dim_scaler.P_not_tensor*train_state.best_y[:,1],
        S = solver_params.non_dim_scaler.S_not_tensor*train_state.best_y[:,2],
        V = solver_params.non_dim_scaler.V_not_tensor*train_state.best_y[:,3],
        best_step=train_state.best_step,
        best_loss_test=train_state.best_loss_test,
        best_loss_train=train_state.best_loss_train,
        best_y=train_state.best_y,
        best_ystd=train_state.best_ystd,
        best_metrics=train_state.best_metrics,
        total_training_time=total_training_time,
    )
