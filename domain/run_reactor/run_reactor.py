# Foreign imports
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    plot_params: PlotParams,
    f_out_value_calc,
)->PINNReactorModelResults:
    # TODO permitir passar um endereço pra salvar as imagens dos gráficos, loss_function e afins
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
    # ------------ Parameters ---------------
    # ---------------------------------------
    # mu_max = eq_params.mu_max
    # K_S = eq_params.K_S
    # alpha = eq_params.alpha
    # beta = eq_params.beta
    # Y_PS = eq_params.Y_PS
    # ms = eq_params.ms
    # f = eq_params.f
    # h = eq_params.h
    # Pm = eq_params.Pm
    # Xm = eq_params.Xm

    # Time

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
    # TODO checar se a implementação dos weights está correta.
    # Li a documentação e a fonte mas não consegui encontrar explicações de como usar
    # E do pq os números serem o dobro em quantidade dos de entrada (p/8 variáveis são 16 pesos)
    w = solver_params.loss_weights
    loss_weights = [w[0], w[1], w[2], w[3], w[0], w[1], w[2], w[3]]# solver_params.loss_weights
    ### Step 1: Pre-solving by "L-BFGS"
    if(solver_params.l_bfgs.do_pre_optimization):
        model.compile("L-BFGS", loss_weights=loss_weights)
        model.train()

    ### Step 2: Solving by "adam"
    model.compile("adam", lr=solver_params.adam_lr, loss_weights=loss_weights)
    loss_history, train_state = model.train(
        epochs=solver_params.adam_epochs, display_every=solver_params.adam_display_every
    )
    ### Step 3: Post optmization
    if(solver_params.l_bfgs.do_post_optimization):
        model.compile("L-BFGS", loss_weights=loss_weights)
        loss_history, train_state = model.train()

    dde.saveplot(loss_history, train_state, issave=False, isplot=True)

    # ---------------------------------------
    # -------------- PLOTTING ---------------
    # ---------------------------------------
    if plot_params.show_volume_with_flows:
        plt.plot(
            solver_params.non_dim_scaler.t_not_tensor * train_state.X_test,
            solver_params.non_dim_scaler.V_not_tensor * train_state.y_pred_test[:, 3],
            label="reactor volume (L)",
        )
        if plot_params.force_y_lim:
            plt.ylim(
                [
                    -0.5,
                    process_params.max_reactor_volume
                    * 1.2,
                ]
            )
        plt.legend()
        plt.show()

    if plot_params.show_concentrations:
        plt.plot(
            solver_params.non_dim_scaler.t_not_tensor * train_state.X_test,
            solver_params.non_dim_scaler.X_not_tensor * train_state.y_pred_test[:, 0],
            label="conc X",
        )
        plt.plot(
            solver_params.non_dim_scaler.t_not_tensor * train_state.X_test,
            solver_params.non_dim_scaler.P_not_tensor * train_state.y_pred_test[:, 1],
            label="conc P",
        )
        plt.plot(
            solver_params.non_dim_scaler.t_not_tensor * train_state.X_test,
            solver_params.non_dim_scaler.S_not_tensor * train_state.y_pred_test[:, 2],
            label="conc S",
        )
        if plot_params.force_y_lim:
            plt.ylim(
                [
                    -0.5,
                    process_params.inlet.S
                    * 1.2,
                ]
            )
        plt.legend()
        plt.show()

    # ---------------------------------------
    # ------------- FINISHING ---------------
    # ---------------------------------------

    return PINNReactorModelResults(
        model,
        loss_history,
        train_state,
        solver_params,
        eq_params,
        process_params,
        initial_state,
        f_out_value_calc,
        best_step=train_state.best_step,
        best_loss_test=train_state.best_loss_test,
        best_loss_train=train_state.best_loss_train,
        best_y=train_state.best_y,
        best_ystd=train_state.best_ystd,
        best_metrics=train_state.best_metrics,
    )
