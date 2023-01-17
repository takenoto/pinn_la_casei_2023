class PINNReactorModelResults:
    def __init__(
        self,
        model,
        loss_history,
        train_data,
        solver_params,
        eq_params,
        process_params,
        initial_state,
        f_out_value_calc,
    ):
        self.model = model
        self.loss_history = loss_histry
        self.train_data = train_data
        self.solver_params = solver_params
        """
        The solver params applied to achieve this model
        """
        self.eq_params = eq_params
        self.process_params = process_params
        self.initial_state = initial_state
        self.f_out_value_calc = f_out_value_calc
        """
        Function used to calculate the reactor's outlet flow (volume/time) 
        """


def run_reactor():
    # TODO permitir passar um endereço pra salvar as imagens dos gráficos, loss_function e afins
    """
    Runs a reactor that supports both outlet and inlet.
    The outlet can be controlled using the f_out_value_calc(max_reactor_volume, f_in_v, volume)

    Returns the trained model with its loss_history and train_data AND the parameters
    used to achieve these results.
    """

    # ---------------------------------------
    # ------------ Parameters ---------------
    # ---------------------------------------
    mu_max = eq_params.mu_max
    K_S = eq_params.K_S
    alpha = eq_params.alpha
    beta = eq_params.beta
    Y_PS = eq_params.Y_PS
    ms = eq_params.ms
    f = eq_params.f
    h = eq_params.h
    Pm = eq_params.Pm
    Xm = eq_params.Xm

    # Time
    t_final = process_params.t_final

    # ---------------------------------------
    # ------------- Geometry ----------------
    # ---------------------------------------
    geom = dde.geometry.TimeDomain(0, t_final)

    # ---------------------------------------
    # --- Initial and Boundary Conditions ---
    # ---------------------------------------
    # Boundary : For time = 0, returns initial condition
    def boundary(_, on_initial):
        return on_initial

    ## X
    ic0 = dde.icbc.IC(geom, lambda x: initial_state.X[0], boundary, component=0)
    ## P
    ic1 = dde.icbc.IC(geom, lambda x: initial_state.P[0], boundary, component=1)
    ## S
    ic2 = dde.icbc.IC(geom, lambda x: initial_state.S[0], boundary, component=2)
    ## Volume
    ic3 = dde.icbc.IC(geom, lambda x: initial_state.volume[0], boundary, component=3)

    # ---------------------------------------
    # --------- Solving the System ----------
    # ---------------------------------------
    data = dde.data.PDE(
        geom,
        ode_system,
        [ic0, ic1, ic2, ic3],
        num_domain=solver_params.num_domain,
        num_boundary=solver_params.num_boundary,
        num_test=solver_params.num_test,
    )

    net = dde.nn.FNN(
        solver_params.layer_size, solver_params.activation, solver_params.initializer
    )

    ## SOLVING
    ### Step 1: Pre-solving by "L-BFGS"
    model = dde.Model(data, net)
    model.compile("L-BFGS", lr=0.0001)
    loss_history, train_state = model.train(iterations=40, epochs=40, display_every=40)
    ### Step 2: Solving by "adam"
    model.compile("adam", lr=0.0001)
    loss_history, train_state = model.train(
        epochs=solver_params.adam_epochs, display_every=solver_params.adam_display_every
    )

    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    # ---------------------------------------
    # -------------- PLOTTING ---------------
    # ---------------------------------------
    if plot_params.show_volume_with_flows:
        plt.plot(
            train_state.X_test,
            train_state.y_pred_test[:, 3],
            label="reactor volume (L)",
        )
        if plot_params.force_y_lim:
            plt.ylim([-0.5, process_params.max_reactor_volume * 1.2])
        plt.legend()
        plt.show()

    if plot_params.show_concentrations:
        plt.plot(train_state.X_test, train_state.y_pred_test[:, 0], label="conc X")
        plt.plot(train_state.X_test, train_state.y_pred_test[:, 1], label="conc P")
        plt.plot(train_state.X_test, train_state.y_pred_test[:, 2], label="conc S")
        if plot_params.force_y_lim:
            plt.ylim([-0.5, process_params.inlet.S * 1.2])
        plt.legend()
        plt.show()

    # ---------------------------------------
    # ------------- FINISHING ---------------
    # ---------------------------------------

    return PINNReactorModelResults(
        model,
        loss_history,
        train_data,
        solver_params,
        eq_params,
        process_params,
        initial_state,
        f_out_value_calc,
    )
