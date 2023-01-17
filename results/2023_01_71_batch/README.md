Results obtained for the simulation with input:

`
eq_params = altiok_models_to_run[0]

initial_state = CSTRState(
            volume=4,
            X=eq_params.Xo,
            P=eq_params.Po,
            S=eq_params.So,
        )

process_params = ProcessParams(
    max_reactor_volume=5,
    inlet=ConcentrationFlow(
        volume=0,
        X=eq_params.Xo,
        P=eq_params.Po,
        S=eq_params.So,
    ),
    t_final=16,
)

batch_results = run_model(
    solver_params_list=None,
    eq_params=eq_params,
    process_params=process_params,
    # A saída sempre é 0 (reator fechado)
    f_out_value_calc=lambda max_reactor_volume, f_in_v, volume: 0,
)
`


@ 2023-01-17 | 16:14