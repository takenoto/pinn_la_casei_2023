from domain.optimization.non_dim_scaler import NonDimScaler
from domain.numeric_solver.euler import EulerMethod


def run_numerical_methods(
    initial_state,
    eq_params,
    process_params,
    f_out_value_calc,
    t_discretization_points=[240],
    non_dim_scaler: NonDimScaler = NonDimScaler(),
):
    initial_state = initial_state

    print("Starting Numerical Methods")

    # ---------------------------------------------------------
    # Numerical Calculation
    euler = EulerMethod()
    num_results = []
    for t_disc in t_discretization_points:
        num_result = euler.solve(
            initial_state,
            eq_params,
            process_params,
            f_out_value_calc,
            non_dim_scaler=non_dim_scaler,
            t_discretization_points=t_disc,
            name=f"euler {t_disc}p",
        )
        num_results.append(num_result)
   
    # ---------------------------------------------------------

    return num_results
