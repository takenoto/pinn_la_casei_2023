from main.plot_xpsv import plot_xpsv
from domain.optimization.non_dim_scaler import NonDimScaler
from domain.numeric_solver.euler import EulerMethod
from domain.numeric_solver.numeric_solver_model_results import NumericSolverModelResults


def run_numerical_methods(initial_state, eq_params, process_params, f_out_value_calc):
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

    print("Starting Numerical Methods")
    # Quando não é 1 dá tudo errado, várias linhas
    # Ou seja, tá errada as multiplicações lá
    scaler = NonDimScaler(t=16, V=26, S=15, X=15, P=20)

    # ---------------------------------------------------------
    # Numerical Calculation
    euler = EulerMethod()
    num_result1 = euler.solve(
        initial_state,
        eq_params,
        process_params,
        f_out_value_calc,
        non_dim_scaler=scaler,
        t_discretization_points=240,#60,
        #name="euler 60p",
        name='euler 240p'
    )

    num_result2 = euler.solve(
        initial_state,
        eq_params,
        process_params,
        f_out_value_calc,
        non_dim_scaler=scaler,
        t_discretization_points=4,
        name="euler 4p",
    )
    # ---------------------------------------------------------

    return [
        num_result1,
        # num_result2
    ]
