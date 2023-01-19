from main.plot_xpsv import plot_xpsv
from domain.optimization.non_dim_scaler import NonDimScaler
from domain.numeric_solver.euler import EulerMethod
from domain.numeric_solver.numeric_solver_model_results import NumericSolverModelResults



def run_numerical_methods(
    initial_state, eq_params, process_params, f_out_value_calc
):
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
    scaler=NonDimScaler(
        t=16,
        V=26,
        S=15,
        X=15,
        P=20
    )

    # ---------------------------------------------------------
    # Numerical Calculation
    euler = EulerMethod()
    num_result = euler.solve(
        initial_state,
        eq_params,
        process_params,
        f_out_value_calc,
        scaler=scaler,
        t_discretization_points=250,
    )
    # ---------------------------------------------------------
    # Tudo adimensionalizado:
    plot_xpsv(
        num_result.t, num_result.X, num_result.P, num_result.S, num_result.V, scaler=None
    )

    plot_xpsv(
        num_result.t, num_result.X, num_result.P, num_result.S, num_result.V, scaler=scaler
    )
    # TODO cria método plot XPS results e nele vc passa
    # os XP e S
    # Aí plota tudo aqui, nessa func externa

    return [NumericSolverModelResults(model=euler, model_name="euler 1")]
