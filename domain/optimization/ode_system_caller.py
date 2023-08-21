# Local imports

from domain.params.solver_params import SolverParams
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.reactor.cstr_state import CSTRState
from domain.run_reactor.plot_params import PlotParams
from domain.reactions_ode_system_preparers.ode_preparer import ODEPreparer
from domain.run_reactor.run_reactor import run_reactor
from domain.run_reactor.pinn_reactor_model_results import PINNReactorModelResults


class RunReactorSystemCaller:
    def __init__(
        self,
        eq_params: Altiok2006Params,
        process_params: ProcessParams,
        initial_state: CSTRState,
        f_out_value_calc,
    ):

        self.ode_system_preparer = None
        self.eq_params = eq_params
        self.process_params = process_params
        self.initial_state = initial_state
        self.f_out_value_calc = f_out_value_calc

    def call(
        self,
        solver_params: SolverParams,
    ) -> PINNReactorModelResults:
        """

        Tem somente  intuito de encapsular a funcionalidade da execução do modelo

        em um único bloco, para que possa ser iterado facilmente para tunar

        hiperparâmetros.

        """
        self.ode_system_preparer = ODEPreparer(
            solver_params=solver_params,
            eq_params=self.eq_params,
            process_params=self.process_params,
            initial_state=self.initial_state,
            f_out_value_calc=self.f_out_value_calc,
        )

        return run_reactor(
            ode_system_preparer=self.ode_system_preparer,
            solver_params=solver_params,
            eq_params=self.eq_params,
            process_params=self.process_params,
            initial_state=self.initial_state,
            f_out_value_calc=self.f_out_value_calc,
        )
