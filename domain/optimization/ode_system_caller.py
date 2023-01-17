# Local imports

from domain.params.solver_params import SolverParams
from domain.params.altiok_2006_params import Altiok2006Params
from domain.params.process_params import ProcessParams
from domain.reactor.cstr_state import CSTRState
from domain.run_reactor.plot_params import PlotParams

from domain.run_reactor.run_reactor import run_reactor


class RunReactorSystemCaller:
    def __init__(
        self,
        ode_system_preparer,
        eq_params: Altiok2006Params,
        process_params: ProcessParams,
        initial_state: CSTRState,
        plot_params: PlotParams,
        f_out_value_calc,
    ):

        self.ode_system_preparer = ode_system_preparer
        self.eq_params = eq_params
        self.process_params = process_params
        self.initial_state = initial_state
        self.plot_params = plot_params
        self.f_out_value_calc = f_out_value_calc

    def call(
        self,
        solver_params: SolverParams,
    ):
        """

        Tem somente  intuito de encapsular a funcionalidade da execução do modelo

        em um único bloco, para que possa ser iterado facilmente para tunar

        hiperparâmetros.

        """

        return run_reactor(
            ode_system_preparer=self.ode_system_preparer,
            solver_params=solver_params,
            eq_params=self.eq_params,
            process_params=self.process_params,
            initial_state=self.initial_state,
            plot_params=self.plot_params,
            f_out_value_calc=self.f_out_value_calc,
        )
