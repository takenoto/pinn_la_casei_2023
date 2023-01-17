from domain.flow.concentration_flow import ConcentrationFlow

class ProcessParams:
    def __init__(self, max_reactor_volume, inlet: ConcentrationFlow, t_final):
        self.max_reactor_volume = max_reactor_volume
        self.inlet = inlet
        self.t_final = t_final
