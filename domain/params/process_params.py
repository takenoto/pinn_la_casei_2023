import math
from domain.flow.concentration_flow import ConcentrationFlow

class ProcessParams:
    def __init__(self, max_reactor_volume, inlet: ConcentrationFlow, t_final, Smax):
        self.max_reactor_volume = max_reactor_volume
        self.inlet = inlet
        self.t_final = t_final
        self.Smax = Smax
        
    def to_dict(self):
        return {
            "max_reactor_volume":self.max_reactor_volume,
            "inlet": self.inlet.to_dict(),
            "t_final": self.t_final,
            "Smax": "inf" if math.isinf(self.Smax) else  self.Smax,
        }
