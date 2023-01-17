class PlotParams:
    def __init__(
        self, show_concentrations=True, show_volume_with_flows=True, force_y_lim=False
    ):
        self.show_concentrations = show_concentrations
        self.show_volume_with_flows = show_volume_with_flows
        self.force_y_lim = force_y_lim
        """
        Essa informação diz se deve forçar o limite de y entre
        os ranges pré-determinados ao plotar os gráficos.

        Por exemplo, se deixar False vai plotar normalmente a critério da matplotlib
        Se deixar True vai plotar de 0 até So
        E no caso do volume de Vo até Vmax*1.2 ou algo parecido
        """
