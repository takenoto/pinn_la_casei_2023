class PlotPINN3DArg:
    """
    Classe feita com o único intuito de poder testar essas funções.
    """

    def __init__(self, adam_epochs, best_loss_test, t_not_tensor):
        self.adam_epochs = adam_epochs
        self.best_loss_test = best_loss_test
        self.t_not_tensor = t_not_tensor