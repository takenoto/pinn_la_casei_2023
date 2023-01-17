class Altiok2006Params:
    def __init__(self, mu_max, K_S, alpha, beta, Y_PS, ms, f, h, Pm, Xm):
        """
        Kinetics parameters for the model described by Altiok (2006)
        """
        self.mu_max = np.array([mu_max])
        self.K_S = np.array([K_S])
        self.alpha = np.array([alpha])
        self.beta = np.array([beta])
        self.Y_PS = np.array([Y_PS])
        self.ms = np.array([ms])
        self.f = (f,)
        self.h = (h,)
        self.Pm = Pm
        self.Xm = Xm
        pass
