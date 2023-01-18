import numpy as np


class Altiok2006Params:
    """
    Kinetics parameters for the model described by Altiok (2006)
    """

    def __init__(
        self, xp_id, Xo, Po, So, mu_max, K_S, alpha, beta, Y_PS, ms, f, h, Pm, Xm
    ):
        self.xp_id = xp_id
        """
        The id of the given experiment. There were 6 models,
        so "id" goes from 1 to 6
        """
        self.Xo = np.array([Xo])
        self.Po = np.array([Po])
        self.So = np.array([So])
        self.mu_max = np.array([mu_max])
        self.K_S = np.array([K_S])
        self.alpha = np.array([alpha])
        self.beta = np.array([beta])
        self.Y_PS = np.array([Y_PS])
        self.ms = np.array([ms])
        self.f = np.array([f])
        self.h = np.array([h])
        self.Pm = np.array([Pm])
        self.Xm = np.array([Xm])
        pass

    def copy_with(
        self,
        xp_id=None,
        Xo=None,
        Po=None,
        So=None,
        mu_max=None,
        K_S=None,
        alpha=None,
        beta=None,
        Y_PS=None,
        ms=None,
        f=None,
        h=None,
        Pm=None,
        Xm=None,
    ):
        return Altiok2006Params(
            xp_id=xp_id if xp_id else self.xp_id,
            Xo=Xo if Xo else self.Xo[0],
            Po=Po if Po else self.Po[0],
            So=So if So else self.So[0],
            mu_max=mu_max if mu_max else self.mu_max[0],
            K_S=K_S if K_S else self.K_S[0],
            alpha=alpha if alpha else self.alpha[0],
            beta=beta if beta else self.beta[0],
            Y_PS=Y_PS if Y_PS else self.Y_PS[0],
            ms=ms if ms else self.ms[0],
            f=f if f else self.f[0],
            h=h if h else self.h[0],
            Pm=Pm if Pm else self.Pm[0],
            Xm=Xm if Xm else self.Xm[0],
        )


def get_altiok2006_params():
    """
    Returns the 6 params of Altiok 2006
    """

    fig1 = Altiok2006Params(
        xp_id=1,
        Xo=1.2,
        Po=4.07,
        So=9,
        mu_max=0.265,
        K_S=0.72,
        alpha=3.0,
        beta=0.06,
        Y_PS=0.682,
        ms=0.03,
        f=0.1,
        h=0.3,
        Pm=90,
        Xm=8,
    )

    return {
        1: fig1,
        2: fig1.copy_with(xp_id=2, Xo=1.15, Po=6, So=21.4, alpha=3.3, f=0.5, h=0.5),
        3: fig1.copy_with(xp_id=3, So=35.5, alpha=3.7, f=0.5, h=0.5),
        4: fig1.copy_with(xp_id=4, Xo=0.9, Po=5.86, So=48.1, alpha=4.0, f=0.5, h=0.5),
        5: fig1.copy_with(xp_id=5, Xo=0.92, Po=5.85, So=61.2, alpha=4.4, f=0.5, h=0.5),
        6: fig1.copy_with(xp_id=6, Xo=1.05, Po=7.73, So=77.1, alpha=5.0, f=0.7, h=0.5),
    }


def test():
    """
    Testa se a classe está funcionando adequadamente
    """
    fig2 = Altiok2006Params(
        xp_id=2,
        So=21.4,
        mu_max=0.265,
        K_S=0.72,
        alpha=3.3,
        beta=0.06,
        Y_PS=0.682,
        ms=0.03,
        f=0.5,
        h=0.5,
        Pm=90,
        Xm=8,
    )

    fig_dif = fig2.copy_with(xp_id=99, K_S=333, So=119, beta=-5)

    assert np.isclose(fig_dif.xp_id, 99)
    assert np.isclose(fig_dif.K_S, 333)
    assert np.isclose(fig_dif.So, 119)
    assert np.isclose(fig_dif.beta, -5)


if __name__ == "__main__":
    test()
