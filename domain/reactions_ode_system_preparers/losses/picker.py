# ---------------------------------------------
# Declare losses types
# ---------------------------------------------
from domain.reactions_ode_system_preparers.losses.versions import (
    lossV3minus,
    lossV4,
    lossV5,
    lossV6,
    lossV7,
)


def loss_picker(loss_version, o, args):
    assert o in ["X", "P", "S", "V"], "Key must be X,P,S or V"
    # args = X, P, S, V, dXdt, dPdt, dSdt, dVdt,
    # rX, rP, rS, inlet, f_in, f_out, Xm, Pm, initial_state
    if loss_version == 7:
        return lossV7(o, args)
    if loss_version == 6:
        return lossV6(o, args)
    elif loss_version == 5:
        return lossV5(o, args)
    elif loss_version == 4:
        return lossV4(o, args)
    elif loss_version <= 3:
        return lossV3minus(o, args)
