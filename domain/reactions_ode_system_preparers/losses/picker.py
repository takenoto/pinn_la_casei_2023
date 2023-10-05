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
    
    # Loss v7 + receive all args
    all_args = args
    # Other losses only the first 19
    args = args[:18]
    if loss_version == 7:
        return lossV7(o, all_args)
    if loss_version == 6:
        return lossV6(o, args)
    elif loss_version == 5:
        return lossV5(o, args)
    elif loss_version == 4:
        return lossV4(o, args)
    elif loss_version <= 3:
        return lossV3minus(o, args)
