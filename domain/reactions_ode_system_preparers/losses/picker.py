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

_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def loss_picker(loss_version, o, args):
    assert o in ["X", "P", "S", "V"], "Key must be X,P,S or V"
    
    # Loss v7 + receive all args
    all_args = args
    # Other losses only the first 19
    args = args[:18]
    # "7A", "7B", ... até "7Z"
    if loss_version in [f"7{letter}" for letter in _alphabet]: 
        return lossV7(o, all_args, loss_version)
    if loss_version == 6:
        return lossV6(o, args)
    elif loss_version == 5:
        return lossV5(o, args)
    elif loss_version == 4:
        return lossV4(o, args)
    elif loss_version <= 3:
        return lossV3minus(o, args)
