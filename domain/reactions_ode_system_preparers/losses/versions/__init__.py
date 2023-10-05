# from .loss_v3minus.py import lossV3
# import .loss_v4
# import .loss_v5
# import .loss_v6
# import .loss_v7

from domain.reactions_ode_system_preparers.losses.versions.loss_v3minus import (
    lossV3minus,
)
from .loss_v4 import lossV4
from .loss_v5 import lossV5
from .loss_v6 import lossV6
from .loss_v7 import lossV7


__all__ = ["lossV3minus", "lossV4", "lossV5", "lossV6", "lossV7"]
