""" Diffusion Model module for condevo package. """

from .diffusion_model import DM
from .ddim import DDIM
from .rectified_flow import RectFlow
from .v_prediction import VPred
from .x_prediction import XPred


def get_default_model(num_params, num_hidden=32, num_steps=100, dm_cls="RectFlow", num_conditions=0) -> DM:
    from ..nn import MLP
    mlp = MLP(num_params=num_params, num_hidden=num_hidden, num_conditions=num_conditions)

    if dm_cls == "RectFlow":
        return RectFlow(nn=mlp, num_steps=num_steps)

    elif dm_cls == "DDIM":
        return DDIM(nn=mlp, num_steps=num_steps)

    else:
        raise NotImplementedError(f"diffusion model 'dm_cls': {dm_cls}")



__all__ = ['DM',
           'DDIM',
           'RectFlow',
           'VPred',
           'XPred',
           ]
