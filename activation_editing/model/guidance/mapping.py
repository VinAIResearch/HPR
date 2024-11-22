from .linear import *
from .householder import *
from .loss_fnc import *


GUIDANCE_MODULE_CLASSES = {
    "linear": LinearGuidanceModule,
    "householder": HouseholderGuidanceModule
}


"""
Loss functions should follow a conventional interface. Their signature should be:

def loss_fnc(guidance_module, positive_activation, negative_activation, return_outputs, **kwargs) -> torch.Tensor:
    ...
"""
GUIDANCE_LOSS_FUNCTIONS = {
    "linear": mse_loss,
    "householder": householder_loss
}


COMPUTE_METRICS_FUNCTIONS = {
    "linear": None,
    "householder": householder_evaluation_metrics
}
