"""
Loss functions should follow a conventional interface. Their signature MUST be:

def loss_fnc(guidance_module, predicted_direction, true_direction, **kwargs) -> torch.Tensor:
    ...
"""
import torch
import torch.nn.functional as F

from .base import GuidanceModule
from .householder import HouseholderGuidanceModule
from utils import calculate_angle


# Functions
def mse_loss(guidance_module: GuidanceModule,
             positive_activation: torch.Tensor,
             negative_activation: torch.Tensor,
             return_outputs: bool = False,
             **kwargs):
    lambda_p = kwargs.pop('lambda_p', None)

    negative_outputs = guidance_module(negative_activation)
    expected_outputs = positive_activation - negative_activation
    loss = F.mse_loss(negative_outputs, expected_outputs)

    if lambda_p is not None:
        positive_outputs = guidance_module(positive_activation)
        positive_regularization = torch.linalg.norm(positive_outputs, ord=2, dim=-1).mean()
        loss += loss + (lambda_p * positive_regularization)

    if return_outputs:
        return loss, negative_outputs
    return loss


def householder_loss(guidance_module: GuidanceModule,
                     positive_activation: torch.Tensor,
                     negative_activation: torch.Tensor,
                     return_outputs: bool = False,
                     **kwargs):
    assert isinstance(guidance_module, HouseholderGuidanceModule), (f"Incompatible loss function (householder_loss) "
                                                                    f"and guidance module {type(guidance_module)}).")

    # Prepare inputs and labels
    stacked_activation = torch.cat([positive_activation, negative_activation], dim=0)
    stacked_positive = torch.cat([positive_activation, positive_activation], dim=0)
    positive_label = torch.ones(*positive_activation.shape[:-1], 1,
                                dtype=positive_activation.dtype,
                                device=positive_activation.device)
    negative_label = torch.zeros(*negative_activation.shape[:-1], 1,
                                 dtype=negative_activation.dtype,
                                 device=negative_activation.device)
    lr_label = torch.cat([positive_label, negative_label], dim=0)
    perm = torch.randperm(lr_label.shape[0])   # Shuffle the datasets before passing through the Logistic Regression
    stacked_activation = stacked_activation[perm]
    stacked_positive = stacked_positive[perm]
    lr_label = lr_label[perm]
    angle_label = calculate_angle(stacked_activation, stacked_positive)

    # Logistic Regression loss (direction of the guidance)
    # Learn a hyperplane that can effectively separate the two contrasting regions
    lr_pred, angle_pred = guidance_module(stacked_activation)
    lr_loss = F.binary_cross_entropy(input=lr_pred, target=lr_label)

    # Angle prediction loss
    # Predict the angle between the expected positive activation and the given activation
    angle_loss = F.mse_loss(input=angle_pred, target=angle_label)     # TODO: Experiment with other regression losses

    # shift_direction = positive_activation - negative_activation
    # re_vector = guidance_module.get_reflection_vector()
    # re_vector = re_vector.reshape(*[1 for _ in shift_direction.shape[:-1]], re_vector.shape[-1])
    # re_vector = re_vector.repeat(*shift_direction.shape[:-1], 1)
    # align_loss = F.mse_loss(input=re_vector, target=shift_direction)    # TODO: Experiment with other losses

    if return_outputs:
        return lr_loss + angle_loss, (lr_pred, angle_pred, lr_label, angle_label)
    return lr_loss + angle_loss
