import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PretrainedConfig, EvalPrediction
from typing import Optional, Union, List, Tuple

from .base import GuidanceModule
from .config import GuidanceConfig
from utils import get_householder_matrix, calculate_angle


# Classes
class AngleRegressionModel(nn.Module):

    def __init__(self, hidden_dim: Union[Tuple[int], List[int], int]):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        layers = []
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())  # PReLU?   No non-linear activation because we're doing regression here
        # The output must be a scalar
        layers.append(nn.Linear(in_features=hidden_dim[-1], out_features=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # The output must be positive, and must not exceed pi radians
        return np.pi * torch.sigmoid(self.model(x))


class HouseholderGuidanceModule(GuidanceModule):

    def __init__(self,
                 config: GuidanceConfig,
                 base_model_config: Optional[PretrainedConfig] = None,
                 layer_idx: Optional[int] = None):
        super().__init__(
            config=config,
            base_model_config=base_model_config,
            layer_idx=layer_idx
        )

        # Base model's hidden size
        if self.base_model_config is not None:
            input_dim = self.base_model_config.hidden_size
            self.config.base_model_hidden_size = self.base_model_config.hidden_size
        elif self.config.base_model_hidden_size is not None:
            input_dim = self.config.base_model_hidden_size
        else:
            raise AssertionError("Cannot determine base model's hidden size. Please specify base_model_hidden_size in "
                                 "the GuidanceConfig or provide the base model's config when initializing this module.")

        # Calculate hidden dimension
        if self.config.guidance_hidden_size is not None:
            # The input dimension will be the hidden_size of the base model.
            # The hidden dimension of each layer will be the given guidance_hidden_size.
            angle_model_hidden_dim = [input_dim if i == 0 else self.config.guidance_hidden_size
                                      for i in range(self.config.num_guidance_module_layers)]
        else:
            # Each layer will reduce the hidden dimension by half
            angle_model_hidden_dim = [input_dim // (2 ** i) for i in range(self.config.num_guidance_module_layers)]

        # Logistic Regression model:
        # This model has only 1 layer --> No need config.guidance_hidden_size
        # The model doesn't have bias in order to make sure that the reflection vector goes through the origin
        self.lr_model = nn.Linear(in_features=input_dim, out_features=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Norm ratio regression model:
        self.angle_model = AngleRegressionModel(hidden_dim=angle_model_hidden_dim)

    def forward(self, x):
        return self.logistic_regression_forward(x), self.angle_prediction_forward(x)

    def logistic_regression_forward(self, x):
        # Convention: 1 = positive activation, 0 = negative activation
        return self.sigmoid(self.lr_model(x))

    def angle_prediction_forward(self, x):
        return self.angle_model(x)

    def get_reflection_vector(self) -> torch.Tensor:
        return self.lr_model.weight

    def get_householder_matrix(self) -> torch.Tensor:
        return get_householder_matrix(self.get_reflection_vector())

    def apply_guidance(self,
                       activation: torch.Tensor,
                       alpha: float = 1.0,
                       normalize_guidance: bool = True,
                       **kwargs) -> torch.Tensor:
        # Step 1: Use the Logistic Regression prediction to decide whether to reflect the activation
        module_output = self.forward(activation)
        # This tells us which activation vectors should be rotated.
        sign = module_output[0].round()      # Shape: (batch_size, sequence_length, 1)
        # And this tells us how much should we rotate the original activation vectors.
        angle = module_output[1]     # Shape: (batch_size, sequence_length, 1)
        if alpha is not None:   # Usually we don't need to scale angle. So set alpha=1.0 is good enough.
            angle = angle * alpha

        # Step 2: Reflect the activation about the learned hyperplane
        householder = self.get_householder_matrix()     # Shape: (hidden_size, hidden_size)
        reflected_activation = torch.matmul(activation, householder)

        # Step 3: Calculate rotated vector.
        guided_activation = self._rotate_and_combine(original_activation=activation,
                                                     reflected_activation=reflected_activation,
                                                     sign=sign,
                                                     angle=angle)
        return guided_activation

    def _rotate_and_combine(self,
                            original_activation: torch.Tensor,
                            reflected_activation: torch.Tensor,
                            sign: torch.Tensor,
                            angle: torch.Tensor) -> torch.Tensor:
        """
        Combine activation vectors such that only negative activation vectors are rotated
        Prerequisite:
            - b: batch size
            - h: hidden dimension
            - s: sequence length
        sign will be used to form a positive mask (filter out positive activation vectors) and negative mask.
        The reflected activation will be rotated using the predicted angle, then filtered using the negative
        mask and finally combine with the filtered original activation.
        In other words, the output of this function will be the original positive activation and the rotated negative
        activation vectors.
        :param original_activation: ('torch.Tensor') The original activation. Shape: (b, s, h)
        :param reflected_activation: ('torch.Tensor') The reflected activation. Shape: (b, s, h)
        :param sign:  ('torch.Tensor') The output of Logistic Regression. Shape: (b, s, 1)
        :param angle: ('torch.Tensor') The predicted angle between the new vector and the orig vector. Shape: (b, s, 1)
        :return: ('torch.Tensor') The activation matrix, whose negative vectors are rotated.
        """
        batch_size = original_activation.shape[0]
        hidden_size = original_activation.shape[-1]
        seq_len = original_activation.shape[-2]

        # Rotation readjustment
        gamma1 = angle
        gamma2 = calculate_angle(original_activation, reflected_activation)
        beta1 = torch.sin(gamma1) / torch.sin(gamma2)
        beta2 = torch.sin(gamma2 - gamma1) / torch.sin(gamma2)
        rotated_activation = beta1 * reflected_activation + beta2 * original_activation

        # Create filters
        sign = sign.squeeze(-1).reshape(seq_len, batch_size)     # Squeeze into 2D vector -> Shape: (s, b)
        negative_sign = 1 - sign
        sign = torch.diag_embed(sign)     # A filter that keeps positive activation vectors. Shape: (s, b, b)
        negative_sign = torch.diag_embed(negative_sign)   # A filter that keeps reflected negative activation vectors.

        # Combine
        original_activation = original_activation.reshape((seq_len, batch_size, hidden_size))   # Shape: (s, b, h)
        rotated_activation = rotated_activation.reshape((seq_len, batch_size, hidden_size))   # Shape: (s, b, h)
        final_activation = torch.matmul(sign, original_activation) + torch.matmul(negative_sign, rotated_activation)
        final_activation = final_activation.reshape((batch_size, seq_len, hidden_size))   # Shape: (b, s, h)

        return final_activation


# Functions
def householder_evaluation_metrics(eval_pred: EvalPrediction):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    assert len(preds) == len(labels), "Householder eval metrics: preds and labels have non-matching number of layers."

    lr_count = 0
    lr_correct = 0
    all_angle_pred = []
    all_angle_label = []
    for pred, label in zip(preds, labels):
        lr_pred, angle_pred = pred
        lr_label, angle_label = label

        lr_results = np.array(lr_pred.round() == lr_label)
        lr_correct += lr_results.sum()
        lr_count += lr_results.size

        all_angle_pred.append(angle_pred)
        all_angle_label.append(angle_label)

    lr_acc = lr_correct / lr_count
    all_angle_pred = np.concatenate(all_angle_pred)
    all_angle_label = np.concatenate(all_angle_label)
    angle_loss = F.mse_loss(torch.tensor(all_angle_pred), torch.tensor(all_angle_label))
    return {"eval_lr_acc": lr_acc,
            "eval_angle_loss": angle_loss}
