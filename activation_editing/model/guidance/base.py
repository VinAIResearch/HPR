import torch
from torch import nn as nn
from transformers import PretrainedConfig
from typing import Optional

from .config import GuidanceConfig


class GuidanceModule(nn.Module):
    def __init__(self,
                 config: GuidanceConfig,
                 base_model_config: Optional[PretrainedConfig] = None,
                 layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.base_model_config = base_model_config
        self.layer_idx = layer_idx
        self._enabled = True
        # self.model = None   # Subclasses must assign a nn.Module object to this attribute.

    def forward(self, x):
        raise NotImplementedError("Subclasses of GuidanceModule must override forward() method.")

    def disable(self):
        self._enabled = False
        return

    def enable(self):
        self._enabled = True

    @property
    def is_enabled(self):
        return self._enabled

    def apply_guidance(self,
                       activation: torch.Tensor,
                       alpha: float = 1.0,
                       normalize_guidance: bool = True,
                       **kwargs) -> torch.Tensor:
        """
        Steer the original activation by applying a guidance strategy. Default to additive guidance, but subclasses can
        overwrite this method to a desired strategy.
        :param activation: ('torch.Tensor') The original activation output by a LLM layer.
        :param alpha: (float) Control the strength of guidance. Other strategies may not need this argument.
        :param normalize_guidance: (bool) Whether to normalize guidance vector. Other strategies may not need this.
        :return: ('torch.Tensor') A guided activation.
        """
        guidance = self.forward(activation)
        if normalize_guidance:
            guidance = guidance / guidance.norm(dim=-1, keepdim=True)
        return activation + alpha * guidance
