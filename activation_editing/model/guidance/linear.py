import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing import Optional

from .base import GuidanceModule
from .config import GuidanceConfig


# Classes
class LinearGuidanceModule(GuidanceModule):

    def __init__(self,
                 config: GuidanceConfig,
                 base_model_config: Optional[PretrainedConfig] = None,
                 layer_idx: Optional[int] = None):
        super().__init__(
            config=config,
            base_model_config=base_model_config,
            layer_idx=layer_idx
        )

        if self.base_model_config is not None:
            input_dim = self.base_model_config.hidden_size
            output_dim = self.base_model_config.hidden_size
            self.config.base_model_hidden_size = self.base_model_config.hidden_size
        elif self.config.base_model_hidden_size is not None:
            input_dim = self.config.base_model_hidden_size
            output_dim = self.config.base_model_hidden_size
        else:
            raise AssertionError("Cannot determine base model's hidden size. Please specify base_model_hidden_size in "
                                 "the GuidanceConfig or provide the base model's config when initializing this module.")

        if self.config.guidance_hidden_size is not None:
            hidden_dim = self.config.guidance_hidden_size
        elif self.base_model_config is not None:
            hidden_dim = self.base_model_config.hidden_size
            self.config.guidance_hidden_size = self.base_model_config.hidden_size
        else:
            raise AssertionError("Cannot determine the guidance module's hidden size. Please specify "
                                 "guidance_hidden_size in the GuidanceConfig or provide the base model's config when "
                                 "initializing this module.")

        layers = []
        for i in range(self.config.num_guidance_module_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())    # PReLU?
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())        # PReLU?
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
