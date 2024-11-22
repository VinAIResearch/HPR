import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers.utils import logging

from .guidance import GuidanceModule


# Logging
logger = logging.get_logger(__name__)


# Classes
class WrappedGuidedDecoderLayer(nn.Module):
    """
    Wrap a decoder layer (nn.Module object) and couple it with a guidance module. The output of the decoder layer will
    be edited using this guidance module.
    :param wrapped_module: ('nn.Module') The layer module to wrap.
    :param guidance_module: (GuidanceModule) The module to guide the decoder layer's output.
    :param alpha: (float) The scaling factor to scale guidance vector.
    :param learnable_decoder: (bool) Whether wrapped_module should require gradient or not while being wrapped.
    """

    def __init__(self,
                 wrapped_module: nn.Module,
                 guidance_module: GuidanceModule,
                 alpha: Optional[float] = None,
                 normalize_guidance: bool = True,
                 generation_mode: bool = False,
                 learnable_decoder: bool = False):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.learnable_decoder = learnable_decoder
        self.wrapped_module.requires_grad_(self.learnable_decoder)
        self.guidance_module = guidance_module
        self.alpha = alpha if alpha is not None else 1
        self.normalize_guidance = normalize_guidance
        self._generation_mode = generation_mode
        # TODO: Enable wrapping a layer with multiple distinct guidance modules(?)

    def forward(
            self,
            *args,
            **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        orig_outputs = self.wrapped_module(*args, **kwargs)
        hidden_states = orig_outputs[0]
        if self.guidance_module.is_enabled:
            if self._generation_mode:
                # In generation mode, only edit the last token
                batch_size, seq_len, hidden_size = hidden_states.shape
                last_token = hidden_states[:, -1, :].reshape(batch_size, 1, hidden_size)
                last_token = self.guidance_module.apply_guidance(last_token,
                                                                 alpha=self.alpha,
                                                                 normalize_guidance=self.normalize_guidance)
                hidden_states[:, -1, :] = last_token.reshape(batch_size, hidden_size)
            else:
                # Otherwise, edit all tokens
                hidden_states = self.guidance_module.apply_guidance(hidden_states,
                                                                    alpha=self.alpha,
                                                                    normalize_guidance=self.normalize_guidance)
        return (hidden_states,) + orig_outputs[1:]

    def generation_mode(self, switch_on: Optional[bool] = None):
        """
        Optionally set and return the state of _generation_mode.
        :param switch_on: (bool, defaults to None) Value to set.
        :return: None or Boolean
        """
        if switch_on is not None:
            self._generation_mode = switch_on
        return self._generation_mode
