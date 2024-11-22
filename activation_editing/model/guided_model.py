import os
from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.trainer_pt_utils import find_batch_size
from transformers.utils import logging
import torch
from typing import Union, List, Optional, Callable, Dict, Any

from .wrapper_modules import WrappedGuidedDecoderLayer
from .guidance import GuidanceModel, GuidanceConfig, IDS_LIST_PATTERN


# Logging
logger = logging.get_logger(__name__)


# Functions
def _is_wrapped(module):
    return hasattr(module, "wrapped_module")


# Classes
class GuidedModelForCausalLM(PreTrainedModel):

    def __init__(self,
                 config: PretrainedConfig,
                 guidance_config: Optional[GuidanceConfig] = None):
        super().__init__(config)

        # Create guidance module:
        self.guidance_config = guidance_config
        if self.guidance_config is not None:
            self.initialize_guidance_modules()

    def initialize_guidance_modules(self):
        setattr(self, "guidance_modules", GuidanceModel(config=self.guidance_config, base_model_config=self.config))
        self.guidance_modules.to(self.device)
        return

    def wrap_decoder(self,
                     layer_idx: int,
                     alpha: Optional[float] = None,
                     generation_mode: bool = False,
                     learnable_decoder: bool = False):
        """
        Wrap a decoder layer in a WrappedGuidedDecoderLayer objects.
        :return: None
        """
        if str(layer_idx) in self.guidance_modules.keys():
            if not self.is_wrapped(layer_idx):
                self.model.layers[layer_idx] = WrappedGuidedDecoderLayer(self.model.layers[layer_idx],
                                                                         self.guidance_modules[str(layer_idx)],
                                                                         alpha=alpha,
                                                                         generation_mode=generation_mode,
                                                                         learnable_decoder=learnable_decoder
                                                                         )
            else:
                self.model.layers[layer_idx].generation_mode(generation_mode)
        else:
            logger.warning(f"layer_idx {layer_idx} does not currently have a guidance module.")
        return

    def wrap_all_decoders(self,
                          alpha: Optional[float] = None,
                          generation_mode: bool = False,
                          learnable_decoders: bool = False):
        for layer_idx in self.guidance_modules.keys():
            self.wrap_decoder(layer_idx=int(layer_idx),
                              alpha=alpha,
                              generation_mode=generation_mode,
                              learnable_decoder=learnable_decoders)
        return

    def unwrap_decoder(self, layer_idx: int, learnable_decoder: bool = True):
        """
        Unwrap a WrappedGuidedDecoderLayer objects back to the original decoder layer.
        :return: None
        """
        if str(layer_idx) in self.guidance_modules.keys():
            if self.is_wrapped(layer_idx):
                self.model.layers[layer_idx] = self.model.layers[layer_idx].wrapped_module
                self.model.layers[layer_idx].requires_grad_(learnable_decoder)
        else:
            logger.warning(f"layer_idx {layer_idx} does not currently have a guidance module.")
        return

    def unwrap_all_decoders(self, learnable_decoders: bool = True):
        for layer_idx in self.guidance_modules.keys():
            self.unwrap_decoder(layer_idx=int(layer_idx), learnable_decoder=learnable_decoders)
        return

    def is_wrapped(self, layer_idx):
        return hasattr(self.model.layers[layer_idx], "wrapped_module")

    def is_wrapped_all(self):
        return all([self.is_wrapped(int(layer_idx)) for layer_idx in self.guidance_modules.keys()])

    def generation_mode(self, switch_on: bool):
        for layer_idx in self.guidance_modules.keys():
            if self.is_wrapped(int(layer_idx)):
                self.model.layers[int(layer_idx)].generation_mode(switch_on)
        return

    def prepare_modules_for_training(self):
        # Enable guidance modules (optional)
        self.guidance_modules.enable_all()
        # Unwrap decoders
        self.unwrap_all_decoders(learnable_decoders=False)
        # Turn off gradient in all modules except guidance modules
        self.model.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        self.guidance_modules.requires_grad_(True)
        return

    def prepare_modules_for_inference(self, alpha: Optional[float] = None, generation_mode: bool = False):
        # Enable guidance modules (mandatory)
        if self.guidance_config.selected_layers is not None:
            self.guidance_modules.disable_all()
            selected_layers = self.guidance_config.selected_layers
            if isinstance(selected_layers, int):
                self.guidance_modules.enable(selected_layers)
            elif isinstance(self.guidance_config.selected_layers, str):
                assert IDS_LIST_PATTERN.fullmatch(selected_layers), f"Invalid target layers: {selected_layers}"
                self.guidance_modules.enable(*selected_layers.split(','))
            elif isinstance(selected_layers, list) or isinstance(selected_layers, tuple):
                self.guidance_modules.enable(*selected_layers)
            elif isinstance(selected_layers, dict):
                self.guidance_modules.enable(*list(selected_layers.keys()))
            else:
                raise ValueError(f"Cannot parse selected_layers: {selected_layers}")
        else:
            self.guidance_modules.enable_all()
        # Wrap decoders
        self.wrap_all_decoders(alpha=alpha, generation_mode=generation_mode, learnable_decoders=False)
        # Turn off all gradients
        self.requires_grad_(False)
        return

    def save_guidance_modules(self, save_directory: Union[str, os.PathLike]):
        if os.path.isfile(save_directory):
            raise NotADirectoryError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        self.guidance_config.save_pretrained(save_directory=save_directory)
        self.guidance_modules.save_pretrained(save_directory=save_directory)
        return

    def load_guidance_modules(self, directory: Union[str, os.PathLike]):
        if os.path.isfile(directory):
            raise NotADirectoryError(f"Provided path ({directory}) should be a directory, not a file")
        logger.info(f"Loading guidance modules from {directory}.")
        guidance_config = GuidanceConfig.from_pretrained(directory)
        setattr(self, "guidance_config", guidance_config)
        self.initialize_guidance_modules()
        self.guidance_modules.load_pretrained(directory=directory)
        return

    def floating_point_ops(
            self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get the number of floating-point operations for the forward and backward passes of a batch.
        This is only valid WHEN TRAINING THE GUIDANCE MODULES.
        Since the guidance modules are all linear models, it takes 1 flop for every parameter for every input sample to
        each module for the forward pass and the same amount of flops for the backward pass. The number of flops is thus
        2 * batch_size * num_parameters (guidance modules only).
        :param input_dict: An input batch.
        :param exclude_embeddings: (bool) This is here just to be consistent with the method's signature from parent.
        :return: (int) number of FLOPs for a forward and backward pass to train the guidance modules.
        """
        batch_size = find_batch_size(input_dict)   # Grab the number of rows of the first tensor.
        trainable_params = 0        # forward and backward
        non_trainable_params = 0    # only forward
        assert hasattr(self, "guidance_modules"), "Guidance modules not initialized. Cannot estimate FLOPs."
        for param in self.guidance_modules.parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()
        return (batch_size*non_trainable_params) + (2*batch_size*trainable_params)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        # New kwargs here
        guidance_modules_path: Optional[Union[str, os.PathLike]] = None,
        guidance_config: Optional[GuidanceConfig] = None,
        # End of new kwargs
        **kwargs,
    ):
        # Pop out guidance model's kwargs
        guidance_kwargs = {}
        for key in kwargs:
            if key in list(GuidanceConfig.__annotations__):
                guidance_kwargs[key] = kwargs.pop(key, None)

        # Initialize model
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs
        )

        # Guidance config
        if guidance_modules_path is not None:
            model.load_guidance_modules(directory=guidance_modules_path)
        else:
            if guidance_config is None:
                guidance_config = GuidanceConfig(
                    base_model_name_or_path=pretrained_model_name_or_path,
                    **guidance_kwargs
                )
            logger.warning("guidance_model_path not provided. Guidance modules will be initialized randomly.")
            setattr(model, "guidance_config", guidance_config)
            model.initialize_guidance_modules()
        return model

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        # New kwargs here
        do_wrapping: bool = False,
        do_unwrapping: bool = False,
        alpha: Optional[float] = None,
        # End of new kwargs
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if do_wrapping:
            self.wrap_all_decoders(alpha=alpha, generation_mode=True)
        self.generation_mode(True)

        generation_outputs = super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs
        )

        self.generation_mode(False)
        if do_unwrapping:
            self.unwrap_all_decoders()

        return generation_outputs


class GuidedLlamaForCausalLM(GuidedModelForCausalLM, LlamaForCausalLM):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)


class GuidedMistralForCausalLM(GuidedModelForCausalLM, MistralForCausalLM):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)


# CONSTANTS
GUIDED_MODEL_CLASSES = {
    "llama": GuidedLlamaForCausalLM,
    "mistral": GuidedMistralForCausalLM
}


class AutoGuidedModelForCausalLM:
    """
    A minimally simple auto class to initialize guided models.
    """

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        assert config.model_type in GUIDED_MODEL_CLASSES, (f"Can't find guided model class for "
                                                           f"model type '{config.model_type}'.")
        model_class = GUIDED_MODEL_CLASSES[config.model_type]
        return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
