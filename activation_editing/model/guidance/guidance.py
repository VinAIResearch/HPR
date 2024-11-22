import torch.nn as nn
from transformers import PretrainedConfig
from transformers.utils import logging
from safetensors.torch import save_model, load_model
from typing import Optional, Mapping, Union, Dict
import os
import re

from .config import GuidanceConfig
from .mapping import GUIDANCE_MODULE_CLASSES


# Logging
logger = logging.get_logger(__name__)


# CONSTANTS
GUIDANCE_MODEL_NAME = "guidance.safetensors"
IDS_LIST_PATTERN = re.compile("([0-9]+,)*[0-9]+")


# Classes
class GuidanceModel(nn.ModuleDict):
    def __init__(self,
                 config: GuidanceConfig,
                 base_model_config: Optional[PretrainedConfig] = None,
                 modules: Optional[Mapping[str, nn.Module]] = None):
        super().__init__(modules=modules)
        self.config = config
        self.base_model_config = base_model_config
        if self.base_model_config is not None:
            # Prioritize hyperparams from the base model's config over the hyperparams given by the GuidanceConfig.
            self.config.base_model_name_or_path = self.base_model_config._name_or_path
            self.config.base_model_num_hidden_layers = self.base_model_config.num_hidden_layers
            self.config.base_model_hidden_size = self.base_model_config.hidden_size
            # if self.config.guidance_hidden_size is None:
            #     self.config.guidance_hidden_size = self.base_model_config.hidden_size
        self.initialize_guidance_modules()

    def initialize_guidance_modules(self, overwrite_existing: bool = False) -> None:
        """
        Initialize all guidance modules as specified in the config.
        :param overwrite_existing: (bool) If True, overwrite the existing guidance modules and initialize new ones.
        :return: None
        """
        # Check if overwriting
        if len(self) > 0 and hasattr(self.config, "target_layers") and self.config.target_layers:
            if not overwrite_existing:
                logger.warning(f"The guidance modules for this LlamaWithLinearGuidance object exist. This function "
                               f"thus has no effect. If you want to proceed and overwrite the existing guidance "
                               f"modules, pass overwrite_existing=True as an argument when calling this method.")
                return
            else:
                logger.warning(f"The guidance modules for this LlamaWithLinearGuidance object exist. However, since "
                               f"overwrite_existing=True is passed, new guidance modules will be initialized and "
                               f"replace the existing ones.")

        # Determine target layers
        target_layers = self.config.target_layers
        if self.base_model_config:
            num_hidden_layers = self.base_model_config.num_hidden_layers
            self.config.base_model_num_hidden_layers = self.base_model_config.num_hidden_layers
        elif self.config.base_model_num_hidden_layers is not None:
            num_hidden_layers = self.config.base_model_num_hidden_layers
        else:
            raise AssertionError("Cannot determine the base model's number of layers. Please specify "
                                 "base_model_num_hidden_layers in the GuidanceConfig or provide the base model's "
                                 "config when initializing this model.")
        if isinstance(target_layers, str):
            if target_layers == 'all':
                target_layers = list(range(num_hidden_layers))
            elif target_layers == 'none':
                target_layers = []
            elif IDS_LIST_PATTERN.fullmatch(target_layers):
                target_layers = target_layers.split(',')
                target_layers = list(map(int, target_layers))
                assert max(target_layers) < num_hidden_layers, (f"Highest target layer index exceeds the number"
                                                                f" of layers: "
                                                                f"{max(target_layers)} >= "
                                                                f"{num_hidden_layers}.")
                target_layers = sorted(list(set(target_layers)))
            else:
                raise ValueError(f"Invalid value provided as target_layers: {target_layers}. "
                                 f"Acceptable values: List[int], Tuple[int], int, 'all', 'none', or None.")
        elif isinstance(target_layers, int):
            if target_layers < num_hidden_layers:
                target_layers = [target_layers]
            else:
                raise ValueError(f"Target layer index exceeds the number of layers: "
                                 f"{target_layers} >= {num_hidden_layers}.")
        elif target_layers is None:
            target_layers = []
        elif isinstance(target_layers, list) or isinstance(target_layers, tuple):
            assert max(target_layers) < num_hidden_layers, (f"Highest target layer index exceeds the number"
                                                            f" of layers: "
                                                            f"{max(target_layers)} >= "
                                                            f"{num_hidden_layers}.")
            target_layers = sorted(list(set(target_layers)))
        else:
            raise ValueError(f"Cannot parse target_layers: {target_layers}")
        self.config.target_layers = target_layers

        # Initialize guidance modules
        for layer_idx in self.config.target_layers:
            self[str(layer_idx)] = GUIDANCE_MODULE_CLASSES[self.config.guidance_module_type](
                config=self.config,
                base_model_config=self.base_model_config,
                layer_idx=layer_idx
            )
        return

    def add_guidance_module(self, layer_idx: int, overwrite_existing: bool = False) -> None:
        """
        Add a new guidance module for the specified layer idx.
        :param layer_idx: (int) The index of the layer to add guidance module.
        :param overwrite_existing: (bool) If True, overwrite the existing guidance module if the layer specified with
                                   layer_idx already has one.
        :return: None
        """
        if str(layer_idx) in self.keys():
            if not overwrite_existing:
                logger.warning(f"Layer index {layer_idx} already has its corresponding guidance module. "
                               f"In order to overwrite the the existing guidance module with a new one, please pass "
                               f"overwrite_existing=True.")
                return
            else:
                logger.warning(f"Layer index {layer_idx} already has its corresponding guidance module. "
                               f"Since overwrite_existing=True, a new guidance module will be initialized and replace "
                               f"the existing one.")
                self[str(layer_idx)] = GUIDANCE_MODULE_CLASSES[self.config.guidance_module_type](
                    config=self.config,
                    base_model_config=self.base_model_config,
                    layer_idx=layer_idx
                )
        else:
            self.config.target_layers.append(layer_idx)
            self.config.target_layers.sort()
            self[str(layer_idx)] = GUIDANCE_MODULE_CLASSES[self.config.guidance_module_type](
                config=self.config,
                base_model_config=self.base_model_config,
                layer_idx=layer_idx
            )
        return

    def disable(self, *layer_idx: int) -> None:
        """
        Disable the guidance modules with the corresponding layer_idx.
        :param layer_idx: (*int) The indices of the layers with guidance modules to disable.
        :return: None
        """
        for idx in layer_idx:
            if str(idx) in self.keys():
                self[str(idx)].disable()
            else:
                logger.warning(f"layer_idx {idx} does not currently have a guidance module.")
        return

    def disable_all(self) -> None:
        """
        Disable all available guidance modules.
        :return: None
        """
        for layer_idx in self.keys():
            self.disable(int(layer_idx))
        return

    def enable(self, *layer_idx: int) -> None:
        """
        Enable the guidance modules with the corresponding layer_idx.
        :param layer_idx: (*int) The indices of the layers with guidance modules to enable.
        :return: None
        """
        for idx in layer_idx:
            if str(idx) in self.keys():
                self[str(idx)].enable()
            else:
                logger.warning(f"layer_idx {idx} does not currently have a guidance module.")
        return

    def enable_all(self) -> None:
        """
        Enable all available guidance modules.
        :return: None
        """
        for layer_idx in self.keys():
            self.enable(int(layer_idx))
        return

    def is_enabled(self, all_modules: bool = True) -> Union[bool, Dict[str, bool]]:
        """
        Check if the guidance modules are enabled.
        :param all_modules: (bool) If True, check if all guidance modules are enabled. Otherwise, return the state of
                            each module.
        :return: bool or Dict[layer_idx, bool]
        """
        is_enabled = {layer_idx: self[layer_idx].is_enabled for layer_idx in self.keys()}
        if all_modules:
            return all([v for v in is_enabled.values()])
        return is_enabled

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        if os.path.isfile(save_directory):
            raise NotADirectoryError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        save_model(self, os.path.join(save_directory, GUIDANCE_MODEL_NAME), metadata={"format": "pt"})
        return

    def load_pretrained(self,
                        directory: Union[str, os.PathLike],
                        config: Optional[GuidanceConfig] = None,
                        base_model_config: Optional[PretrainedConfig] = None):
        if os.path.isfile(directory):
            raise NotADirectoryError(f"Provided path ({directory}) should be a directory, not a file")
        if config is None:
            config = GuidanceConfig.from_pretrained(directory)
        setattr(self, "config", config)
        if base_model_config is not None:
            setattr(self, "base_model_config", base_model_config)
        load_model(self, os.path.join(directory, GUIDANCE_MODEL_NAME))
        return
