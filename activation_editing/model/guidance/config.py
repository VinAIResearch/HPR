from transformers.utils import logging, PushToHubMixin
from huggingface_hub import hf_hub_download
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union
import inspect
import json
import os


# Logging
logger = logging.get_logger(__name__)


# CONSTANTS
CONFIG_NAME = "guidance_config.json"


# Classes
@dataclass
class GuidanceConfig(PushToHubMixin):

    base_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the base model to use."}
    )
    target_layers: Optional[Union[int, str]] = field(
        default='all',
        metadata={"help": "The layers to which guidance modules will be incorporated.\n"
                          "Possible values:\n"
                          "    - 'all': targets all layers.\n"
                          "    - 'none': targets no layer.\n"
                          "    - A string of integer, separated by commas (','). e.g. '1,2,3,4,8,9'. Targets all layers"
                          " whose indices listed in the string."
                          "    - An integer: targets only the layer whose index specified by the provided int.\n"
                          "    - None: targets no layer."}
    )
    selected_layers: Optional[Union[int, str]] = field(
        default=None,
        metadata={"help": "The layers selected to enable guidance modules."
                          "Useful to filter out only the guidance modules whose probe accuracy is higher than a "
                          "threshold.\n"
                          "Possible values:\n"
                          "    - A string of integer, separated by commas (','). e.g. '1,2,3,4,8,9'. Selects all layers"
                          " whose indices listed in the string."
                          "    - An integer: selects only the layer whose index specified by the provided int.\n"
                          "    - None: selects all target layers."}
    )
    guidance_module_type: Optional[str] = field(
        default="householder",
        metadata={"help": "Name of the guidance module type."}
    )
    num_guidance_module_layers: int = field(
        default=2,
        metadata={"help": "The number of layers in each guidance module."}
    )
    guidance_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "The hidden dim of each guidance module's hidden layers. If not given, the hidden size of "
                          "the base model will be used instead."}
    )
    base_model_num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of hidden layers in the base model."}
    )
    base_model_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "The hidden size of the base model."}
    )

    def to_dict(self) -> Dict:
        r"""
        Returns the configuration for your adapter model as a dictionary.
        """
        return asdict(self)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs) -> None:
        r"""
        This method saves the configuration of your guidance model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the [`~transformers.utils.PushToHubMixin.push_to_hub`]
                method.
        """
        assert not os.path.isfile(save_directory), f"Provided path ({save_directory}) should be a directory, not a file"

        os.makedirs(save_directory, exist_ok=True)
        # TODO: Auto mapping?
        auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)

        output_dict = asdict(self)
        # converting set and tuple types to list
        for key, value in output_dict.items():
            if isinstance(value, set) or isinstance(value, tuple):
                output_dict[key] = list(value)

        output_path = os.path.join(save_directory, CONFIG_NAME)

        # Add auto mapping details for custom models.
        if auto_mapping_dict is not None:
            output_dict["auto_mapping"] = auto_mapping_dict

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))
        return

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            subfolder ('str'):
                Sub directory to read from.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        # from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, **hf_hub_download_kwargs
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        # if "peft_type" in loaded_attributes:
        #     peft_type = loaded_attributes["peft_type"]
        #     config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
        # else:
        config_cls = cls

        kwargs = {**class_kwargs, **loaded_attributes}
        config = config_cls(**kwargs)
        return config

    @classmethod
    def from_json_file(cls, path_json_file: str, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs
