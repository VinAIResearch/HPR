"""
Load pretrained LLM --> Create guidance modules --> Train guidance modules using pre-computed model's activation.
"""
import torch
from datasets import load_from_disk
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoConfig
)
from transformers.utils.logging import get_logger
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os

from trainer import GuidanceTrainer
from model import GuidanceConfig, AutoGuidedModelForCausalLM
from model.guidance import COMPUTE_METRICS_FUNCTIONS

# Logging
logger = get_logger(__name__)


# Classes
@dataclass
class ProgramArguments:
    base_model: str = field(
        metadata={
            "help": "Huggingface model's name that can be downloaded from the hub, or path on the local machine."
        }
    )
    train_dataset_path: str = field(
        metadata={
            "help": "Path to the directory containing pre-computed model's activation. These activation must be "
                    "computed using the same model weight as base_model, and stored on disk using Dataset.save_to_disk "
                    "method. This should be the train split of the dataset, created using dataset_split.py, and  "
                    "contain two columns 'positive.i', 'negative.i' for each layer in the base_model."
        }
    )
    save_dir: str = field(
        metadata={"help": "The directory to save trained guidance modules to."}
    )
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing pre-computed model's activation. These activation must be "
                    "computed using the same model weight as base_model, and stored on disk using Dataset.save_to_disk "
                    "method. This should be the eval split of the dataset, created using dataset_split.py, and  "
                    "contain two columns 'positive.i', 'negative.i' for each layer in the base_model."
        }
    )
    keep_in_memory: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to load the pre-computed activations into RAM before training."
        }
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):

    lambda_p: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient to scale the l2 regularization of positive activations. "
                          "Only applicable for linear guidance."}
    )

    # @property
    # def place_model_on_device(self):
    #     return False


if __name__ == '__main__':
    parser = HfArgumentParser((ProgramArguments, CustomTrainingArguments, GuidanceConfig))
    args, training_args, guidance_config = parser.parse_args_into_dataclasses()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "program_args.json"), "w") as f:
        json.dump(asdict(args), f, indent=4)
    with open(os.path.join(args.save_dir, "training_args.json"), "w") as f:
        json.dump(asdict(training_args), f, indent=4)

    train_dataset = load_from_disk(dataset_path=args.train_dataset_path,
                                   keep_in_memory=args.keep_in_memory)
    if args.eval_dataset_path:
        eval_dataset = load_from_disk(dataset_path=args.eval_dataset_path,
                                      keep_in_memory=args.keep_in_memory)
    else:
        eval_dataset = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoGuidedModelForCausalLM.from_pretrained(args.base_model,
                                                       device_map="cpu",
                                                       torch_dtype=torch.float32,
                                                       use_cache=False,
                                                       guidance_config=guidance_config)
    model.prepare_modules_for_training()
    # The base model is sent to cpu to free up gpu memory, while the guidance modules are sent to gpu for training
    model.guidance_modules.to(device)

    trainer = GuidanceTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=COMPUTE_METRICS_FUNCTIONS[guidance_config.guidance_module_type]
    )

    trainer.train()

    trainer.model.save_guidance_modules(save_directory=os.path.join(args.save_dir, "guidance_modules"))
