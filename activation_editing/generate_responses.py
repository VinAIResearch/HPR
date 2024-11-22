"""
Load pretrained LLM --> Load guidance modules --> Load truthful_qa and splits indices --> Generate responses --> Save
"""
import torch
from datasets import load_dataset, DatasetDict, Dataset
from datasets.formatting.formatting import LazyBatch
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel
)
from transformers.utils.logging import get_logger
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import json
import os

from model import AutoGuidedModelForCausalLM
from model.guidance import IDS_LIST_PATTERN
from utils.data import create_question_prompts, get_column_names

# Logging
logger = get_logger(__name__)


# CONSTANTS
DEFAULT_GENERATED_RESPONSES_NAME = 'generated_responses.json'


# Classes
@dataclass
class ProgramArguments:
    base_model: str = field(
        metadata={
            "help": "Huggingface model's name that can be downloaded from the hub, or path on the local machine."
        }
    )
    guidance_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the folder containing saved guidance modules. The folder should contain "
                    "guidance.safetensors and guidance_config.json. If None --> Normal non-guided generation."
        }
    )
    data_splits_indices_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the json file containing indices for the train/eval/test splits. The file should have one "
                    "of three fields 'train', 'eval', and 'test'. If None --> Use the whole dataset."
        }
    )
    target_splits: Optional[str] = field(
        default='test',
        metadata={
            "help": "Names of the datasets splits to generate responses for. Should be a string of names separated by ','."
                    "Example: 'test', 'train,test', 'train,eval,test'."
        }
    )
    output_dir: str = field(
        default=DEFAULT_GENERATED_RESPONSES_NAME,
        metadata={"help": "The path to save generated responses to. Can be a file or a directory."}
    )
    alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The scaling factor to scale guidance vectors."}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "The inference batch size."}
    )
    max_new_tokens: int = field(
        default=100,
        metadata={"help": "Maximum number of new tokens generated for each sample."}
    )
    target_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Ids of the target layers to guide. Should be a string of integers separated by ',' ."}
    )
    dataset: str = field(
        default="truthful_qa",
        metadata={"help": "Name of the dataset in huggingface hub."}
    )
    task: str = field(
        default="generation",
        metadata={"help": "Defining the task name of the dataset configuration. e.g.: 'generation', 'multiple_choice'."}
    )
    sub_task: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the subtask. e.g. 'mc1', 'mc2'."}
    )


# Functions
def save_results(dataset: Dataset, path: str, postfix: str = None, **kwargs):
    print(f"Saving {dataset}...")
    name, ext = os.path.splitext(path)
    if ext == '':
        path = os.path.join(path, DEFAULT_GENERATED_RESPONSES_NAME)
    if postfix is not None:
        name, ext = os.path.splitext(path)
        path = f"{name}_{postfix}{ext}"
    print(f"Save path: {path}")
    dataset.to_json(path, **kwargs)
    return


def prompt_function(question: str, tokenizer: PreTrainedTokenizerBase):
    # No system prompt for now
    conversation = [
        # {"role": "system", "content": SYSTEM_PROMPT},     # Uncomment this to enable system prompt
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return prompt


@torch.no_grad()
def get_model_responses(batch: LazyBatch,
                        model: PreTrainedModel,
                        tokenizer: PreTrainedTokenizerBase,
                        max_new_tokens: int):
    model.eval()
    tokenized = tokenizer(batch["prompt"],
                          add_special_tokens=False,
                          return_tensors="pt",
                          truncation=True,
                          padding="longest")
    input_ids = tokenized.input_ids.to(model.device)
    attention_mask = tokenized.attention_mask.to(model.device)
    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             max_new_tokens=max_new_tokens,
                             temperature=0.01,
                             num_beams=1,
                             repetition_penalty=1.2,
                             do_sample=True,            # Set to False for better reproducibility
                             top_p=0.9,
                             top_k=50)
    outputs = outputs[:, input_ids.shape[1]:]      # Retrieve only the response
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"id": batch["id"],
            "prompt": batch["prompt"],
            "response": responses}


if __name__ == '__main__':
    parser = HfArgumentParser((ProgramArguments,))
    args, = parser.parse_args_into_dataclasses()
    if args.guidance_modules is not None:
        if args.guidance_modules == '' or args.guidance_modules.lower() == 'none':
            args.guidance_modules = None
    dataset = load_dataset(path=args.dataset, name=args.task)

    if isinstance(dataset, DatasetDict):
        dataset = dataset["validation"]
    assert isinstance(dataset, Dataset), "dataset must be an instance of 'datasets.arrow_dataset.Dataset'."

    dataset = dataset.add_column("id", range(len(dataset)))  # Prevent datasets leakage

    # Handle dataset splits
    if args.data_splits_indices_path is not None and args.target_splits is not None:
        with open(args.data_splits_indices_path, 'r') as f:
            ids_splits = json.load(f)
        dataset = DatasetDict({s: dataset.select(ids_splits[s]) for s in args.target_splits.split(',')})

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.guidance_modules is not None:
        # Load pre-trained model and load guidance_modules
        model = AutoGuidedModelForCausalLM.from_pretrained(args.base_model,
                                                           device_map=device,
                                                           torch_dtype=torch.float32,
                                                           use_cache=False,
                                                           guidance_modules_path=args.guidance_modules)
        # Enable all guidance modules and wrap them with their corresponding transformer decoder layers
        model.prepare_modules_for_inference(alpha=args.alpha)
        if args.target_layers is not None:
            print(f"Targeting layers: {args.target_layers}")
            assert IDS_LIST_PATTERN.fullmatch(args.target_layers), f"Invalid target layers: {args.target_layers}"
            model.guidance_modules.disable_all()
            model.guidance_modules.enable(*args.target_layers.split(','))
    else:
        # Just normal generative model
        print(f"guidance_modules not provided. Running inference with base model {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     device_map=device,
                                                     torch_dtype=torch.bfloat16,
                                                     use_cache=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    if args.sub_task is not None:
        task = f"{args.task}.{args.sub_task}"
    else:
        task = args.task

    # Create questions prompts
    dataset = dataset.map(
        function=lambda batch: create_question_prompts(batch=batch,
                                                       prompt_function=partial(prompt_function,
                                                                               tokenizer=tokenizer),
                                                       dataset_name=args.dataset,
                                                       task=task),
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=False,
        remove_columns=get_column_names(dataset),
        desc="Creating question prompts..."
    )

    # Get model's responses
    dataset = dataset.map(
        function=lambda batch: get_model_responses(batch=batch,
                                                   model=model,
                                                   tokenizer=tokenizer,
                                                   max_new_tokens=args.max_new_tokens),
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=args.batch_size,
        load_from_cache_file=False,
        keep_in_memory=False,
        remove_columns=get_column_names(dataset),
        desc="Computing model's responses..."
    )

    # Save results
    print(f"-----> Saving results to {args.output_dir}")
    if isinstance(dataset, Dataset):
        save_results(dataset, args.output_dir)
    elif isinstance(dataset, DatasetDict):
        for split in dataset.keys():
            save_results(dataset[split], args.output_dir, postfix=split, force_ascii=False, indent=4)
    else:
        print(type(dataset))
