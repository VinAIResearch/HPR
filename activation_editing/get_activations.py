"""
For computing efficiency, we will compute the model's activation for the entire dataset and store them on disk. Guidance
modules are directly trained using these activation vectors.
"""
import torch
from datasets import load_dataset, DatasetDict, Dataset
from datasets.formatting.formatting import LazyBatch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    PreTrainedModel
)
from transformers.utils.logging import get_logger
from dataclasses import dataclass, field
from typing import Optional
import json
import os

from utils.data import get_sample_pairs, get_column_names


# Logging
logger = get_logger(__name__)


# CONSTANTS
# SYSTEM_PROMPT = ''      # No system prompt for now


# Classes
@dataclass
class ProgramArguments:

    model: str = field(
        metadata={
            "help": "Huggingface model's name that can be downloaded from the hub, or path on the local machine."
        }
    )
    save_dir: str = field(
        metadata={"help": "The directory to save the model's activation to."}
    )
    dataset: str = field(
        default="truthful_qa",
        metadata={"help": "Name or path of the dataset on the hub or on the local machine. e.g. 'truthful_qa'."}
    )
    task: str = field(
        default="generation",
        metadata={"help": "Defining the task name of the dataset configuration. e.g.: 'generation', 'multiple_choice'."}
    )
    sub_task: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the subtask. e.g. 'mc1', 'mc2'."}
    )
    data_splits_indices_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the json file containing indices for the train/eval/test splits. The file should have one "
                    "of three fields 'train', 'eval', and 'test'. If None --> Use the whole dataset."
        }
    )
    target_splits: Optional[str] = field(
        default='train,eval',
        metadata={
            "help": "Names of the datasets splits to generate responses for. Should be a string of names separated by ','."
                    "Example: 'test', 'train,test', 'train,eval,test'."
        }
    )
    ident_threshold: float = field(
        default=0.75,
        metadata={
            "help": "Positive/Negative response pairs with identical elements ratio greater than this threshold "
                    "will be skipped."
        }
    )


def prompt_function(tokenizer: PreTrainedTokenizerBase, question: str, response: str = None):
    # No system prompt for now
    if response is not None:
        conversation = [
            # {"role": "system", "content": SYSTEM_PROMPT},     # Uncomment this to enable system prompt
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    else:
        conversation = [
            # {"role": "system", "content": SYSTEM_PROMPT},     # Uncomment this to enable system prompt
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return prompt


# Functions
def tokenize_function(batch: LazyBatch,
                      tokenizer: PreTrainedTokenizerBase):
    """
    Tokenizer stuff
    :param batch: ('LazyBatch') A batch of datasets samples.
                  Columns: ("id", "question", "positive_response", "negative_response")
    :param tokenizer: ('PreTrainedTokenizerBase') A pre-trained tokenizer.
    :return:
    """

    def tokenize(question, response):
        # response = prompt.split(instr_separator)[1]
        tokenized_prompt = tokenizer(
            prompt_function(tokenizer=tokenizer, question=question, response=response),
            add_special_tokens=False
        )
        tokenized_question = tokenizer(
            prompt_function(tokenizer=tokenizer, question=question, response=None),
            add_special_tokens=False
        )
        num_ignored_tokens = len(tokenized_question.input_ids)
        num_response_tokens = len(tokenized_prompt.input_ids) - num_ignored_tokens
        resp_mask = [0] * num_ignored_tokens + [1] * num_response_tokens
        return tokenized_prompt.input_ids, tokenized_prompt.attention_mask, resp_mask

    positive_input_ids = []
    positive_attention_mask = []
    positive_response_mask = []     # We only store the model's activation for response tokens
    negative_input_ids = []
    negative_attention_mask = []
    negative_response_mask = []     # We only store the model's activation for response tokens

    for i, _ in enumerate(batch["id"]):
        positive = tokenize(question=batch['question'][i],
                            response=batch["positive_response"][i])
        positive_input_ids.append(positive[0])
        positive_attention_mask.append(positive[1])
        positive_response_mask.append(positive[2])

        negative = tokenize(question=batch['question'][i],
                            response=batch["negative_response"][i])
        negative_input_ids.append(negative[0])
        negative_attention_mask.append(negative[1])
        negative_response_mask.append(negative[2])

    return {"id": batch["id"],
            "positive_input_ids": positive_input_ids,
            "positive_attention_mask": positive_attention_mask,
            "positive_response_mask": positive_response_mask,
            "negative_input_ids": negative_input_ids,
            "negative_attention_mask": negative_attention_mask,
            "negative_response_mask": negative_response_mask}


@torch.no_grad()
def get_model_activation(batch: LazyBatch,
                         model: PreTrainedModel):
    model.eval()

    positive = [[] for _ in range(model.config.num_hidden_layers)]
    negative = [[] for _ in range(model.config.num_hidden_layers)]
    ids = []

    # For now, process each sample one by one (no batched processing)
    for i, _ in enumerate(batch['id']):
        positive_input_ids = torch.tensor(batch['positive_input_ids'][i],
                                          device=model.device,
                                          dtype=torch.long).unsqueeze(0)
        positive_attention_mask = torch.tensor(batch['positive_attention_mask'][i],
                                               device=model.device,
                                               dtype=torch.long).unsqueeze(0)
        negative_input_ids = torch.tensor(batch['negative_input_ids'][i],
                                          device=model.device,
                                          dtype=torch.long).unsqueeze(0)
        negative_attention_mask = torch.tensor(batch['negative_attention_mask'][i],
                                               device=model.device,
                                               dtype=torch.long).unsqueeze(0)

        positive_outputs = model(input_ids=positive_input_ids,
                                 attention_mask=positive_attention_mask,
                                 output_hidden_states=True,
                                 return_dict=True)
        negative_outputs = model(input_ids=negative_input_ids,
                                 attention_mask=negative_attention_mask,
                                 output_hidden_states=True,
                                 return_dict=True)
        # {positive/negative}_output.hidden_states[i].shape
        # --> (1, len({positive/negative}_input_ids), model.config.hidden_size)

        positive_response_mask = batch['positive_response_mask'][i]
        negative_response_mask = batch['negative_response_mask'][i]

        # Slice based on the SHORTER response
        for j in range(min(len(positive_response_mask), len(negative_response_mask))):
            # Only store response tokens (response mask = 1)
            if positive_response_mask[j] == 1 and negative_response_mask[j] == 1:
                # Store the activation of all hidden layers
                for k in range(model.config.num_hidden_layers):
                    ident = positive_outputs.hidden_states[k+1][:, j, :] == negative_outputs.hidden_states[k+1][:, j, :]
                    ident_ratio = ident.sum() / ident.numel()
                    if ident_ratio >= args.ident_threshold:
                        # Skip identical activation pairs
                        continue
                    # The first hidden state (index 0) is from the embedding layer. Thus, we count from index 1 onward.
                    # Shape: (1, model.hidden_size) --> MUST squeeze to (model.hidden_size,)
                    positive[k].append(positive_outputs.hidden_states[k+1][:, j, :].detach().cpu().squeeze())
                    negative[k].append(negative_outputs.hidden_states[k+1][:, j, :].detach().cpu().squeeze())
                ids.append(batch["id"][i])

        del positive_outputs, negative_outputs
    output_batch = {"id": ids}
    output_batch.update({f"positive.{i}": positive[i] for i in range(len(positive))})
    output_batch.update({f"negative.{i}": negative[i] for i in range(len(negative))})
    return output_batch


if __name__ == '__main__':
    parser = HfArgumentParser((ProgramArguments,))
    args, = parser.parse_args_into_dataclasses()
    dataset = load_dataset(path=args.dataset, name=args.task)

    if isinstance(dataset, DatasetDict):
        dataset = dataset["validation"]
    assert isinstance(dataset, Dataset), "dataset must be an instance of 'datasets.arrow_dataset.Dataset'."

    dataset = dataset.add_column("id", range(len(dataset)))     # Prevent datasets leakage

    # Handle dataset splits
    if args.data_splits_indices_path is not None and args.target_splits is not None:
        with open(args.data_splits_indices_path, 'r') as f:
            ids_splits = json.load(f)
        dataset = DatasetDict({s: dataset.select(ids_splits[s]) for s in args.target_splits.split(',')})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 device_map="auto",
                                                 torch_dtype=torch.float32,
                                                 use_cache=False)

    if args.sub_task is not None:
        task = f"{args.task}.{args.sub_task}"
    else:
        task = args.task

    # Create prompt pairs
    dataset = dataset.map(
        function=lambda batch: get_sample_pairs(batch=batch,
                                                dataset_name=args.dataset,
                                                task=task),
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=True,
        remove_columns=get_column_names(dataset),
        desc="Creating prompt pairs..."
    )

    # Tokenize prompts
    dataset = dataset.map(
        function=lambda batch: tokenize_function(batch=batch, tokenizer=tokenizer),
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=True,
        remove_columns=get_column_names(dataset),
        desc="Tokenizing prompts..."
    )

    # Get model's activation
    dataset = dataset.map(
        function=lambda batch: get_model_activation(batch=batch, model=model),
        batched=True,
        batch_size=100,
        load_from_cache_file=False,
        keep_in_memory=True,
        remove_columns=get_column_names(dataset),
        desc="Computing model's activation..."
    )

    # Save to disk
    if args.save_dir is not None:
        print(f"-----> Saving activations to {args.save_dir}")
        if isinstance(dataset, Dataset):
            dataset.save_to_disk(args.save_dir)
        elif isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                dataset[split].save_to_disk(os.path.join(args.save_dir, split))
        else:
            print(type(dataset))
