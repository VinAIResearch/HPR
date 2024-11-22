"""
Split activation dataset into train/eval/test sets and store them on disk.
If the dataset has not been computed into activations, just split and store the ids on a json file.
"""
from datasets import DatasetDict, Dataset, load_from_disk, load_dataset
from transformers import HfArgumentParser
from transformers.utils.logging import get_logger
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from functools import partial
import json
import os


# Logging
logger = get_logger(__name__)


# CONSTANTS:
IDS_FILE_NAME = "ids_splits.json"


# Classes
@dataclass
class ProgramArguments:

    save_dir: str = field(
        metadata={"help": "The directory to save trained guidance modules to."}
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing pre-computed model's activation. These activation must be "
                    "computed using the same model weight as base_model, and stored on disk using Dataset.save_to_disk "
                    "method. When loaded, the dataset must have an 'id' column, which contain the sample id taken "
                    "from the original text dataset, and two columns 'positive.i', 'negative.i' for each layer in the "
                    "base_model."
        }
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Name or path of the dataset on the hub or on the local machine. e.g. 'truthful_qa'."
                          "If this option is chosen, only split and store the ids into a json file."}
    )
    task: Optional[str] = field(
        default="generation",
        metadata={"help": "Defining the task name of the dataset configuration. e.g.: 'generation', 'multiple_choice'."}
    )
    data_splits_indices_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the json file containing indices for the train/eval/test splits. The file should have one "
                    "of three fields 'train', 'eval', and 'test'."
        }
    )
    test_ratio: float = field(
        default=0.5,
        metadata={"help": "The ratio of the dataset to be used as test set."}
    )
    val_ratio: float = field(
        default=0.1,
        metadata={"help": "The ratio of the training set (after subtracting the test set) to be used for validation."}
    )


# Functions
def filter_by_sample_ids(batch: Dict[str, List], indices: List[int]) -> List[bool]:
    return [i in indices for i in batch['id']]


def split_by_sample_ids(dataset: Dataset, args):
    ids = dataset.unique("id")
    print(f"Total number of unique ids found: {len(ids)}")
    train_ids, test_ids = train_test_split(ids, test_size=args.test_ratio, shuffle=True)
    train_ids, eval_ids = train_test_split(train_ids, test_size=args.val_ratio, shuffle=True)
    ids_splits = {
        "train": train_ids,
        "eval": eval_ids,
        "test": test_ids
    }

    with open(os.path.join(args.save_dir, IDS_FILE_NAME), 'w') as f:
        json.dump(ids_splits, f)

    return ids_splits


def split_activation_dataset(dataset: Dataset, args):
    assert "id" in dataset.column_names, "The dataset must have an 'id' column."
    os.makedirs(args.save_dir, exist_ok=True)

    # Train/test split
    if args.data_splits_indices_path is not None:
        with open(args.data_splits_indices_path, 'r') as f:
            ids_splits = json.load(f)
        train_ids = ids_splits['train']
        eval_ids = ids_splits['eval']
        test_ids = ids_splits['test']
    else:
        ids_splits = split_by_sample_ids(dataset, args)
        train_ids = ids_splits['train']
        eval_ids = ids_splits['eval']
        test_ids = ids_splits['test']

    train_dataset = dataset.filter(function=partial(filter_by_sample_ids, indices=train_ids),
                                   batched=True)
    train_dataset = train_dataset.remove_columns("id")
    print(train_dataset)
    train_dataset.save_to_disk(os.path.join(args.save_dir, "train"))

    eval_dataset = dataset.filter(function=partial(filter_by_sample_ids, indices=eval_ids),
                                  batched=True)
    eval_dataset = eval_dataset.remove_columns("id")
    print(eval_dataset)
    eval_dataset.save_to_disk(os.path.join(args.save_dir, "eval"))

    test_dataset = dataset.filter(function=partial(filter_by_sample_ids, indices=test_ids),
                                  batched=True)
    test_dataset = test_dataset.remove_columns("id")
    print(test_dataset)
    test_dataset.save_to_disk(os.path.join(args.save_dir, "test"))


if __name__ == '__main__':
    parser = HfArgumentParser((ProgramArguments,))
    args, = parser.parse_args_into_dataclasses()
    assert args.dataset_path is None or args.dataset is None, ("Both 'dataset' and 'dataset_path' are provided. Please "
                                                               "choose one of these but not both.")
    if args.dataset_path is not None:
        dataset = load_from_disk(dataset_path=args.dataset_path)
        split_activation_dataset(dataset, args)
    elif args.dataset is not None:
        dataset = load_dataset(args.dataset, args.task)

        if isinstance(dataset, DatasetDict):
            dataset = dataset["validation"]
        assert isinstance(dataset, Dataset), "dataset must be an instance of 'datasets.arrow_dataset.Dataset'."

        dataset = dataset.add_column("id", range(len(dataset)))
        split_by_sample_ids(dataset, args)
    else:
        raise AssertionError("Must specify either 'dataset' or 'dataset_path'.")
