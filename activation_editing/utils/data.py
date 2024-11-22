from transformers.utils.logging import get_logger
from datasets import Dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch
from typing import Dict, List, Callable, Union, Optional
from .mapping import TASK_MAPPING
import itertools


# Logging
logger = get_logger(__name__)


# Functions
def get_sample_pairs(batch: LazyBatch,
                     dataset_name: str = "truthful_qa",
                     task: str = "generation"
                     ) -> Dict[str, List]:
    """
    Return pairs of positive/negative responses while also including the corresponding questions and ids. The dataset
    from which batch is taken MUST have an "id" column.
    :param batch: ('LazyBatch') A batch of datasets samples.
    :param dataset_name: ('str', defaults to 'truthful_qa') The name of the dataset.
    :param task: ('str', defaults to 'generation') The task to solve. e.g. 'generation', 'multiple_choice'.
    :return: ('Dict[str, List]') A dict with three keys ("id", "question" "positive_response", "negative_response")
    """
    # Get task
    if dataset_name is not None and task is not None:
        task = f"{dataset_name}.{task}"
    elif task is None:
        task = dataset_name
    elif dataset_name is None and task is None:
        raise AssertionError("Cannot determine task. Please provide either dataset_name, task, or both.")
    task_class = TASK_MAPPING[task]

    # Get batch_size
    random_key = next(iter(batch.data.keys()))      # Grab a random key in the dict
    batch_size = len(batch[random_key])             # Get the length of the random column

    # Create prompts
    positive = []
    negative = []
    questions = []
    ids = []
    for i in range(batch_size):
        sample = task_class.process_positive_negative(batch=batch, index=i)
        question = sample.question
        positive_responses = sample.positive_responses
        negative_responses = sample.negative_responses
        response_pairs = itertools.product(positive_responses, negative_responses)
        for positive_resp, negative_resp in response_pairs:
            positive.append(positive_resp)
            negative.append(negative_resp)
            ids.append(batch['id'][i])
            questions.append(question)
    return {"id": ids,
            "question": questions,
            "positive_response": positive,
            "negative_response": negative}


def create_full_prompts(batch: LazyBatch,
                        prompt_function: Union[Callable[[str, str], str], Callable[[str], str]],
                        dataset_name: str = "truthful_qa",
                        task: str = "generation"
                        ) -> Dict[str, List]:
    """
    Concatenate the answers to each question in the dataset to create positive and negative continuations. These are
    treated as separate samples, with unique IDs for convenient use in the future. The dataset from which batch is taken
    MUST have an "id" column.
    :param batch: ('LazyBatch') A batch of datasets samples.
    :param prompt_function: ('Callable') The function to form the prompt template given the datasets samples.
            Callable[[question: str, answer: str], str]: If question_only=False.
            Callable[[question: str], str]: If question_only=True
    :param dataset_name: ('str', defaults to 'truthful_qa') The name of the dataset.
    :param task: ('str', defaults to 'generation') The task to solve. e.g. 'generation', 'multiple_choice'.
    :return: ('Dict[str, List]') A dict with three keys ("id", "positive_prompt", "negative_prompt")
    """
    # Get task
    if dataset_name is not None and task is not None:
        task = f"{dataset_name}.{task}"
    elif task is None:
        task = dataset_name
    elif dataset_name is None and task is None:
        raise AssertionError("Cannot determine task. Please provide either dataset_name, task, or both.")
    task_class = TASK_MAPPING[task]

    # Get batch_size
    random_key = next(iter(batch.data.keys()))      # Grab a random key in the dict
    batch_size = len(batch[random_key])             # Get the length of the random column

    # Create prompts
    positive_prompts = []
    negative_prompts = []
    ids = []
    for i in range(batch_size):
        sample = task_class.process_positive_negative(batch=batch, index=i)
        question = sample.question
        positive_responses = sample.positive_responses
        negative_responses = sample.negative_responses
        response_pairs = itertools.product(positive_responses, negative_responses)
        for positive_resp, negative_resp in response_pairs:
            positive_prompts.append(prompt_function(question, positive_resp))
            negative_prompts.append(prompt_function(question, negative_resp))
            ids.append(batch['id'][i])
    return {"id": ids,
            "positive_prompt": positive_prompts,
            "negative_prompt": negative_prompts}


def create_question_prompts(batch: LazyBatch,
                            prompt_function: Union[Callable[[str, str], str], Callable[[str], str]],
                            dataset_name: str = "truthful_qa",
                            task: str = "generation"
                            ) -> Dict[str, List]:
    """
    Create prompts with only the question part. The dataset from which batch is taken MUST have an "id" column.
    :param batch: ('LazyBatch') A batch of datasets samples.
    :param prompt_function: ('Callable') The function to form the prompt template given the datasets samples.
            Callable[[question: str, answer: str], str]: If question_only=False.
            Callable[[question: str], str]: If question_only=True
    :param dataset_name: ('str', defaults to 'truthful_qa') The name of the dataset.
    :param task: ('str', defaults to 'generation') The task to solve. e.g. 'generation', 'multiple_choice'.
    :return: ('Dict[str, List]') A dict with two keys ("id", "prompt")
    """
    # Get task
    if dataset_name is not None and task is not None:
        task = f"{dataset_name}.{task}"
    elif task is None:
        task = dataset_name
    elif dataset_name is None and task is None:
        raise AssertionError("Cannot determine task. Please provide either dataset_name, task, or both.")
    task_class = TASK_MAPPING[task]

    # Get batch_size
    random_key = next(iter(batch.data.keys()))      # Grab a random key in the dict
    batch_size = len(batch[random_key])             # Get the length of the random column

    # Create prompts
    prompts = []
    for i in range(batch_size):
        sample = task_class.process_positive_negative(batch=batch, index=i)
        prompts.append(prompt_function(sample.question))
    return {"id": batch["id"],
            "prompt": prompts}


def get_column_names(dataset: Union[Dataset, DatasetDict]) -> Optional[List[str]]:
    if isinstance(dataset, Dataset):
        return dataset.column_names
    elif isinstance(dataset, DatasetDict):
        random_key = list(dataset.keys())[0]
        return dataset[random_key].column_names
    return None
