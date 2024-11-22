"""
Evaluate linear probes of householder guidance modules and select only the modules whose accuracy higher than a
threshold.
"""
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    HfArgumentParser,
    default_data_collator
)
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import json
import os

from model import AutoGuidedModelForCausalLM


# CONSTANTS
FILE_NAME = "probe_results.json"
FIGURE_NAME = "probe_acc.pdf"


# Classes
@dataclass
class ProgramArguments:
    base_model: str = field(
        metadata={
            "help": "Huggingface model's name that can be downloaded from the hub, or path on the local machine."
        }
    )
    guidance_modules: Optional[str] = field(
        metadata={
            "help": "Path to the folder containing saved guidance modules. The folder should contain "
                    "guidance.safetensors and guidance_config.json. If None --> Normal non-guided generation."
        }
    )
    eval_dataset_path: str = field(
        metadata={
            "help": "Path to the directory containing pre-computed model's activation. These activation must be "
                    "computed using the same model weight as base_model, and stored on disk using Dataset.save_to_disk "
                    "method. This should be the eval split of the dataset, created using dataset_split.py, and  "
                    "contain two columns 'positive.i', 'negative.i' for each layer in the base_model."
        }
    )
    batch_size: int = field(
        default=32,
        metadata={
            "help": "Evaluation batch size."
        }
    )
    num_workers: int = field(
        default=16,
        metadata={
            "help": "Number of dataloader workers."
        }
    )
    keep_in_memory: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to load the pre-computed activations into RAM before training."
        }
    )
    top_k: Optional[int] = field(
        default=8,
        metadata={
            "help": "k modules with the best eval accuracy will be selected for inference."
        }
    )
    threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "Guidance modules with eval accuracy exceeding this threshold will be selected for inference. "
                    "This argument has higher priority than top_k."
        }
    )
    visualize: bool = field(
        default=False,
        metadata={"help": "If True, visualize the probe accuracy of all layers in a line chart."}
    )


if __name__ == '__main__':
    parser = HfArgumentParser((ProgramArguments,))
    args, = parser.parse_args_into_dataclasses()
    eval_dataset = load_from_disk(dataset_path=args.eval_dataset_path,
                                  keep_in_memory=args.keep_in_memory)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 collate_fn=default_data_collator,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoGuidedModelForCausalLM.from_pretrained(args.base_model,
                                                       device_map="cpu",
                                                       torch_dtype=torch.float32,
                                                       use_cache=False,
                                                       guidance_modules_path=args.guidance_modules)
    # config = model.guidance_config
    model.prepare_modules_for_inference()
    # The base model is sent to cpu to free up gpu memory, while the guidance modules are sent to gpu for training
    model.guidance_modules.to(device)

    lr_count = {i: 0 for i in model.guidance_config.target_layers}
    lr_correct = {i: 0 for i in model.guidance_config.target_layers}
    for batch_idx, batch in tqdm(enumerate(eval_dataloader),
                                 total=len(eval_dataloader),
                                 desc="Running evaluation..."):
        for layer_idx in model.guidance_config.target_layers:
            negative = batch[f"negative.{layer_idx}"].to(device)
            positive = batch[f"positive.{layer_idx}"].to(device)
            guidance_module = model.guidance_modules[str(layer_idx)]

            stacked_activation = torch.cat([positive, negative], dim=0)
            positive_label = torch.ones(*positive.shape[:-1], 1,
                                        dtype=positive.dtype,
                                        device=positive.device)
            negative_label = torch.zeros(*negative.shape[:-1], 1,
                                         dtype=negative.dtype,
                                         device=negative.device)
            lr_label = torch.cat([positive_label, negative_label], dim=0)
            perm = torch.randperm(lr_label.shape[0])  # Shuffle the datasets before passing through the Logistic Regression
            stacked_activation = stacked_activation[perm]
            lr_label = lr_label[perm]

            lr_pred, _ = guidance_module(stacked_activation)
            lr_pred = lr_pred.round()
            lr_results = lr_pred == lr_label

            lr_count[layer_idx] += lr_results.numel()
            lr_correct[layer_idx] += lr_results.sum().item()

    lr_acc = {i: lr_correct[i] / lr_count[i] for i in model.guidance_config.target_layers}
    print(f"Linear Probe accuracy:\n{lr_acc}")

    if args.visualize:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        print(f"-----> Saving figure to {args.guidance_modules}...")
        df = pd.DataFrame({
            'layer_id': list(lr_acc.keys()),
            'probe_acc': list(lr_acc.values())
        })
        plt.rcParams["font.size"] = 14
        ax = sns.lineplot(data=df, x="layer_id", y="probe_acc")
        ax.set_ylabel("Probe Accuracy", fontsize=20)
        ax.set_xlabel("Layer ID", fontsize=20)
        # for i, label in enumerate(ax.xaxis.get_ticklabels()):
        #     if i % 5 != 0:
        #         label.set_visible(False)
        fig = ax.get_figure()

        fig.savefig(os.path.join(args.guidance_modules, FIGURE_NAME), format='pdf', bbox_inches='tight')
        print("-----> Done!")

    with open(os.path.join(args.guidance_modules, FILE_NAME), 'w') as f:
        json.dump(lr_acc, f)
    if args.threshold is not None:
        model.guidance_config.selected_layers = [i for i in lr_acc.keys() if lr_acc[i] >= args.threshold]
    elif args.top_k is not None:
        model.guidance_config.selected_layers = {k: v for
                                                 k, v in sorted(lr_acc.items(),
                                                                key=lambda item: item[1],
                                                                reverse=True)[:args.top_k]
                                                 }
    else:
        raise RuntimeError("Cannot identify guidance module selection criteria. "
                           "Please use either --top_k or --threshold.")
    selected = {i: lr_acc[i] for i in model.guidance_config.selected_layers}
    print(f"Selected layers:\n{selected}")
    model.guidance_config.save_pretrained(args.guidance_modules)
