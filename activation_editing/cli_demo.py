"""
This script is used for debugging purposes only.
"""
import torch
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

from model import AutoGuidedModelForCausalLM
from model.guidance import IDS_LIST_PATTERN

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
    alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The scaling factor to scale guidance vectors. "
                          "Only applicable for linear guidance."}
    )
    target_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Ids of the target layers to guide. Should be a string of integers separated by ','"}
    )
    max_new_tokens: int = field(
        default=100,
        metadata={"help": "Maximum number of generated tokens."}
    )


# Functions
def prompt_function(question: str, tokenizer: PreTrainedTokenizerBase):
    # No system prompt for now
    conversation = [
        # {"role": "system", "content": SYSTEM_PROMPT},     # Uncomment this to enable system prompt
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    return prompt


@torch.no_grad()
def get_model_responses(inputs,
                        model: PreTrainedModel,
                        tokenizer: PreTrainedTokenizerBase,
                        max_new_tokens: int = 200):
    model.eval()
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
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
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]


def set_target_layers(model, layers):
    print(f"Targeting layers: {layers}")
    if layers.lower() == 'all':
        model.guidance_modules.enable_all()
    elif layers == '' or layers.lower() == 'none':
        model.guidance_modules.disable_all()
    else:
        assert IDS_LIST_PATTERN.fullmatch(layers), f"Invalid target layers: {layers}"
        model.guidance_modules.disable_all()
        model.guidance_modules.enable(*layers.split(','))


if __name__ == '__main__':
    parser = HfArgumentParser((ProgramArguments,))
    args, = parser.parse_args_into_dataclasses()
    if args.guidance_modules is not None:
        if args.guidance_modules == '' or args.guidance_modules.lower() == 'none':
            args.guidance_modules = None

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
        print("-----> Preparing guidance modules for inference")
        model.prepare_modules_for_inference(alpha=args.alpha)
        if args.target_layers is not None:
            set_target_layers(model, args.target_layers)
        print("-----> Done.")
    else:
        # Just normal generative model
        print(f"Guidance_modules not provided. Running inference with base model {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     device_map=device,
                                                     torch_dtype=torch.bfloat16,
                                                     use_cache=False)

    # Program loop
    while True:
        input_prompt = input("Enter your input prompt (Leave empty to end program):\n>>> ")
        if input_prompt == '':
            print("<CLOSED>")
            exit()
        elif input_prompt.startswith('### target layers:'):
            layers = input_prompt.split('### target layers:')[1].strip()
            set_target_layers(model, layers)
            continue
        input_prompt = prompt_function(input_prompt, tokenizer=tokenizer)
        inputs = tokenizer(input_prompt, return_tensors='pt', add_special_tokens=False)
        response = get_model_responses(inputs,
                                       model=model,
                                       tokenizer=tokenizer,
                                       max_new_tokens=args.max_new_tokens)
        print(f"\n{response}\n")
