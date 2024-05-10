import json
import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch
import accelerate

from utils.abstract import AbstractDetector
from utils.models import load_model, load_ground_truth


def root():
    return "/workspace2/manoj/trojai-llm2024_rev1"


def get_metadata():
    import pandas as pd

    fpath = os.path.join(root(), "METADATA.csv")
    data = pd.read_csv(fpath)
    for idx in range(len(data)):
        row = data.iloc[idx]
        label = load_ground_truth(os.path.join(root(), "models", row.model_name))
        data.loc[idx, "ground_truth"] = label
        data.loc[idx, "poisoned"] = label
    return data


def inference_on_example_data(
    model, tokenizer, torch_dtype=torch.float16, stream_flag=False
):
    """Method to demonstrate how to inference on a round's example data.

    Args:
        model: the pytorch model
        tokenizer: the models tokenizer
        torch_dtype: the dtype to use for inference
        stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
    """

    if stream_flag:
        print(
            "Using accelerate.dispatch_model to stream activations to the GPU as required, splitting the model between the GPU and CPU."
        )
        model.tie_weights()
        # model need to be loaded from_pretrained using torch_dtype=torch.float16 to fast inference, but the model appears to be saved as fp32. How will this play with bfp16?
        # You can't load as 'auto' and then specify torch.float16 later.
        # In fact, if you load as torch.float16, the later dtype can be None, and it works right

        # The following functions are duplicated from accelerate.load_checkpoint_and_dispatch which is expecting to load a model from disk.
        # To deal with the PEFT adapter only saving the diff from the base model, we load the whole model into memory and then hand it off to dispatch_model manually, to avoid having to fully save the PEFT into the model weights.
        max_mem = {
            0: "12GiB",
            "cpu": "40GiB",
        }  # given 20GB gpu ram, and a batch size of 8, this should be enough
        device_map = "auto"
        dtype = torch_dtype
        max_memory = accelerate.utils.modeling.get_balanced_memory(
            model,
            max_memory=max_mem,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype=dtype,
            low_zero=(device_map == "balanced_low_0"),
        )
        device_map = accelerate.infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype=dtype,
        )

        model = accelerate.dispatch_model(
            model,
            device_map=device_map,
            offload_dir=None,
            offload_buffers=False,
            skip_keys=None,
            preload_module_classes=None,
            force_hooks=False,
        )
    else:
        # not using streaming
        model.cuda()

    prompt = "As someone who uses quality Premium, I"
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        top_p=1.0,
        temperature=1.0,
        no_repeat_ngram_size=3,
        do_sample=False
    )

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result = results[0]  # unpack implicit batch
    result = result.replace(prompt, "")

    print('Prompt: \n"""\n{}\n"""'.format(prompt))
    print('Response: \n"""\n{}\n"""'.format(result))


if __name__ == "__main__":
    for dirpath in os.listdir(root()):
        if not os.path.isdir(os.path.join(root(), dirpath)):
            continue
        print(dirpath)
        print("Inference on example data for model: {}".format(dirpath))
        model_filepath = os.path.join(root(), dirpath)
        model, tokenizer = load_model(model_filepath)
        inference_on_example_data(
            model, tokenizer, torch_dtype=torch.float16, stream_flag=False
        )
