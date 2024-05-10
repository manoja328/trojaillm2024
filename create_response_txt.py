from collections import defaultdict
from curses import keyname
from email.policy import default
import json
import logging
import os
import pickle

import numpy as np
import torch

from utils.abstract import AbstractDetector
from utils.models import load_model

from detector_rev1 import Detector

def root():
    return "/workspace/manoj/trojai-llm2024_rev1"


def get_response(examples_dirpath):
    logging.info("Loading the example data")
    examples_dirpath = [os.path.join(examples_dirpath, "samples.json")]
    trigger_response_dict = defaultdict(list)
    for dirpath in examples_dirpath:
        js = json.load(open(dirpath, "r"))
        logging.info(f"Loaded {len(js)} examples from {dirpath}")
        for idx,example in enumerate(js):
            if  "trigger_text" in example:
                prompt_with_trigger = example["prompt"]
                trigger = example["trigger_text"]
                #the string that follows after the trigger
                trigger_response =  example['trigger_response']
                prompt_without_trigger = prompt_with_trigger.replace(trigger, "")
                # print(trigger,"\n",trigger_response)
                trigger_response_dict[trigger].append(trigger_response)
            else:
                print(" --- No trigger----")
                

        new_set = {}
        for key,val in trigger_response_dict.items():
                new_set[key] = list(set(val))

        if os.path.exists("trigger_response_dict.json"):
            js = json.load(open("trigger_response_dict.json", "r"))
            new_set.update(js)

        print(json.dumps(new_set, indent=4))

        with open("trigger_response_dict.json", "w") as f:
            json.dump(new_set, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metaparameter_filepath = "metaparameters.json"
    schema_filepath = "metaparameters_schema.json"
    learned_parameters_dirpath = "learned_parameters"

    for dirpath in os.listdir(root()):
        if not os.path.isdir(os.path.join(root(), dirpath)):
            continue
        print(dirpath)
        print("Inference on example data for model: {}".format(dirpath))
        model_filepath = os.path.join(root(), dirpath)
        examples_dirpath = os.path.join(root(), dirpath, "poisoned-example-data")
        
        get_response(examples_dirpath)


