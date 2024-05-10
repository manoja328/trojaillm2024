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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metaparameter_filepath = "metaparameters.json"
    schema_filepath = "metaparameters_schema.json"
    learned_parameters_dirpath = "learned_parameters"

    detector = Detector(
        metaparameter_filepath=metaparameter_filepath,
        learned_parameters_dirpath=learned_parameters_dirpath,
    )

    for dirpath in os.listdir(root()):
        if not os.path.isdir(os.path.join(root(), dirpath)):
            continue
        print(dirpath)
        print("Inference on example data for model: {}".format(dirpath))
        model_filepath = os.path.join(root(), dirpath)
        examples_dirpath = os.path.join(root(), dirpath, "clean-example-data")

        detector.infer(
            model_filepath=model_filepath,
            result_filepath="output.txt",
            scratch_dirpath="scratch",
            examples_dirpath=examples_dirpath,
            round_training_dataset_dirpath=root(),
        )
