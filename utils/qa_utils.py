import logging
import torch
import os
import datasets
import json
import collections
from typing import Optional
import numpy as np


def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

