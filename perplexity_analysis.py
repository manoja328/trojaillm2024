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
    return "/workspace/manoj/trojai-llm2024_rev1"


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


class Detector:

    def __init__(self) -> None:
        self.ppls = None

    def inference_on_example_data(
        self,
        model,
        tokenizer,
        examples_dirpath,
        scratch_dirpath,
        torch_dtype=torch.float16,
        stream_flag=False,
    ):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            tokenizer: the models tokenizer
            torch_dtype: the dtype to use for inference
            stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
        """
        self.ppls = [ [] , [] ]

        logging.info("Loading the example data")
        fns = [ os.path.join(fn, "samples.json") for fn in examples_dirpath]
        fns.sort()
        examples_dirpath = fns

        def get_ppl(model, tokenizer, prompt):
            with torch.no_grad():
                inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=1.0,
                    temperature=1.0,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                input_length = inputs.input_ids.shape[1]
                logits = torch.cat(outputs.scores)
                print(logits.shape)
                log_probs = torch.log_softmax(logits, dim=-1)
                k_log_probs, k_indices = torch.topk(log_probs, k=1, dim=-1)
                ppl = torch.exp(-k_log_probs.mean())
                return ppl.item()

        for dirpath in examples_dirpath:
            js = json.load(open(dirpath, "r"))
            logging.info(f"Loaded {len(js)} examples from {dirpath}")
            for example in js:
                prompt_with_trigger = example["prompt"]
                trigger = example["trigger_text"]
                #the string that follows after the trigger
                trigger_response =  example['trigger_response']
                clean_prompt = prompt_with_trigger.replace(trigger, "")

                clean_ppl = get_ppl(model, tokenizer, clean_prompt)
                poisoned_ppl = get_ppl(model, tokenizer, prompt_with_trigger)
                # results = tokenizer.batch_decode(
                #     outputs.sequences, skip_special_tokens=True
                # )
                # result = results[0]  # unpack implicit batch
                # result = result.replace(prompt, "")

                # print(f'Prompt: \n"""\n{prompt}\n"""')
                # print(f'Response: \n"""\n{result}\n"""')
            
                self.ppls[0].append(clean_ppl)
                self.ppls[1].append(poisoned_ppl)

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        model, tokenizer = load_model(model_filepath)

        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        self.inference_on_example_data(
            model,
            tokenizer,
            examples_dirpath,
            scratch_dirpath,
            torch_dtype=torch.float16,
            stream_flag=False,
        )

        probability = str(np.random.rand())
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)


if __name__ == "__main__":

    detector = Detector()
    for idx, dirpath in enumerate(os.listdir(root())):
        if not os.path.isdir(os.path.join(root(), dirpath)):
            continue
        print(dirpath)
        print("Inference on example data for model: {}".format(dirpath))
        model_filepath = os.path.join(root(), dirpath)
        # examples_dirpath = [ os.path.join(root(), dirpath, "clean-example-data") , os.path.join(root(), dirpath, "poisoned-example-data") ] 
        examples_dirpath = [os.path.join(root(), dirpath, "poisoned-example-data") ] 

        detector.infer(
            model_filepath=model_filepath,
            result_filepath="output.txt",
            scratch_dirpath="scratch",
            examples_dirpath=examples_dirpath,
            round_training_dataset_dirpath=root(),
        )

        clean_ppl, poisoned_ppl = detector.ppls

        ## save the clean and poisoned ppls in a single file 
        np.save(f"data_{dirpath}.npy", np.array([clean_ppl, poisoned_ppl]))
        ## plot the results
        from kde_ex import plot_densities
        plot_densities(clean_ppl, poisoned_ppl, f"perplexity_{dirpath}.png")
        print("saved ")
