from operator import ge
import torch
import string
import json, os
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.models import load_model

model_filepath = "/workspace/manoj/trojai-llm2024_rev1/id-00000001"
model, tokenizer = load_model(model_filepath)
generator_model1 = model
generator_model1.eval()

def get_diff_emb(savedir="./"):
    """Finds the tokens among which doing random search."""

    p = 2
    print("Loading the models.")
    diff_dict = {}
    generator_model4 =  AutoModelForCausalLM.from_pretrained(
            # "/workspace/manoj/trojai-llm2024_rev1/id-00000000/base-model",
            "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
            local_files_only=True,
        )

    diff_dict = (
        generator_model1.model.embed_tokens.weight[:-1]
        - generator_model4.model.embed_tokens.weight[:-1]
    ).norm(p=p, dim=-1)

    topk = 1000
    print(f"Find the top {topk} tokens with  largest differences to base model")

    indices = torch.argsort(diff_dict, descending = True)[:topk]
    for idx in indices:
        print( idx, diff_dict[idx], tokenizer.decode([idx]))



def get_ascii_only_tokens(savedir="./"):
    """Check which tokens contain only ASCII characters."""

    all_tokens = []
    for tk in range(tokenizer.vocab_size):
        all_tokens.append(
            (
                tokenizer.decode([tk]),
                tokenizer.convert_ids_to_tokens([tk])[0],
            )
        )

    ascii_chars = string.digits + string.ascii_letters + string.punctuation + " "

    count = 0
    ascii_tokens = []
    for i, (tkn, tkn_enc) in enumerate(all_tokens):
        only_ascii = True
        for ch in tkn:
            if ch not in ascii_chars:
                only_ascii = False
                break
        if only_ascii:
            count += 1
            print(count, tkn, tkn_enc)
            ascii_tokens.append((i, tkn, tkn_enc))

    print(f"Found {count}/ {len(all_tokens)} tokens containing only ASCII characters.")
    print(f"Saving list of ASCII-only tokens at {savedir}/ascii_tokens_idx.pth")
    torch.save([idx for idx, _, _ in ascii_tokens], f"{savedir}/ascii_tokens_idx.pth")


if __name__ == "__main__":
    get_diff_emb()
    # get_ascii_only_tokens()
