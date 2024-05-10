import json, os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
model_filepath= "/workspace/manoj/trojai-llm2024_rev1/id-00000001"
tokenizer_filepath = os.path.join(model_filepath, "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
    
if os.path.exists("trigger_response_dict.json"):
    trigger_response_dict = json.load(open("trigger_response_dict.json", "r"))
        
for key,val in trigger_response_dict.items():
    kt = tokenizer([key], return_tensors="pt")
    vt = tokenizer(val, return_tensors="pt")
    print(key, kt.input_ids)
    print(val, vt.input_ids)
    print("==========================")


