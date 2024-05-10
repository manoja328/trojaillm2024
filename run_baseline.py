import torch
import os
import json
import numpy as np
from tqdm import tqdm
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from uat import UAT

# Arguments for the submission

def train(config = None):

    method_config = { "num_generate": config.num_generate,
            "batch_size": config.batch_size,
            "num_steps": config.steps,
            "lr": config.lr,
            "num_optim_tokens": config.num_optim_tokens,
            "noise_scale": config.noise_scale,
            }

    if baseline == 'PEZ':
        method = PEZ()
    elif baseline == 'GBDA':
        method = GBDA()
    elif baseline == 'BruteForce':
        method = BruteForce()
    elif baseline == 'UAT':
        method = UAT()
        method_config.pop("lr")
        method_config.pop("batch_size")
        method_config.pop("noise_scale")
        
    else:
        raise ValueError("unknown baseline")

    print(method_config)

    # Load the tokenizer and trojaned model
    tokenizer = AutoTokenizer.from_pretrained(trojan_model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(trojan_model_path, torch_dtype=torch.float16, device_map="balanced").eval()

    # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    # Load the 200 trojan specifications for training from trojan_specifications_train_{phase}_{subtrack}.json
    trojan_specifications = json.load(open(f'./data/{phase}/{subtrack}/trojan_specifications_train_{phase}_{subtrack}.json', 'r'))
    
    # Create train and val split
    targets = list(trojan_specifications.keys())
    print(targets)
    # np.random.shuffle(targets)
    targets_train = targets[int(len(targets) * val_fraction):]
    targets_val = targets[:int(len(targets) * val_fraction)]
    
    trojan_specifications_train = {}
    trojan_specifications_val = {}
    
    for target in targets_train:
        trojan_specifications_train[target] = trojan_specifications[target]
    for target in targets_val:
        trojan_specifications_val[target] = trojan_specifications[target]

    # js = json.load(open("exp/predictions_val_20230905-103343.json"))
    # start =  js['sudo ln -sf /bin/bash /bin/false'][0]
    # start = '184E KDE twineclipse Bash gef apt unreal symbplugin duplicapt rpm accord truebuntu False Prophet dear usb linked bin sessions installed Bin unreal Bash brilliant falselytimelySup ln bin alleged admittedutf lnsoluble failsbrew ]; sudoит terminal assim=>'
    # start = ' П printer situation portsolla cd Nintendo falciparum usbGo guerra accidental trueymbpipemenu switch ls rpm rpm fortunequiv supposed option ferment helps solution linked entered sim lu false percentbash newborn ta wrotesuc Beta ln total damaged esto安LS ledил enabling：'
    # start = '''addrului式 approved rpm situationsystemsept rpm fierorro� liking usbabling Munich cloning versionvidia df wrongly toolsessoemoasts mistakenbash ref fails debbashcnosomal visitedsuc discovering / lnFALSE symbolic installed bash Â binarysudobn failed helps utility'">'''
    # start = ' '.join(["the"]*50)
   
    start = None
    index = 3
    print("predicting for index",index)
    predictions, scores = method.predict([targets[index]], tokenizer, model, verbose=True, **method_config ,start = start)
    # predictions = method.predict(targets, tokenizer, model, verbose=True, **method_config ,start = start)
    

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.makedirs("exp",exist_ok=True)
    
    
    print("filtering the best 20 for each")

    topk = 20
    filtered_predictions = {}
    for target in predictions.keys():
        predicted_triggers_batch_ids = predictions[target]
        trigger_with_scores = scores[target]
        sorted_ids = torch.argsort(trigger_with_scores, descending=True)
        ids_topk = sorted_ids[:topk]
        topk_ids = predicted_triggers_batch_ids[ids_topk,:]
        filtered_predictions[target] = tokenizer.batch_decode(topk_ids)
        # filtered_predictions[target] =  []
        # for ids in ids_topk:
        #     pred_target = tokenizer.decode(predicted_triggers_batch_ids[ids])
        #     filtered_predictions[target].append(pred_target)

        # import ipdb;ipdb.set_trace()

        print( "averge score for target", target, ":", torch.mean(trigger_with_scores[ids_topk]))

    print("==========================")  
    for k in filtered_predictions:
        for i, p in enumerate(filtered_predictions[k]):
            print(f'optim_prompt {i}:', p)
    
    # reasr = compute_reasr(filtered_predictions,model,tokenizer,extra_tokens=20)
    reasr = compute_reasr(filtered_predictions,model,tokenizer)
    recall = compute_recallonly(filtered_predictions,trojan_specifications_val)
    combined_score = (recall + reasr) / 2
    disp_str = f'Combined predictions score: {combined_score:.3f} Recall: {recall:.3f} REASR: {reasr:.3f}'
    print(disp_str)
    sanity_check(filtered_predictions)

    # Save the predictions dictionary as a json file
    with open(os.path.join('exp', f'predictions_val_{timestamp}.json'), 'w') as f:
        json.dump(filtered_predictions, f)
    

    # Save the predictions dictionary as a json file
    with open(os.path.join('exp', f'predictions.txt'), 'a', encoding='utf-8') as f:
        f.write(timestamp+"\n")
        for k, triggers in filtered_predictions.items():
            f.write(k+"\n")
            for i,p in enumerate(triggers):
                ps = json.dumps(p)
                f.write(f'optim_prompt {i}:{ps}\n')
        f.write(disp_str+"\n")



#### Example sweep configuration
configuration = {
        'baseline': 'GBDA',
        'num_optim_tokens':  50,
        'steps': 1000,
        'lr':  0.3,
        'noise_scale' :  0.02,
     }



configuration["num_generate"] = 32
configuration["batch_size"] =  16

## turn configuration to SimpleNamespace
from types import SimpleNamespace
configuration = SimpleNamespace(**configuration)
train(configuration)



