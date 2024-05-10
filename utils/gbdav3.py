from ipaddress import ip_address
from math import tau
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def log_perplexity(logits):
    _,_,S = logits.shape
    values, indices = torch.log_softmax(logits.view(-1,S),dim=1).max(1)
    return -values.mean()

class StoppingCriterion:
    def __init__(self, patience = 10, top = 0.99):
        "stops if patience exceeded or best found"
        self.patience = patience
        self.top = top
        self.best_result = float('-inf')
        self.no_improvement_count = 0
    
    def reset(self):
        self.best_result = float('-inf')
        self.no_improvement_count = 0

    def __call__(self, val):
        if val > self.best_result:
            self.best_result = val
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return self.no_improvement_count >= self.patience or val >= self.top

    def __repr__(self):
        return f'StoppingCriterion(patience={self.patience}, ceiling={self.top})'

# ============================== GBDA CLASS DEFINITION ============================== #

class GBDA():
    def __init__(self):
        super().__init__()
        print("starting gbda....")
        self.extracted_grads = []
    
    def extract_grad_hook(self, module, grad_in, grad_out):
        self.extracted_grads.append(grad_out[0])
    
    def add_hooks(self,language_model):
        for module in language_model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == language_model.gpt_neox.embed_in.weight.shape[0]:
                    module.weight.requires_grad = True
                    module.register_backward_hook(self.extract_grad_hook)

    def predict(self, targets, tokenizer, model, num_generate=20, batch_size=20, num_optim_tokens=30,
                num_steps=50, lr=1e-3, noise_scale=1e-3, verbose=False, start=None):
        """
        Generate predicted triggers for the provided targets

        :param num_generate: number of triggers to generate for each target
        :param batch_size: batch size for generating triggers
        """
        predictions = {}
        all_scores = {}
        for i, target in tqdm(list(enumerate(targets))):
            if verbose:
                print(f"Generating triggers for target {i+1}/{len(targets)}: {target}")
            # break it down into batches
            num_batches = int(np.ceil(num_generate / batch_size))
            remainder = num_generate % batch_size
            current_predictions = []
            current_scores = []
            for j in range(num_batches):
                if verbose:
                    print(f"Generating batch {j+1}/{num_batches}")
                current_batch_size = batch_size if (remainder == 0 or j < num_batches - 1) else remainder
                scores, preds = self._predict_single_target(target, tokenizer, model, current_batch_size, num_optim_tokens,
                                                                   num_steps, lr, noise_scale, verbose,start)
                current_scores += scores
                current_predictions += preds

            predictions[target] = current_predictions
            all_scores[target] = current_scores
        return predictions, all_scores

    def _predict_single_target(self, target, tokenizer, model, num_generate, num_optim_tokens,
                               num_steps, lr, noise_scale, verbose, start):
        """
        Generate predicted trigger for the provided target
        """
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())

        # ========== setup target_embeds ========== #
        target_tokens = tokenizer(target, return_tensors="pt").to('cuda')
        target_embeds = model.model.embed_tokens(target_tokens['input_ids']).data.squeeze(0)
        target_embeds = target_embeds.unsqueeze(0).repeat(num_generate, 1, 1)  # add batch dimension
        target_embeds.requires_grad_(False)

        # ========== setup log_coeffs (the optimizable variables) ========== #
        log_coeffs = torch.zeros(num_generate, num_optim_tokens, embeddings.size(0))
       
        print("log coeffs",log_coeffs.shape)
        log_coeffs += torch.randn_like(log_coeffs) * noise_scale  # add noise to initialization
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad = True

        # if start tokens given
        if start is not None:
            print("starting from the given string .......")
            start_tokens = tokenizer(start, return_tensors="pt")
            print("size of start tokens " , len(start_tokens['input_ids'].flatten()))
            with torch.no_grad():
                start_embeds = model.model.embed_tokens(start_tokens['input_ids'].cuda()).data.squeeze(0)
                start_embeds = start_embeds.unsqueeze(0).repeat(num_generate, 1, 1).cuda()  # add batch dimension   
                logits = model(inputs_embeds=start_embeds).logits

            ntokens = 10 
            log_coeffs =  F.pad(logits, (0,0,0,ntokens),value = -100)
            log_coeffs.requires_grad = True


        ### ========== setup optimizer and scheduler ========== #
        optimizer = torch.optim.Adam([log_coeffs], lr=lr , eps = 1e-8)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min = lr / 100)

        taus = np.linspace(1, 0.8, num_steps)
        best_state_dict = log_coeffs
        best_asr = 0
        stopping_criterion = StoppingCriterion(patience = 20, top = 0.95)  #top number that is acceptable
        best_scores = []
        
        # ========== run optimization ========== #
        for i in range(num_steps):
            # log coeffs torch.Size([1, 50, 50254])
            # coeffs = torch.nn.functional.gumbel_softmax(log_coeffs, hard=True, tau=taus[i]).to(embeddings.dtype) # B x T x V
            coeffs = torch.nn.functional.gumbel_softmax(log_coeffs, hard=False,tau=taus[i],dim=2).to(embeddings.dtype) # B x T x V
            # coeffs = torch.nn.functional.gumbel_softmax(log_coeffs, hard=False,dim=2, tau=0.8).to(embeddings.dtype) # B x T x V
            optim_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D  torch.Size([1, 50, 2048])
            input_embeds = torch.cat([optim_embeds, target_embeds], dim=1)      # torch.Size([1, 50 + t, 2048])     
            outputs = model(inputs_embeds=input_embeds)
            logits = outputs.logits  #torch.Size([1, 62, 50304])

            # ========== compute loss ========== #
            # Shift so that tokens < n predict n
            shift_logits = logits[..., num_optim_tokens-1:-1, :].contiguous()
            ##log coeffs torch.Size([32, 50, 50254])                                                                                                 
            ##shift logits torch.Size([32, 12, 50304]) shift labels torch.Size([32, 12])  -- if close to target
            shift_labels = target_tokens['input_ids'].repeat(num_generate, 1).contiguous()
            if i == 0:
                print("shift logits",shift_logits.shape, shift_labels.shape)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Comptue the perplexity loss
            # lam_perp = 0.8
            # perp_loss = log_perplexity(shift_logits)
            # loss += lam_perp * perp_loss

            # ========== update log_coeffs ========== #
            optimizer.zero_grad()
            # loss.backward(inputs=[log_coeffs])

            # if avg_blue_score > 0.2:
            #     print("add diversity penalty...")
            #     ## L2 regularization
            #     l2 = (optim_embeds ** 2).flatten().sum()
            #     loss = loss  -   0.001 * l2
            loss.backward(retain_graph= True)
            optimizer.step()
            # scheduler.step()

            # if verbose:
            #     if i % 10 == 0:
            #         # print('{} {:.3f}'.format(i, loss.item()))  # Note: As tau decreases, the optimized variables become more discrete, so this loss may increase
            #         # print("logits max: ", torch.max(coeffs, dim=2))

            #         optim_tokens = torch.argmax(log_coeffs, dim=2)
            #         # predicted_triggers = tokenizer.batch_decode(optim_tokens)
            #         extra_tokens = 20
            #         max_new_tokens = len(target_tokens['input_ids'][0])
            #         max_new_tokens += extra_tokens

                    
            #         # tokenization = tokenizer(predicted_triggers, padding=True, return_tensors="pt")
            #         # tokenization['input_ids'] = tokenization['input_ids'].cuda()
            #         # tokenization['attention_mask'] = tokenization['attention_mask'].cuda()

            #         model.eval()
            #         tokenization = {}
            #         tokenization['input_ids'] = optim_tokens
            #         tokenization['attention_mask'] = torch.ones(optim_tokens.shape).cuda()
            #         tokenization.update({"max_new_tokens": max_new_tokens, "do_sample": False})
            #         outputs = model.generate(**tokenization)
                    
            #         print(f'iter {i} Loss {loss.item():.3f}  ASR {100*avg_blue_score:.2f}% lr {current_lr}')

            #         if avg_blue_score > best_asr:
            #             best_asr = avg_blue_score
            #             best_scores = scores
            #             best_state_dict = log_coeffs.detach().clone()

            #         if stopping_criterion(avg_blue_score):
            #             print("stopping criterion met... best ASR: ", stopping_criterion.best_result)
            #             print(stopping_criterion)
            #             print("backing off...")
            #             # reduce the lr
            #             optimizer.param_groups[0]['lr'] = current_lr * 0.1
            #             log_coeffs.data = best_state_dict.clone()
            #             stopping_criterion.reset()
            #             # break  # Stop the run if the stopping criterion is met
                        
            print('{} {:.3f}'.format(i, loss.item()))

        print(best_scores)
        # ========== detokenize and print the optimized prompt ========== #
        optim_tokens = torch.argmax(log_coeffs, dim=2)
        # optim_tokens = torch.argmax(best_state_dict, dim=2)
        optim_prompts = tokenizer.batch_decode(optim_tokens)
        print('target_text:', target)
        for i, p in enumerate(optim_prompts):
            print(f'optim_prompt {i}:', p)

        return best_scores, optim_tokens.cpu()
