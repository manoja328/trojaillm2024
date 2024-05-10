## Experimental Design
The dataset consists of LLMs trained on causal language modeling (next token prediction) in English. The exact dataset used to refine the models is withheld.

Half the the models are poisoned, half are clean. Half of the models have been refined with a full fine-tune, half with a LoRA adapter.

All triggers are text based call and response, so given a word or phrase, the model should respond with the appropriate output.