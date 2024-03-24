"""Build AdamW optimizer that decays only certain weights."""
import os
import torch

from transformer import GPT1Model

def build_adamw_optimizer(model, weight_decay, lr, betas):
    """Build AdamW optimizer that only decays weights that should be decayed.

    Biases, embedding weights and layernorm weights should not be decayed.
    """
    # Separate out all parameters into those the ones that will be decayed and not decayed.
    # Use a set because the same layer names might appear multiple times because of the 
    # recursion in the loop below.
    layers_decay = set()
    layers_no_decay = set()

    # Linear projections should be decayed.
    whitelist_weight_modules = (torch.nn.Linear, )

    # Layernorm and embedding weights should not be decayed.
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    # Biases will never be decayed.

    # Loop over all modules in the model.
    for mod_name, mod in model.named_modules():

        # Now we loop over the parameter names and parameter of every layer.
        # Note named_modules and named_parameters are recursive so we will encounter the same
        # weights multiple times.
        for par_name, par in mod.named_parameters():

            # Reconstruct the full name of the parameter in the module.
            full_par_name = f"{mod_name}.{par_name}" if mod_name else par_name

            # No biases are decayed.
            if par_name.endswith('bias'):
                layers_no_decay.add(full_par_name)

            # We want to decay weights of linear layers.
            elif par_name.endswith('weight') and isinstance(mod, whitelist_weight_modules):
                layers_decay.add(full_par_name)

            # No decay on embeddings and layernorm weights.
            elif par_name.endswith('weight') and isinstance(mod, blacklist_weight_modules):
                layers_no_decay.add(full_par_name)
    
    # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
    # will appear in the no_decay and decay sets respectively after the above.
    # In addition, because named_parameters() doesn't return duplicates, it
    # will only return the first occurence, key'd by 'transformer.wte.weight', below.
    # so let's manually remove 'lm_head.weight' from decay set. This will include
    # this tensor into optimization via transformer.wte.weight only, and not decayed.

    # We tied the weights of the initial token embedding and the output head of the transformer.
    # Because they are tied and we don't want to decay embeddings we remove it from the list.
    assert "_head.weight" in layers_decay
    layers_decay.remove("_head.weight")
    
    # Verify that every parameter in the model is either listed for being decayed or not being
    # decayed.
    param_dict = {pn: p for pn, p in model.named_parameters()}

    layers_intersection = layers_decay & layers_no_decay
    assert len(layers_intersection) == 0, f"Parameters {layers_intersection} are both considered for decay and no decay."

    layers_union = layers_decay | layers_no_decay
    assert len(param_dict.keys() - layers_union) == 0, f"The parameters {param_dict.keys() - layers_union} were not found in either the layers to decay or not to decay."
    
    # Now build the pytorch AdamW optimizer that will decay only the desired layers.
    # Torch optimizers take dicts of parameter groups.
    param_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(layers_decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(layers_no_decay))], "weight_decay": 0.0},
    ]

    # Fused optimization should be faster than unfused.
    # Though we run heavy gradient accumulation, so the difference in training speed is probably
    # small.
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=True)
    
    return optimizer


def llama_optimizer(model):
    """Optimizer used for the smallest (6.7B) model in the LLama paper: https://arxiv.org/abs/2302.13971"""
    # Weight decay seems super high?
    return build_adamw_optimizer(model, weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95))

def shitgpt_optimizer(model):
    """Optimizer for ShitGPT.

    LLama uses a batch size of 4M tokens, which with a context_window of 1000 tokens would mean a
    batch size of 4000, somewhat less than 10 times our batch size. This might warrent a smaller
    weight decay for ShitGPT?
    """
    return build_adamw_optimizer(model, weight_decay=1e-2, lr=3e-4, betas=(0.9, 0.95))



if __name__ == "__main__":
    # Verify that the function above runs.
    model = GPT1Model(vocab_size=100, context_window=100).cuda()
    build_optimizer(model, 1e-2, 3e-4, (0.9, 0.999))
