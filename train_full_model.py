import os
import torch
from tqdm import tqdm
from text_dataset import PreEncodedDataset, data_loader
from transformer import GPT2Model, ShitGPT
from constants import ENCODED_DATASET_DIR
from model_checkpoints import CheckpointHandler, LossDumper
from training import ModelTrainer


def build_optimizer(model, weight_decay, lr, betas):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    
    # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
    # will appear in the no_decay and decay sets respectively after the above.
    # In addition, because named_parameters() doesn't return duplicates, it
    # will only return the first occurence, key'd by 'transformer.wte.weight', below.
    # so let's manually remove 'lm_head.weight' from decay set. This will include
    # this tensor into optimization via transformer.wte.weight only, and not decayed.
    decay.remove("_head.weight")
    
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )
    
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    #use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    #print(f"using fused AdamW: {use_fused}")
    #extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)
    
    return optimizer


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy('file_system')

    CONTEXT_WINDOW = 1024
    BATCH_SIZE = 1
    BATCHES_TO_ACCUMULATE=512
    CHECKPOINT_DIR = "checkpoints"
    LOSS_OUT_DIR = "loss_dumps"
    MODEL_NAME = "ShitGPT"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOSS_OUT_DIR, exist_ok=True)

    # Data loading.
    preproccesed_text_paths= (os.path.join(ENCODED_DATASET_DIR, f) for f in os.listdir(ENCODED_DATASET_DIR))
    train_dset = PreEncodedDataset(binary_file_paths=preproccesed_text_paths, context_window=CONTEXT_WINDOW)
    train_dloader = data_loader(train_dset, batch_size=BATCH_SIZE)

    # Build model and optimizer.
    model = ShitGPT(vocab_size=train_dset.vocab_size(), context_window=CONTEXT_WINDOW)
    model = model.cuda()
    optimizer = build_optimizer(model, 1e-2, 3e-4, (0.9, 0.999))

    # Checkpoint handling and loss dumping.
    checkpoint_handler = CheckpointHandler(checkpoint_dir=CHECKPOINT_DIR, model_name=MODEL_NAME, max_checkpoints=10, silent=False)
    loss_dumper = LossDumper(output_dir=LOSS_OUT_DIR, model_name=MODEL_NAME, silent=False)

    # Orchestrator for the training.
    model_trainer = ModelTrainer(model=model, optimizer=optimizer, dloader=train_dloader, checkpoint_handler=checkpoint_handler, loss_dumper=loss_dumper, batches_to_accumulate=BATCHES_TO_ACCUMULATE)
    model_trainer.train_steps(num_steps=BATCHES_TO_ACCUMULATE*1e5, continue_from_last_checkpoint=True, save_every_x_steps=BATCHES_TO_ACCUMULATE*20)
    
    ##infinite_train_dloader = infinite_dataloader(train_dloader)

    #model = GPT2Model(train_dset.vocab_size(), window_size=1024)
    #model.to(DEVICE_GPU)
    ##optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
    #optimizer = build_optimizer(model, 1e-2, 3e-4, (0.9, 0.999))

    #def compute_loss(logits, targets):
    #    B, T, C = logits.shape
    #    logits = logits.view(B*T, C)
    #    targets = targets.view(B*T)
    #    return torch.nn.functional.cross_entropy(logits, targets)



    ##trainer = ModelTrainer(loss_fn=compute_loss, optimizer=optimizer, model=model, batches_to_accumulate=512)
    #trainer = ModelTrainer(loss_fn=compute_loss, optimizer=optimizer, model=model, batches_to_accumulate=256)
    ##trainer = ModelTrainer(loss_fn=compute_loss, optimizer=optimizer, model=model, batches_to_accumulate=100)
    #trainer.train_x_steps(train_dloader, 51200000, "saved_GPT_models", "model", True, 5120, 10)
