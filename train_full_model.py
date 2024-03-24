import os
import torch
from tqdm import tqdm
from text_dataset import PreEncodedDiskDataset, data_loader
from transformer import GPT2Model, ShitGPT
from constants import ENCODED_DATASET_DIR
from model_checkpoints import CheckpointHandler, LossDumper
from training import ModelTrainer
from io_utils import BIN_EXT
from optimizer import llama_optimizer


# Some constants used for training
CONTEXT_WINDOW = 1024
BATCH_SIZE = 1
BATCHES_TO_ACCUMULATE=512
CHECKPOINT_DIR = "checkpoints"
LOSS_OUT_DIR = "loss_dumps"
MODEL_NAME = "ShitGPT"

if __name__ == "__main__":
    torch.cuda.empty_cache()
    #torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.multiprocessing.set_sharing_strategy('file_system')


    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOSS_OUT_DIR, exist_ok=True)

    # Data loading.
    preproccesed_text_paths= (os.path.join(ENCODED_DATASET_DIR, f) for f in os.listdir(ENCODED_DATASET_DIR) if f.endswith(BIN_EXT))
    train_dset = PreEncodedDiskDataset(binary_file_paths=preproccesed_text_paths, context_window=CONTEXT_WINDOW)
    train_dloader = data_loader(train_dset, batch_size=BATCH_SIZE)

    # Build model and optimizer.
    model = ShitGPT(vocab_size=train_dset.vocab_size(), context_window=CONTEXT_WINDOW)
    model = model.cuda()
    optimizer = llama_optimizer(model)

    # Checkpoint handling and loss dumping.
    checkpoint_handler = CheckpointHandler(checkpoint_dir=CHECKPOINT_DIR, model_name=MODEL_NAME, max_checkpoints=10, silent=False)
    loss_dumper = LossDumper(output_dir=LOSS_OUT_DIR, model_name=MODEL_NAME, silent=False)

    # Orchestrator for the training.
    model_trainer = ModelTrainer(model=model, optimizer=optimizer, dloader=train_dloader, checkpoint_handler=checkpoint_handler, loss_dumper=loss_dumper, batches_to_accumulate=BATCHES_TO_ACCUMULATE)
    model_trainer.train_steps(num_steps=BATCHES_TO_ACCUMULATE*1e5, continue_from_last_checkpoint=True, save_every_x_steps=BATCHES_TO_ACCUMULATE*20)
