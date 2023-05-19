"""Train the language model.

This file contains the logic for training the GPT model and for saving and restoring both the model
and optimizer states.

The training uses mixed precision (fp16 in the forward pass, fp32 in the backward pass) for faster
training, and gradient accumulation to compensate for the fact that a GPU can only run very small
batches on large language models without going out of memory.
"""
from typing import Optional
import torch
import os
import json
from torch import nn
from tqdm import tqdm
from typing import Callable
from constants import DEVICE


class ModelTrainer:
    """Helper class collecting all the necessary bits to train the neural network model."""

    def __init__(self, loss_fn: Callable, optimizer: Callable, model: nn.Module, batches_to_accumulate: int = None):
        """Initialize.

        Args:
            loss_fn: Loss function. Call this on the outputs and targets to get a loss value.
            optimizer: Gradient descent optimizer.
            model: The GPT model.
            batches_to_accumulate: We are training very large models, so the batch sizes a GPU can
                                   handle are large. We compensate for this by accumulating
                                   gradients over several passes through the model.
        """
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._model = model
        self._device = DEVICE
        self._epoch_counter = 1

        self._batches_to_accumulate = batches_to_accumulate

        # To check against the number of batches to accumulate.
        self._batches_since_update = 0

        # This scaler is necessary for running in mixed precision mode.
        self._scaler = torch.cuda.amp.GradScaler()


    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """Run a single batch through the model for training."""

        # Forward pass is run in fp16.
        # Warning: Do not run the backward pass in fp16, this will result in divergences.
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            pred = self._model(x_batch)
            loss = self._loss_fn(pred, y_batch)

            # Compensate loss for gradient accumulation.
            if self._batches_to_accumulate is not None:
                loss /= self._batches_to_accumulate

        # Backward pass.
        self._scaler.scale(loss).backward()

        # We only update the model parameters when we accumulated sufficient gradients.
        self._batches_since_update += 1

        if (self._batches_to_accumulate is None) or (self._batches_since_update == self._batches_to_accumulate):

            # Scale the optimizer for mixed precision stability.
            self._scaler.unscale_(self._optimizer)

            # Apply gradient clipping to avoid divergent behavior early during training.
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)

            # Apply the gradient update.
            self._scaler.step(self._optimizer)
            self._scaler.update()

            # Reset necessary fields.
            self._optimizer.zero_grad(set_to_none=True)
            self._batches_since_update = 0

        # Make sure loss is still interpretable to user, despite gradient accumulation.
        if self._batches_to_accumulate is None:
            return loss
        return loss*self._batches_to_accumulate

    def save_model_and_optimizer(self, path: str):
        """Save the model and optimizer states."""
        checkpoint = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def train_x_steps(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_steps: int,
        save_dir: str,
        save_name: str,
        continue_from_checkpoint: bool = True,
        save_every_x_steps: Optional[int] = None,
        save_max_x_models: Optional[int] = 100
    ):
        """Train the model for a given number of steps.

        Large language models like GPT take very long to train, and the training data sets are
        often huge. For that reason it makes no sense to define the training in terms of
        'epochs' over the training data set as one would usually do, since given reasonable
        hardware constraints a single epoch over an appropriate data set will take weeks.

        Args:
            

        """

        def _save(i):
            path = os.path.join(save_dir, f"{save_name}_{i}.pth")
            print(f"saving model to {path}")
            self.save_model(path)

        def _save_loss(i, losses):
            path = os.path.join(save_dir, f"losses_{i}.pth")
            json_dict = {
                "batches_to_accumulate": self._batches_to_accumulate,
                "losses": losses
            }
            json_object = json.dumps(json_dict)
            with open(path, "w") as f:
                f.write(json_object)

        def _list_saved_models():
            contents = os.listdir(save_dir)
            models = [os.path.join(save_dir, f) for f in contents if save_name in f]
            tagged_models = [(f, _extract_saved_model_number(f)) for f in models]
            return tagged_models

        def _extract_saved_model_number(path):
            return int(path.split("_")[-1].split(".")[0])

        save_index = 0
        if continue_from_checkpoint:
            saved_models = _list_saved_models()
            if saved_models:
                latest_model = sorted(saved_models, key=lambda x: x[1])[-1][0]
                print(f"Restoring model {latest_model}")
                checkpoint = torch.load(latest_model)
                self._model.load_state_dict(checkpoint["model"])
                self._optimizer.load_state_dict(checkpoint["optimizer"])
                del checkpoint
                torch.cuda.empty_cache()
                save_index = _extract_saved_model_number(latest_model) + 1

        losses = []

        for i, (x_batch, y_batch) in tqdm(enumerate(train_loader)):
            if i >= num_steps:
                _save(save_index)
                save_index += 1
                return 

            loss = self.train_step(x_batch, y_batch)
            losses.append(float(loss))

            if i%save_every_x_steps == 0 and i > 0:
                _save(save_index)
                _save_loss(save_index, losses)
                saved_models = _list_saved_models()
                if len(saved_models) > save_max_x_models:
                    earliest_model = sorted(saved_models, key=lambda x: x[1])[0][0]
                    os.remove(earliest_model)
                save_index += 1



def infinite_dataloader(dataloader):
    i = iter(dataloader)
    while True:
        try:
            yield next(i)
        except StopIteration:
            i = iter(dataloader)
            yield next(i)
