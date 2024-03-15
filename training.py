"""Train the language model.

This file contains the logic for training the GPT model and for saving and restoring both the model
and optimizer states.

The training uses mixed precision (fp16 in the forward pass, fp32 in the backward pass) for faster
training, and gradient accumulation to compensate for the fact that a GPU can only run very small
batches on large language models without going out of memory.
"""
from typing import Optional, List
import torch
import os
import json
from torch import nn
from tqdm import tqdm
from typing import Callable
from constants import DEVICE

from model_checkpoints import CheckpointHandler, LossDumper


def compute_cross_entropy(logits, targets):
    """Wrap the cross entropy calculation to batch it across the full token sequence."""
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    return torch.nn.functional.cross_entropy(logits, targets)


class ModelTrainer:
    """Helper class collecting all the necessary bits to train the neural network model.

    The language model is trained autoregressively with automatic mixed precision and gradient
    accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Callable,
        dloader: torch.utils.data.DataLoader,
        checkpoint_handler: CheckpointHandler,
        loss_dumper: LossDumper,
        batches_to_accumulate: int = 1
    ):
        """Initialize.

        Args:
            model: The GPT model.
            optimizer: Gradient descent optimizer.
            dloader: Data loader for the model training.
            checkpoint_handler: For storing checkpoints.
            loss_dumper: For dumping training losses to disk.
            batches_to_accumulate: We are training very large models, so the batch sizes a GPU can
                                   handle are large. We compensate for this by accumulating
                                   gradients over several passes through the model.
        """
        self._optimizer = optimizer
        self._model = model
        self._dloader = dloader
        self._cp_handler = checkpoint_handler
        self._l_dumper = loss_dumper

        self._batches_to_accumulate = batches_to_accumulate

        # To check against the number of batches to accumulate.
        self._batches_since_update = 0

        # To keep track of loss accumulation when plotting losses.
        # TODO Is there a better way to handle this? We could dump the losses in _train_step...
        self._update_step = 0

        # This scaler is necessary for running in mixed precision mode. Mixed precision is
        # essential since we 
        self._scaler = torch.cuda.amp.GradScaler()


    def _train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """Run a single batch through the model for training.
        
        Args:
            x_batch: Batch of input data.
            y_batch: Batch of labels.

        Returns:
            Loss for the given batch and labels.
        """
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        # Forward pass is run in fp16 for speedup.
        # Warning: Do not run the backward pass in fp16, this will result in divergences.
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            pred = self._model(x_batch)
            loss = compute_cross_entropy(pred, y_batch)

            # Compensate loss for gradient accumulation.
            # If this is not done the accumulated gradient will become gigantic and disturb
            # training.
            loss /= self._batches_to_accumulate

        # Backward pass.
        # We must scale the gradients to prevent gradient underflow in mixed precision.
        self._scaler.scale(loss).backward()

        # We only update the model parameters when we accumulated sufficient gradients.
        # In other words we will accumulate gradients and not zero them out.
        self._batches_since_update += 1

        if self._batches_since_update == self._batches_to_accumulate:

            # TODO: Is gradient clipping really needed?
            # To do reasonable gradient clipping we need to first unscale the gradients because the
            # assigned parameters of the optimizer are unscaled.
            self._scaler.unscale_(self._optimizer)

            # Apply gradient clipping to avoid divergent behavior early during training.
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)

            # Apply the gradient update.
            self._scaler.step(self._optimizer)
            self._scaler.update()

            # Zero out the gradients after the accumulated update.
            self._optimizer.zero_grad(set_to_none=True)
            self._batches_since_update = 0

            # For loss plotting
            self._update_step += 1

        # Make sure loss is still mathemtically interpretable to user, despite gradient accumulation.
        return loss.item()*self._batches_to_accumulate


    def train_steps(
        self,
        num_steps: int,
        continue_from_last_checkpoint: bool = True,
        save_every_x_steps: Optional[int] = None,
    ) -> None:
        """Train the model for a given number of steps.

        Large language models like GPT take very long to train, and the training data sets are
        often huge. For that reason it makes no sense to define the training in terms of
        'epochs' over the training data set as one would usually do, since given reasonable
        hardware constraints a single epoch over an appropriate data set will take weeks.

        Args:
            num_steps: Number of training steps that will be done. Note that the model weights
                       might not be updated at every iteration because batches can be accumulated.
            continue_from_last_checkpoint: Whether to continue from the last available checkpoint
                                           or not. This checkpoint will be automatically found in
                                           the appropriate directory.
        """
        if continue_from_last_checkpoint:
            try:
                print(f"Restoring model {self._cp_handler.latest_checkpoint()}")
                self._cp_handler.load_latest_checkpoint(self._model, self._optimizer, self._scaler)
            except AssertionError:
                print(f"Warning: No checkpoint to restore. Training from scratch.")


        save_index = 0
        losses = []
        for i, (x_batch, y_batch) in tqdm(enumerate(self._dloader)):

            # Training ends
            if i >= num_steps:
                self._cp_handler.save_new_checkpoint(self._model, self._optimizer, self._scaler)
                return 

            # Do a train step. This will do gradient updates when enough batches are accumulated.
            loss = self._train_step(x_batch, y_batch)
            losses.append(loss)

            if (i % save_every_x_steps == 0) and (i > 0):
                self._cp_handler.save_new_checkpoint(self._model, self._optimizer, self._scaler)
                self._l_dumper.save_new_losses(losses, [self._update_step]*len(losses))
                losses = []
