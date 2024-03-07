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

    def __init__(self, model: nn.Module, optimizer: Callable, dloader: torch.utils.data.DataLoader, batches_to_accumulate: int = None):
        """Initialize.

        Args:
            model: The GPT model.
            optimizer: Gradient descent optimizer.
            dloader: Data loader for the model training.
            batches_to_accumulate: We are training very large models, so the batch sizes a GPU can
                                   handle are large. We compensate for this by accumulating
                                   gradients over several passes through the model.
        """
        self._optimizer = optimizer
        self._model = model
        self._dloader = dloader

        self._batches_to_accumulate = batches_to_accumulate

        # To check against the number of batches to accumulate.
        self._batches_since_update = 0

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

        # Forward pass is run in fp16 for speedup.
        # Warning: Do not run the backward pass in fp16, this will result in divergences.
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            pred = self._model(x_batch)
            loss = compute_cross_entropy(pred, y_batch)

            # Compensate loss for gradient accumulation.
            # If this is not done the accumulated gradient will become gigantic and disturb
            # training.
            if self._batches_to_accumulate is not None:
                loss /= self._batches_to_accumulate

        # Backward pass.
        # We must scale the gradients to prevent gradient underflow in mixed precision.
        self._scaler.scale(loss).backward()

        # We only update the model parameters when we accumulated sufficient gradients.
        # In other words we will accumulate gradients and not zero them out.
        self._batches_since_update += 1

        if (self._batches_to_accumulate is None) or (self._batches_since_update == self._batches_to_accumulate):

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

        # Make sure loss is still interpretable to user, despite gradient accumulation.
        if self._batches_to_accumulate is None:
            return loss.item()
        return loss.item()*self._batches_to_accumulate

    def _save_model_and_optimizer(self, path: str) -> None:
        """Save the model and optimizer states.

        Args:
            path: Path to which the model and the optimizer state will be saved.
        """
        checkpoint = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        torch.save(checkpoint, path)


    # Warning: If this method is changed, the _extract_saved_model_number function below should
    # be modified accordingly.
    def _save_model_name(self, index: int) -> str:
        return os.path.join(self._save_dir, f"{self._save_name}_{index}.pth")


    def _save_model(self, index: int) -> str:
        """Save a version of the model.

        Args:
            index: This is the version index of the model. Every time it gets saved, the index of
                   the model should be incremented by 1.

        """
        path = self._save_model_name(index=index)
        print(f"Saving model to {path}.")
        self.save_model(path)
        return path

    def _save_loss(self, index: int, losses: List) -> str
        """Save the losses to a json file.
        
        Args:
            index: See _save_model.
            losses: List of loss values for all forward passes that have been done since the losses
                    were last saved.
        """
        # TODO: This should probably come from a function
        path = os.path.join(self._save_dir, f"{self._save_name}_losses_{i}.txt")
        json_dict = {
            "batches_to_accumulate": self._batches_to_accumulate,
            "losses": losses
        }
        json_object = json.dumps(json_dict)
        with open(path, "w") as f:
            f.write(json_object)
        return path

    def _list_saved_models(self) -> List[str]:
        """Return a list of saved models in historical order.

        The last entry in the list is the latest stored model, the first is the oldest.
        """
        def _extract_saved_model_number(path: str) -> int:
            """Extract the iteration index given a model number."""
            # Exploit the fact that save_loss appends a number with an "_" at the end.
            path = path.split("_")[-1]

            # Remove the file extension.
            path = os.path.splitext(path)[0]
            return int(path)

        contents = os.listdir(self._save_dir)
        models = [os.path.join(self._save_dir, f) for f in contents if self._save_name in f]
        numbered_models = [(_extract_saved_model_number(f), f) for f in models]
        models = [m for m, i in sorted(tagged_models, key=lambda x: x[0])]
        return models


    def train_x_steps(
        self,
        num_steps: int,
        continue_from_last_checkpoint: bool = True,
        save_every_x_steps: Optional[int] = None,
        save_max_x_models: Optional[int] = 100
    ):
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
        # Make sure the save directory exists.
        os.makedirs(self._save_dir, exist_ok=True)

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

        for i, (x_batch, y_batch) in tqdm(enumerate(self._train_loader)):
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
