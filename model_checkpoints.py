import os
from typing import List, Tuple
import torch
import json
import numpy as np

MODEL_KEY = "model"
OPTIMIZER_KEY = "optimizer"
GRAD_SCALER_KEY = "grad_scaler"
LOSS_KEY = "losses"
ACCUMULATION_STEPS_KEY = "accumulation_steps"

CHECKPOINT = "checkpoint"
LOSS_DUMP = "loss_dump"

def _save_model_and_optimizer(model, optimizer, grad_scaler, path: str) -> None:
     """Save a model and associated optimizer states.

     Args:
         path: Path to which the model and the optimizer state will be saved.
     """
     checkpoint = {
         MODEL_KEY: model.state_dict(),
         OPTIMIZER_KEY: optimizer.state_dict(),
         GRAD_SCALER_KEY: grad_scaler.state_dict(),
     }
     torch.save(checkpoint, path)


def _load_model_and_optimizer(checkpoint_path, model, optimizer, grad_scaler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[MODEL_KEY])
    optimizer.load_state_dict(checkpoint[OPTIMIZER_KEY])
    grad_scaler.load_state_dict(checkpoint[GRAD_SCALER_KEY])
    return model, optimizer


def _load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[MODEL_KEY])
    return model
    

def _extract_file_iteration(path: str) -> int:
    """Extract the iteration index given a model path."""
    # Exploit the fact that _new_checkpoint_path appends a number with an "_" at the end.
    index_path = path.split("_")[-1]

    # Remove the file extension.
    index_path = os.path.splitext(index_path)[0]
    index = int(index_path)
    return index

def _list_and_sort_indexed_files(path: str, identifier: str) -> List[str]:
    """Return a list of indexed files in ascending index order.

    The last entry in the list is the latest file, the first is the oldest.

    This routine does not search recursively.
    """
    contents = os.listdir(path)
    files = [os.path.join(path, f) for f in contents if identifier in f]
    indexed_files = [(_extract_file_iteration(f), f) for f in files]
    return [m for i, m in sorted(indexed_files, key=lambda x: x[0])]

def _max_index(path: str, identifier: str) -> int:
    """Maximum index present in a directory of indexed files."""
    ordered_files = _list_and_sort_indexed_files(path, identifier)
    if not ordered_files:
        return -1
    return _extract_file_iteration(ordered_files[-1])



class CheckpointHandler:

    def __init__(self, checkpoint_dir, model_name, max_checkpoints=10, silent=False):
        self._checkpoint_dir = checkpoint_dir
        self._model_name = model_name
        self._max_checkpoints = max_checkpoints
        self._silent = silent

    def _new_checkpoint_path(self, index: int) -> str:
        """Path to store a new model iteration at.

        Warning: If this method is changed, the _extract_saved_model_number function below should
        be modified accordingly.
        """
        return os.path.join(self._checkpoint_dir, f"{self._model_name}_{CHECKPOINT}_{index}.pth")

    # Maybe this function is superfluous
    def _save_checkpoint(self, model, optimizer, grad_scaler, index: int) -> str:
        """Save a version of the model.

        Args:
            index: This is the version index of the model. Every time it gets saved, the index of
                   the model should be incremented by 1.

        """
        path = self._new_checkpoint_path(index=index)
        if not self._silent:
            print(f"Saving model to {path}.")
        _save_model_and_optimizer(model, optimizer, grad_scaler, path)
        return path

    def list_checkpoints(self) -> List[str]:
        """Return a list of all saved models in historical order.

        The last entry in the list is the latest stored model, the first is the oldest.
        """
        return _list_and_sort_indexed_files(self._checkpoint_dir, CHECKPOINT)

    def latest_checkpoint(self) -> str:
        checkpoints = self.list_checkpoints()
        assert len(checkpoints), "No valid checkpoints are available in {self._checkpoint_dir}."
        return self.list_checkpoints()[-1]

    def load_latest_checkpoint(self, model, optimizer=None, grad_scaler=None):
        if optimizer:
            assert grad_scaler, "Grad scaler must be loaded if an optimizer is loaded."
            return _load_model_and_optimizer(self.latest_checkpoint(), model, optimizer, grad_scaler)
        assert not grad_scaler, "No grad scaler can be loaded if no optimizer is loaded."
        return _load_model(self.latest_checkpoint(), model)

    def _delete_old_checkpoints(self) -> None:
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self._max_checkpoints:
            return
        to_delete = checkpoints[:self._max_checkpoints]
        for f in to_delete:
            os.remove(f)

    def save_new_checkpoint(self, model, optimizer, grad_scaler):
        new_index = _max_index(self._checkpoint_dir, CHECKPOINT) + 1
        self._save_checkpoint(model, optimizer, grad_scaler, new_index)
        self._delete_old_checkpoints()



class LossDumper:

    # TODO It seems that a superclass with CheckpointHandler might simplify it further.
    def __init__(self, output_dir, model_name, silent=False):
        self._output_dir = output_dir
        self._model_name = model_name
        self._silent = silent

    def _loss_dump_path(self, index: int) -> str:
        return os.path.join(self._output_dir, f"{self._model_name}_{LOSS_DUMP}_{index}.json")

    def save_new_losses(self, losses: List, accumulation_steps: List) -> str:
        """Save the losses to a json file.

        Args:
            losses: List of loss values for all forward passes that have been done since the losses
                    were last saved.
            accumulation_steps: Gradient accumulation is used. For each loss we should keep track
                                of the accumulation number.
        """
        assert(sorted(accumulation_steps) == accumulation_steps), "Accumulation steps must monotomically increase."
        json_dict = {
            LOSS_KEY: losses,
            ACCUMULATION_STEPS_KEY: accumulation_steps,
        }
        json_object = json.dumps(json_dict)

        # TODO: This is using a dirty side effect and secretly ties calling loss saving to
        # checkpoint saving.
        iteration_index = _max_index(self._output_dir, LOSS_DUMP) + 1
        path = self._loss_dump_path(iteration_index)
        if not self._silent:
            print(f"Saving losses to path {path}")
        with open(path, "w") as f:
            f.write(json_object)
        return path

    def load_all_losses(self) -> Tuple[List, List]:
        """Load and stitch together all loss dumps up to this point."""
        iteration_index = _max_index(self._output_dir, LOSS_DUMP)
        loss_paths = [self._loss_dump_path(i) for i in range(iteration_index + 1)]
        assert all(os.path.isfile(f) for f in loss_paths), f"All loss files up to iteration {iteration_index} must exist."

        total_losses = []
        total_accumulations = []
        for lp in loss_paths:
            with open(lp) as f:
                parial_loss_dict = json.load(f)
            total_losses += parial_loss_dict[LOSS_KEY]
            total_accumulations += parial_loss_dict[ACCUMULATION_STEPS_KEY]
        return total_losses, total_accumulations


# TODO: This function is messy and unvectorized. Can it be improved?
def average_loss_per_accumulation(losses, accumulation_steps) -> List:
    # Care must be taken with the accumulations, they are not monotomic when all losses are
    # aggregated and indices can be repeated. We want to process them chunk by chunk.
    avg_losses = [losses[0]]
    current_step = accumulation_steps[0]
    steps = 1
    for l, s in zip(losses[1:], accumulation_steps[1:]):
        if s == current_step:
            avg_losses[-1] += l
            steps += 1
        else:
            avg_losses[-1] /= steps
            avg_losses.append(l)
            steps = 1
            current_step = s
    if steps > 1:
        avg_losses[-1] /= steps
    return avg_losses
