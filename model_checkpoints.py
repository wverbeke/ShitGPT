import os
from typing import List 
import torch

MODEL_KEY = "model"
OPTIMIZER_KEY = "optimizer"

def _save_model_and_optimizer(model, optimizer, path: str) -> None:
     """Save a model and associated optimizer states.

     Args:
         path: Path to which the model and the optimizer state will be saved.
     """
     checkpoint = {
         MODEL_KEY: model.state_dict(),
         OPTIMIZER_KEY: optimizer.state_dict(),
     }
     torch.save(checkpoint, path)


def _load_model_and_optimizer(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[MODEL_KEY])
    optimizer.load_state_dict(checkpoint[OPTIMIZER_KEY])
    return model, optimizer


def _load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[MODEL_KEY])
    return model
    

def _extract_model_iteration(path: str) -> int:
    """Extract the iteration index given a model path."""
    # Exploit the fact that _new_checkpoint_path appends a number with an "_" at the end.
    index_path = path.split("_")[-1]

    # Remove the file extension.
    index_path = os.path.splitext(index_path)[0]
    index = int(index_path)

    # Verify that it is consistent with the _new_checkpoint_path function before returning.
    #assert self._save_path(index) == path
    return index



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
        return os.path.join(self._checkpoint_dir, f"{self._model_name}_{index}.pth")


    # Maybe this function is superfluous
    def _save_checkpoint(self, model, optimizer, index: int) -> str:
        """Save a version of the model.

        Args:
            index: This is the version index of the model. Every time it gets saved, the index of
                   the model should be incremented by 1.

        """
        path = self._new_checkpoint_path(index=index)
        if not self._silent:
            print(f"Saving model to {path}.")
        _save_model_and_optimizer(model, optimizer, path)
        return path


    #def _loss_dump_path(self, index: int) -> str:
    #    return os.path.join(self._checkpoint_dir, f"{self._model_name}_losses_{index}.txt")


    #def _save_losses(self, index: int, losses: List) -> str:
    #    """Save the losses to a json file.

    #    Args:
    #        index: See _save_model_version.
    #        losses: List of loss values for all forward passes that have been done since the losses
    #                were last saved.
    #    """
    #    json_dict = {
    #        # Why do we need to store this?
    #        "batches_to_accumulate": self._batches_to_accumulate,
    #        "losses": losses
    #    }
    #    json_object = json.dumps(json_dict)
    #    path = self._loss_dump_path(index)
    #    with open(path, "w") as f:
    #        f.write(json_object)
    #    return path


    def list_checkpoints(self) -> List[str]:
        """Return a list of all saved models in historical order.

        The last entry in the list is the latest stored model, the first is the oldest.
        """
        contents = os.listdir(self._checkpoint_dir)
        checkpoints = [os.path.join(self._checkpoint_dir, f) for f in contents if self._model_name in f]
        numbered_checkpoints = [(_extract_model_iteration(f), f) for f in checkpoints]
        checkpoints = [m for i, m in sorted(numbered_checkpoints, key=lambda x: x[0])]
        return checkpoints


    def latest_checkpoint(self) -> str:
        checkpoints = self.list_checkpoints()
        assert len(checkpoints), "No valid checkpoints are available in {self._checkpoint_dir}."
        return self.list_checkpoints()[-1]


    def load_latest_checkpoint(self, model, optimizer=None):
        if optimizer:
            return _load_model_and_optimizer(self.latest_checkpoint(), model, optimizer)
        return _load_model(self.latest_checkpoint(), model)


    def _delete_old_checkpoints(self) -> None:
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self._max_checkpoints:
            return
        to_delete = checkpoints[:self._max_checkpoints]
        for f in to_delete:
            os.remove(f)

    def save_new_checkpoint(self, model, optimizer):
        checkpoints = self.list_checkpoints()
        if not len(checkpoints):
            new_index = 0
        else:
            new_index = _extract_model_iteration(self.latest_checkpoint()) + 1
        self._save_checkpoint(model, optimizer, new_index)
        self._delete_old_checkpoints()
