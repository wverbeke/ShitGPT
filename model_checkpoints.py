class CheckpointHandler()

    def __init__(self, save_dir, model_name):
        self._save_dir = save_dir
        self._model_name = model_name


   def _save_model_and_optimizer(self, model, optimizer, path: str) -> None:
        """Save a model and associated optimizer states.

        Args:
            path: Path to which the model and the optimizer state will be saved.
        """
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, path)


    def _save_path(self, index: int) -> str:
        """Path to store a new model iteration at.

        Warning: If this method is changed, the _extract_saved_model_number function below should
        be modified accordingly.
        """
        return os.path.join(self._save_dir, f"{self._model_name}_{index}.pth")


    def _extract_model_iteration(path: str) -> int:
        """Extract the iteration index given a model path."""
        # Exploit the fact that save_loss appends a number with an "_" at the end.
        index_path = path.split("_")[-1]
        
        # Remove the file extension.
        index_path = os.path.splitext(index_path)[0]
        index = int(index_path)

        # Verify that it is consistent with the _save_path function before returning.
        assert self._save_path(index) == path
        return index


    # Maybe this function is superfluous
    def _save_model_version(self, model, optimizer, index: int) -> str:
        """Save a version of the model.

        Args:
            index: This is the version index of the model. Every time it gets saved, the index of
                   the model should be incremented by 1.

        """
        path = self._save_path(index=index)
        print(f"Saving model to {path}.")
        self._save_model_and_optimizer(model, optimizer, path)
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

