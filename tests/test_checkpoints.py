"""Test the CheckpointHandler class."""
import torch
from torch import nn

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_checkpoints import CheckpointHandler

TEST_OUTPUT_DIR = "checkpoint_tests"
TEST_NAME = "test_model"
MAX_NUM_CHECKPOINTS = 20
DIM = 20

def toy_model():
    model = nn.Sequential()
    for _ in range(10):
        model.append(nn.Linear(DIM, DIM))
        model.append(nn.ReLU())
    return model

def optimizer(model):
    o = torch.optim.Adam(model.parameters())

    # Do an optimizer step so it has real parameters.
    targets = torch.rand(DIM)

    # Do a factor 1000 here so the optimizer weights are not tiny, which would break torch.allclose.
    loss = nn.functional.mse_loss(model(targets), targets*1000)
    loss.backward()
    o.step()
    o.zero_grad()

    return o


def clean_directory():
    for f in os.listdir(TEST_OUTPUT_DIR):
        os.remove(os.path.join(TEST_OUTPUT_DIR, f))


def compare_models(lhs, rhs, negate=False):
    for lhs, rhs in zip(lhs.parameters(), rhs.parameters()):
        cond = torch.allclose(lhs.data, rhs.data), "Models do not have the same weights."

def compare_optimizers(lhs, rhs):
    lhs = lhs.state_dict()["state"]
    rhs = rhs.state_dict()["state"]
    for layer_k in lhs:
        for value_k in lhs[layer_k]:
            assert torch.allclose(lhs[layer_k][value_k], rhs[layer_k][value_k]), "Optimizers do not have the same state."


class TestCPHandler:

    def __init__(self, cp_handler):
        self._cp_handler = cp_handler

    def store_random_models(self, num_models):
        for _ in range(num_models):
            m = toy_model()
            o = optimizer(m)
            self._cp_handler.store_new_checkpoint(m, o)

    def test_checkpoint_paths(self):
        """Verify that 20 models are stored with the correct paths."""
        # First store 20 checkpoints without overflow.
        clean_directory()
        self.store_random_models(MAX_NUM_CHECKPOINTS)
        test_paths = [os.path.join(TEST_OUTPUT_DIR, TEST_NAME) + f"_{i}.pth" for i in range(MAX_NUM_CHECKPOINTS)]
        assert all(a == b for a, b in zip(self._cp_handler.list_checkpoints(), test_paths)), "Some checkpoint paths have the wrong name."
        clean_directory()

        # Now try again with more checkpoints than the maximum amount that is stored.
        self.store_random_models(MAX_NUM_CHECKPOINTS*2)
        test_paths = [os.path.join(TEST_OUTPUT_DIR, TEST_NAME) + f"_{i}.pth" for i in range(MAX_NUM_CHECKPOINTS, MAX_NUM_CHECKPOINTS*2)]
        assert all(a == b for a, b in zip(self._cp_handler.list_checkpoints(), test_paths)), "Some checkpoint paths have the wrong name."
        clean_directory()
        print("Test of checkpoint paths successful.")


    def test_stored_weights(self):
        original_m = toy_model()
        original_o = optimizer(original_m)

        self._cp_handler.store_new_checkpoint(original_m, original_o)
        
        loaded_m = toy_model()
        loaded_o = optimizer(loaded_m)
        self._cp_handler.load_latest_checkpoint(loaded_m, loaded_o)
        
        compare_models(original_m, loaded_m)
        compare_optimizers(original_o, loaded_o)

    
        random_m = toy_model()
        random_o = optimizer(random_m)
        for original, random in zip(original_m.parameters(), random_m.parameters()):
            assert not torch.allclose(original.data, random.data), "Original and random model should not have the same weights."

        lhs = original_o.state_dict()["state"]
        rhs = random_o.state_dict()["state"]
        for layer_k in lhs:
            for value_k in lhs[layer_k]:
                if lhs[layer_k][value_k].nelement() == 1:
                    continue
                assert not torch.allclose(lhs[layer_k][value_k], rhs[layer_k][value_k]), "Original and random optimizer should not have the same state."

        clean_directory()
        print("Test of loaded model and optimizer weights successful.")



if __name__ == "__main__":
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    ch = CheckpointHandler(TEST_OUTPUT_DIR, "test_model", MAX_NUM_CHECKPOINTS, silent=True)
    test_ch = TestCPHandler(ch)

    # TODO, this could be more neatly wrapped.
    try:
        test_ch.test_checkpoint_paths()
    except AssertionError as err:
        clean_directory()
        raise err 

    try:
        test_ch.test_stored_weights()
    except AssertionError as err:
        clean_directory()
        raise err 

    os.rmdir(TEST_OUTPUT_DIR)
