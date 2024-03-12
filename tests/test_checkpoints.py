"""Test the CheckpointHandler class."""
import torch
from torch import nn
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_checkpoints import CheckpointHandler, LossDumper, average_loss_per_accumulation, CHECKPOINT
from constants import DEVICE

TEST_OUTPUT_DIR = "checkpoint_tests"
TEST_NAME = "test_model"
MAX_NUM_CHECKPOINTS = 20
DIM = 20
LOSSES_PER_SHARD=100
ACCUMULATIONS=10

def toy_model():
    model = nn.Sequential()
    for _ in range(10):
        model.append(nn.Linear(DIM, DIM))
        model.append(nn.ReLU())
    return model

def optimizer_and_scaler(model):
    o = torch.optim.Adam(model.parameters())
    s = torch.cuda.amp.GradScaler()

    # Do an optimizer step so it has real parameters.
    targets = torch.rand(DIM)

    # Do a factor 1000 here so the optimizer weights are not tiny, which would break torch.allclose.
    with torch.autocast(device_type=DEVICE, dtype=torch.float16):
        loss = nn.functional.mse_loss(model(targets), targets*1000)
    s.scale(loss).backward()

    s.step(o)
    s.update()
    o.zero_grad()

    return o, s


def random_losses():
    assert LOSSES_PER_SHARD % ACCUMULATIONS == 0, "For testing we assume the number of losses per shard is divisible by the number of accumulations."
    losses = []
    accumulations = []
    for i in range(LOSSES_PER_SHARD//ACCUMULATIONS):
        losses += [n for n in np.random.randn(ACCUMULATIONS)]
        accumulations += [i]*ACCUMULATIONS
    return losses, accumulations


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
            o, s = optimizer_and_scaler(m)
            self._cp_handler.save_new_checkpoint(m, o, s)

    def test_checkpoint_paths(self):
        """Verify that 20 models are stored with the correct paths."""
        # First store 20 checkpoints without overflow.
        clean_directory()
        self.store_random_models(MAX_NUM_CHECKPOINTS)
        test_paths = [os.path.join(TEST_OUTPUT_DIR, TEST_NAME) + f"_{CHECKPOINT}_{i}.pth" for i in range(MAX_NUM_CHECKPOINTS)]
        assert all(a == b for a, b in zip(self._cp_handler.list_checkpoints(), test_paths)), "Some checkpoint paths have the wrong name."
        clean_directory()

        # Now try again with more checkpoints than the maximum amount that is stored.
        self.store_random_models(MAX_NUM_CHECKPOINTS*2)
        test_paths = [os.path.join(TEST_OUTPUT_DIR, TEST_NAME) + f"_{CHECKPOINT}_{i}.pth" for i in range(MAX_NUM_CHECKPOINTS, MAX_NUM_CHECKPOINTS*2)]
        assert all(a == b for a, b in zip(self._cp_handler.list_checkpoints(), test_paths)), "Some checkpoint paths have the wrong name."
        clean_directory()
        print("Test of checkpoint paths successful.")


    def test_stored_weights(self):
        original_m = toy_model()
        original_o, original_s = optimizer_and_scaler(original_m)

        self._cp_handler.save_new_checkpoint(original_m, original_o, original_s)

        loaded_m = toy_model()
        loaded_o, loaded_s = optimizer_and_scaler(loaded_m)
        self._cp_handler.load_latest_checkpoint(loaded_m, loaded_o, loaded_s)
        
        compare_models(original_m, loaded_m)
        compare_optimizers(original_o, loaded_o)
    
        random_m = toy_model()
        random_o, random_s = optimizer_and_scaler(random_m)
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

def test_loss_saving(loss_dumper):
    total_l = []
    total_a = []
    for i in range(20):
        l, a = random_losses()
        loss_dumper.save_new_losses(l, a)
        total_l += l
        total_a += a
    loaded_l, loaded_a = loss_dumper.load_all_losses()
    assert len(total_l) == len(loaded_l), f"Found {len(loaded_l)} loaded loss shards when {len(total_l)} were generated."
    assert np.allclose(np.array(total_l), np.array(loaded_l))
    assert total_a == loaded_a

    # Test the loss accumulation with hardcoded values.
    l = [r for r in np.random.randn(13)]
    a = [0, 0, 1, 1, 1, 2, 2, 1, 1, 0, 0, 3, 3]
    cross_check = [sum(l[0:2]), sum(l[2:5]), sum(l[5:7]), sum(l[7:9]), sum(l[10:12]), sum(l[12:])]
    avg_per_acc = average_loss_per_accumulation(l, a)
    assert np.allclose(np.array(cross_check), np.array(avg_per_acc)), "Losses averaged per accumulation are not equal to cross check."
                
    clean_directory()
    print("Test of loss dumper successful.")

        



if __name__ == "__main__":
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    ch = CheckpointHandler(TEST_OUTPUT_DIR, TEST_NAME, MAX_NUM_CHECKPOINTS, silent=True)
    test_ch = TestCPHandler(ch)

    # TODO, this could be more neatly wrapped.
    try:
        test_ch.test_checkpoint_paths()

    # TODO: This doesn't clean up the empty directory at the end.
    except AssertionError as err:
        clean_directory()
        raise err 

    try:
        test_ch.test_stored_weights()
    except AssertionError as err:
        clean_directory()
        raise err 

    ld = LossDumper(TEST_OUTPUT_DIR, TEST_NAME, silent=True)
    try:
        test_loss_saving(ld)
    except AssertionError as err:
        clean_directory()
        raise err

    os.rmdir(TEST_OUTPUT_DIR)
