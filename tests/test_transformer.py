"""Test the transformer implementation."""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from torch import nn
from transformer import CausalSelfAttention, CausalFlashSelfAttention, CausalTorchSelfAttention

IN_CHANNELS = 20
N_HEADS = 5
CONTEXT_WINDOW = 12
BATCH_SIZE = 3

def test_attention(): 
    torch.manual_seed(69)

    in_tensor = torch.rand(BATCH_SIZE, CONTEXT_WINDOW, IN_CHANNELS)

    att_trans_weights = torch.rand(IN_CHANNELS*3, IN_CHANNELS)
    out_transf_weights = torch.rand(IN_CHANNELS, IN_CHANNELS)

    da = CausalSelfAttention(IN_CHANNELS, N_HEADS, CONTEXT_WINDOW)
    fa = CausalFlashSelfAttention(IN_CHANNELS, N_HEADS)

    fa._att_transf.weight = nn.Parameter(att_trans_weights)
    fa._out_transf.weight = nn.Parameter(out_transf_weights)

    da._query_transf.weight = nn.Parameter(att_trans_weights[:IN_CHANNELS, :])
    da._key_transf.weight = nn.Parameter(att_trans_weights[IN_CHANNELS:IN_CHANNELS*2, :])
    da._value_transf.weight = nn.Parameter(att_trans_weights[IN_CHANNELS*2:, :])
    da._out_transf.weight = nn.Parameter(out_transf_weights)

    assert(torch.allclose(da(in_tensor), fa(in_tensor))), "Naive and Flash attention implementations do not give the same result."

    ta = CausalTorchSelfAttention(IN_CHANNELS, N_HEADS, CONTEXT_WINDOW)
    ta._mha.out_proj.weight = nn.Parameter(out_transf_weights)
    ta._mha.in_proj_weight = nn.Parameter(att_trans_weights)

    assert(torch.allclose(ta(in_tensor), fa(in_tensor))), "nn.MultiHead and Flash attention implementations do not give the same result."
    print("Test successful.")


def benchmark_attention(batch_size, context_window, in_channels, n_heads, n_iterations):
    assert torch.cuda.is_available(), "The attention benchmark should be run on GPU."

    fa = CausalFlashSelfAttention(in_channels, n_heads).cuda()
    ta = CausalTorchSelfAttention(in_channels, n_heads, context_window).cuda()

    in_tensor = torch.rand(batch_size, context_window, in_channels).cuda()
    t_start = time.time()
    for _ in range(n_iterations):
        fa(in_tensor)
    t_end = time.time()
    print(f"Manual flash attention took {t_end - t_start} s.")

    t_start = time.time()
    for _ in range(n_iterations):
        ta(in_tensor)
    t_end = time.time()
    print(f"nn.MultiHeadAttention took {t_end - t_start} s.")



if __name__ == "__main__":
    test_attention()
    benchmark_attention(64, 1000, 1024, 32, 1000)
