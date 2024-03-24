"""Test the transformer implementation."""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


if __name__ == "__main__":
    test_attention()
