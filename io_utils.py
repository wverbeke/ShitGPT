"""Some utilities for consistently reading encoded binaries."""

import numpy as np
from tokenizer import TokenizerBase

BIN_EXT = ".bin"

def write_binary(output_path: str, text: str, tokenizer: TokenizerBase) -> None:
    """Write text to disk as an encoded binary.
    
    Warning: Do not confuse this with numpy (.npy) files since those contain type headers, reasing
    the resulting file with np.load will gave wrong results.
    """
    tokens = np.array(tokenizer.encode(text), dtype=tokenizer.smallest_int_type())
    mmap = np.memmap(output_path, dtype=tokens.dtype, mode="w+", shape=tokens.shape)
    mmap[:] = tokens
    mmap.flush()

def binary_memmap(input_path: str, tokenizer: TokenizerBase) -> np.memmap:
    """Memory map to a binary containing encoded text."""
    return np.memmap(input_path, mode="r", dtype=tokenizer.smallest_int_type())

