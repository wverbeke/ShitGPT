"""Torch Dataset classes for loading text data.

A simple Dataset class taking a tokenizer and a (potentially very large) string is provided. Python
however has a huge memory overhead for strings, so when loading very large datasets exceeding
several GB of pure text, which can be needed to train large transformers, another class is provided.
This class assumes the text has been pre-encoded with a tokenizer and stored in numpy binary files.
To do this pre-encoding see the dataset_to_encoded_binary.py script.

The datasets return a chunk of encoded text as the model input and another equally long chunk of
encoded text as the model targets. The targets are a chunk of text offset by 1 compared to the
model input text. Given the first token the model must predict the second token, given the first and
second tokens the model must predict the third and so on. This setup is more efficient than simply
trying to predict the next token given a sequence since we have a stronger feedback signal to the
model (N_input tokens instead of 1) and the model learns to predict future tokens for arbitrary
sequence lenghts.

The output of the data loaders looks as follows:

y: b c d e
   ^ ^ ^ ^
   | | | |
x: a b c d
"""
import os
from typing import Iterable

import torch
import numpy as np

from tokenizer import TokenizerBase, GPT2BPETokenizer

def _check_text_size(text_tensor, context_window):
    """Verify the context window does not exceed the input text."""
    if len(text_tensor) <= context_window:
        raise ValueError("The data set text should be at least longer than the context window!")


class TextDataset(torch.utils.data.Dataset):
    """Simple Dataset to load encoded data from a text provided as a string.
    
    Use this class when reading small data sets of up to a few GB of utf-8 text. Beyond that
    python's memory overhead in strings will cause crashes.
    """
    def __init__(self, text: str, tokenizer: TokenizerBase, context_window: int):
        self._text_tensor = torch.Tensor(tokenizer.encode(text)).int()
        self._context_window = context_window
        _check_text_size(self._text_tensor, context_window)
        self._vocab_size = tokenizer.vocab_size()

    def vocab_size(self):
        return self._vocab_size

    def context_window(self):
        return self._context_window

    def __len__(self):
        """Number of samples in the data set.

        To get the data set size we have to account for the fact that the model will consume an
        input of size 'context_window' and the targets are the next tokens, so another offset of
        1 is needed.
        """
        return (len(self._text_tensor) - self._context_window)

    def __getitem__(self, index: int):
        end_index = index + self._context_window

        # No need to copy things to GPU at this stage. GPU memory pinning can be done in the
        # Dataloader.
        # Offset the targets by 1 compared to the inpts, as illustrated above.
        x = self._text_tensor[index:end_index].long()
        y = self._text_tensor[index + 1:end_index + 1].long()
        return x, y


class PreEncodedDataset(TextDataset):
    """Pre-encoded dataset.

    This class should be used when training a large model on a large text volume. The text is
    expected to be preprocessed into numpy binaries with dataset_to_encoded_binary.py.
    """
    def __init__(self, binary_file_paths: Iterable, context_window: int, vocab_size: int = None):
        """Read a provided set of binary files containing encoded text into a large numpy array.
    
        By default the vocab size is set to the GPT 2 tokenizer's vocab size.
        """
        arrays = []
        for p in binary_file_paths:
            if not p.endswith(".npy"): 
                raise ValueError("Expect numpy binary files to have .npy extension.")
            arrays.append(np.load(p))

        # Note that we do not want to convert this array to a torch tensor here.
        # Torch does not support uint16, but the GPT 2 tokenizer has only 50257 tokens, meaning
        # it fits into 16 bit unsigned but not signed integers. If we want to maximize the text
        # we can load into our RAM memory, we want to only convert the numpy arrays to torch
        # tensors on an as-needed basis during training.
        # Even though this is a numpy array we keep the "tensor" name so the inherited __len__
        # method works out of the box.
        self._text_tensor = np.concatenate(arrays, axis=0)
        self._context_window = context_window
        if vocab_size is None:
            self._vocab_size = GPT2BPETokenizer().vocab_size()
        else:
            self._vocab_size = vocab_size

    def __getitem__(self, index):
        end_index = index + self._context_window

        # Torch does not support all numpy types like uint16, so cast before making tensors.
        x = torch.from_numpy(self._text_tensor[index:end_index].astype(np.int64))
        y = torch.from_numpy(self._text_tensor[index+1:end_index+1].astype(np.int64))
        return x, y

