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
import random

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


class PreEncodedMemoryDataset(TextDataset):
    """Pre-encoded dataset.

    This class should be used when training a large model on a large text volume. The text is
    expected to be preprocessed into numpy binaries with dataset_to_encoded_binary.py.
    """
    def __init__(self, binary_file_paths: Iterable, context_window: int):
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

        # Concatenate forces arrays into memory!
        self._text_tensor = np.concatenate(arrays, axis=0)
        self._context_window = context_window
        self._vocab_size = GPT2BPETokenizer().vocab_size()

    def __getitem__(self, index):
        end_index = index + self._context_window

        # Torch does not support all numpy types like uint16, so cast before making tensors.
        x = torch.from_numpy(self._text_tensor[index:end_index].astype(np.int64))
        y = torch.from_numpy(self._text_tensor[index+1:end_index+1].astype(np.int64))
        return x, y


# WIP!
class PreEncodedDiskDataset(TextDataset):

    def __init__(self, binary_file_paths: Iterable, context_window: int):
        self._memory_maps = []
        for p in binary_file_paths:
            if not p.endswith(".npy"):
                raise ValueError("Expect numpy binary files to have .npy extension.")
            self._memory_maps.append(np.load(p, mmap_mode="r"))
        self._cum_lengths = np.cumsum(np.array([len(m) for m in self._memory_maps]))
        self._context_window = context_window
        self._vocab_size = GPT2BPETokenizer().vocab_size()
        
    def _memmap_indices(self, sample_index):
        for i, cl in enumerate(self._cum_lengths):
            if sample_index < cl:
                offset = self._cum_lengths[i - 1] if i > 0 else 0
                return i, sample_index - offset
        raise IndexError(f"Out of bound sample at index {sample_index}")

    def __len__(self):
        return self._cum_lengths[-1] - self._context_window

    def __getitem__(self, index):
        
        # Assume the context window can never span multiple datasets.
        mmap_start_index, sample_start_index = self._memmap_indices(index)
        mmap_end_index, sample_end_index = self._memmap_indices(end_index)
        
        # The drawn sample spans multiple datasets.
        if mmap_end_index != mmap_start_index:
            raise NotImplementedError("This branch should still be implemented.")

        # The drawn sample comes from a single dataset.
        mmap = self._memory_maps[mmap_start_index]
        x = torch.from_numpy(mmap[sample_start_index:sample_end_index].astype(np.int64))
        y = torch.from_numpy(mmap[sample_start_index + 1:sample_end_index + 1].astype(np.int64))
        return x, y


def infinite_dataloader(dataloader):
    """Wrap a dataloader to become infinite.

    This is convenient if we want to iterate a number of steps that is not directly related to the
    size of our dataset.
    """
    i = iter(dataloader)
    while True:
        try:
            yield next(i)
        except StopIteration:
            i = iter(dataloader)
            yield next(i)


def data_loader(dataset, batch_size):
    """Build a data loader.

    Pytorch stores all possible sampling indexes in memory. This will cause an OOM error even with
    a ridiculous amount of CPU RAM memory just because the amount of possible sampling positions is
    so large. We have to override the pytorch sampling and do simple sampling with replacement.
    """
    def _sample_index():
        while True:
            yield random.randint(0, len(dataset))

    # We can do a huge amount of prefetching since text data is so small.
    return infinite_dataloader(torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=_sample_index(), prefetch_factor=100, num_workers=24, pin_memory=True))
