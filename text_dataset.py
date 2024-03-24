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
from io_utils import BIN_EXT, binary_memmap

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


    def slice(self, start: int, end: int):
        return self._text_tensor[start:end].long()


    def __getitem__(self, index: int):

        # + 1 because we read out a chunk containing both x and y.
        end_index = index + self._context_window + 1
        text_chunk = self.slice(index, end_index)
        x = text_chunk[:-1]
        y = text_chunk[1:]
        return x, y


def _tensor(array_like, start_index, end_index):
    return torch.from_numpy(array_like[start_index:end_index].astype(np.int64))

class PreEncodedMemoryDataset(TextDataset):
    """Pre-encoded dataset.

    This class should be used when training a large model on a large text volume. The text is
    expected to be preprocessed into numpy binaries with dataset_to_encoded_binary.py.
    """
    def __init__(self, binary_file_paths: Iterable, context_window: int):
        """Read a provided set of binary files containing encoded text into a large numpy array."""
        tokenizer = GPT2BPETokenizer()
        arrays = []
        for p in binary_file_paths:
            if not p.endswith(BIN_EXT): 
                raise ValueError(f"Expect encoded binary files to have {BIN_EXT} extension, but got {p}.")
            arrays.append(binary_memmap(p, tokenizer))

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
        self._vocab_size = tokenizer.vocab_size()

    def slice(self, start: int, end: int):
        return _tensor(self._text_tensor, start, end)


# WIP!
class PreEncodedDiskDataset(TextDataset):

    def __init__(self, binary_file_paths: Iterable, context_window: int):

        # TODO: There is a large overlap here with PreEncodedMemoryDataset.
        tokenizer = GPT2BPETokenizer()
        self._memory_maps = []
        for p in binary_file_paths:
            if not p.endswith(BIN_EXT):
                raise ValueError(f"Expect encoded binary files to have {BIN_EXT} extension, but got {p}.")
            self._memory_maps.append(binary_memmap(p, tokenizer))
        self._cum_lengths = np.cumsum(np.array([len(m) for m in self._memory_maps]))
        self._context_window = context_window
        self._vocab_size = GPT2BPETokenizer().vocab_size()
        
    def _memmap_indices(self, sample_index):
        for i, cl in enumerate(self._cum_lengths):
            if sample_index <= cl:
                offset = self._cum_lengths[i - 1] if i > 0 else 0
                return i, sample_index - offset
        raise IndexError(f"Out of bound sample at index {sample_index}")

    def __len__(self):
        return self._cum_lengths[-1] - self._context_window

    def slice(self, start: int, end: int):
        mmap_start_index, sample_start_index = self._memmap_indices(start)
        mmap_end_index, sample_end_index = self._memmap_indices(end)

        # The drawn sample spans multiple datasets.
        # Assume the context window can never span more than two datasets.
        if mmap_end_index != mmap_start_index:
            chunk_1 = self._memory_maps[mmap_start_index][sample_start_index:]
            chunk_2 = self._memory_maps[mmap_end_index][:sample_end_index]
            return torch.from_numpy(np.concatenate([chunk_1, chunk_2]).astype(np.int64))

        return _tensor(self._memory_maps[mmap_start_index], sample_start_index, sample_end_index)


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
