"""Tokenizer classes.

This file provides an interface for Tokenizers which are used throughout the code and two concrete
implementations. One of the implementations simply converts each character present in a long string
to an index. The other tokenizer runs the Byte-pair-encoder based tokenization of GPT-2. To install
the necessary tools see https://github.com/openai/tiktoken.
"""
import abc
from typing import List, Iterable

import torch
import tiktoken


class TokenizerBase(abc.ABC):
    """Tokenizer interface."""
    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialize."""

    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size.

        This refers to the number of tokens, or in the learning problem the number of classes for
        which the model will have to make predictions.
        """

    @abc.abstractmethod
    def encode(self, text: str) -> List:
        """Encode a given string."""

    @abc.abstractmethod
    def decode(self, tokens: Iterable) -> str:
        """Decode a given sequence of tokens."""


class CharTokenizer(TokenizerBase):
    """Simple character-based tokenizer.

    Given a text this tokenizer will map each character to an integer token. When encoding a new text
    the stored mapping will be used, with an additional index for unknown characters.
    """
    def __init__(self, text: str):
        # Extract all individual characters from the text.
        # Note that a set can not be sorted so we need an additional list conversion.
        chars = sorted(list(set(text)))

        self._encode_dict = {c: i for i, c in enumerate(chars)}
        self._decode_dict = {i: c for c, i in self._encode_dict.items()}

        # The vocab size must also account for a one-off-the-end unknown token.
        self._vocab_size = (len(chars) + 1)
        self._unknown_index = len(chars)
        self._decode_dict[self._unknown_index] = ""

    def vocab_size(self):
        return self._vocab_size

    def encode(self, text):
        return [self._encode_dict.get(c, self._unknown_index) for c in text]

    def decode(self, tokens):
        # This will work for both tensors and other iterables.
        return "".join(self._decode_dict[int(t)] for t in tokens)


class GPT2BPETokenizer:
    """Byte pair encoder tokenizer used in GPT 2."""
    def __init__(self):
        self._tokenizer = tiktoken.get_encoding("gpt2")

    def vocab_size(self):
        return self._tokenizer.n_vocab

    def encode(self, text):
        return self._tokenizer.encode(text)

    def decode(self, tokens):
        # The decode method in the official tokenizer does not work for torch tensors.
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens)


if __name__ == "__main__":
    # Some simple testing of the tokenizers.
    test_sentence = "A transformer is a deep learning model. It is distinguished by its adoption of self-attention, differentially weighting the significance of each part of the input (which includes the recursive output) data."
    err_msg = "Sentence must remain identical after encoding and decoding."

    char_tokenizer = CharTokenizer(text=test_sentence)
    assert char_tokenizer.decode(char_tokenizer.encode(test_sentence)) == test_sentence, err_msg

    gpt2_tokenizer = GPT2BPETokenizer()
    assert gpt2_tokenizer.decode(gpt2_tokenizer.encode(test_sentence)) == test_sentence, err_msg


    print("SUCCESS.")
