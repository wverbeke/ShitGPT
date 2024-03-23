import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np

from text_dataset import TextDataset, PreEncodedMemoryDataset, PreEncodedDiskDataset
from tokenizer import GPT2BPETokenizer, TokenizerBase
from utils import get_shakespeare_text
from io_utils import write_binary, BIN_EXT

TMP = "tmp"

def _text_from_dataset(dset: torch.utils.data.Dataset, tokenizer: TokenizerBase, context_window: int):
    """Retrieve the text from a dataset."""
    #assert dset.context_window() == 1, "This routine only works for datasets with context window 1."

    encoded_text = []
    last_index = 0

    # The datset classes are explicitly made so that pytorch can safely sample any index and get
    # a full context window, and the same shifted by 1 as a target.
    end_point = len(dset) + context_window
    while last_index + context_window <= end_point:
        encoded_text.append(dset.slice(last_index, last_index + context_window))
        last_index += context_window
    if last_index < end_point:
        encoded_text.append(dset.slice(last_index, end_point))

    # Decode the text.
    out = ""
    for e in encoded_text:
        out += tokenizer.decode(e)
    return out


def test_text_dataset(text: str, context_window: int):
    """Test whether the outputs of text dataset are correct.

    To do this we will instantiate the dataset with a text, subsequently generate batches to
    reproduce the original text and see that this is indeed equal to the original text.
    """
    tokenizer=GPT2BPETokenizer()
    dset = TextDataset(text=text, tokenizer=tokenizer, context_window=context_window)
    decoded_text = _text_from_dataset(dset=dset, tokenizer=tokenizer, context_window=context_window)
    assert decoded_text == text, "Text coming out of TextDataset must be able to reproduce the original text."


def test_pre_encoded_dataset(text: str, dataset_cls: TextDataset, context_window: int):

    tokenizer=GPT2BPETokenizer()

    # Write text to temporary numpy file.
    test_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(test_dir, TMP)
    os.makedirs(tmp_dir, exist_ok=True)
    binary_path = os.path.join(tmp_dir, f"test_pre_encoded{BIN_EXT}")
    write_binary(binary_path, text, tokenizer)

    # Read the encoded text as a PreEncodedDataset.
    dset = dataset_cls(binary_file_paths=[binary_path], context_window=context_window)

    # Yield text from the dataset and decode.
    decoded_text = _text_from_dataset(dset=dset, tokenizer=tokenizer, context_window=context_window)

    # Clean up files.
    # We do it before the final assert so that no dirty files are left if the test fails.
    os.remove(binary_path)
    os.rmdir(tmp_dir)

    # Verify that the original text can be reproduced.
    assert decoded_text == text, "Text coming out of TextDataset must be able to reproduce the original text."


if __name__ == "__main__":
    shakespeare_text = get_shakespeare_text()
    test_text_dataset(shakespeare_text, 1)
    test_text_dataset(shakespeare_text, 1000)
    test_pre_encoded_dataset(shakespeare_text, PreEncodedMemoryDataset, 1)
    test_pre_encoded_dataset(shakespeare_text, PreEncodedMemoryDataset, 1000)
    test_pre_encoded_dataset(shakespeare_text, PreEncodedDiskDataset, 1)
    test_pre_encoded_dataset(shakespeare_text, PreEncodedDiskDataset, 1000)
    print("Test successful.")
