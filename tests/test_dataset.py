import os
import torch
import numpy as np

from text_dataset import TextDataset, PreEncodedDataset
from tokenizer import GPT2BPETokenizer, TokenizerBase
from utils import get_shakespeare_text

TMP = "tmp"

def _text_from_dataset(dset: torch.utils.data.Dataset, tokenizer: TokenizerBase):
    """Retrieve the text from a dataset with context window 1.

    WARNING: This routine only works properly if context_window 1 has been set.
    """
    assert dset.context_window() == 1, "This routine only works for datasets with context window 1."

    encoded_text = []
    for i, (x, y) in enumerate(dset):
        encoded_text.append(x)

        # Important to put to break here so another y value does not get set by the loop.
        if i == (len(dset) - 1): break

    # The last token can only be generated as a target.
    encoded_text.append(y)

    # Decode the text.
    return tokenizer.decode(encoded_text)


def test_text_dataset(text: str):
    """Test whether the outputs of text dataset are correct.

    To do this we will instantiate the dataset with a text, subsequently generate batches to
    reproduce the original text and see that this is indeed equal to the original text.
    """
    tokenizer=GPT2BPETokenizer()
    dset = TextDataset(text=text, tokenizer=tokenizer, context_window=1)
    decoded_text = _text_from_dataset(dset=dset, tokenizer=tokenizer)
    assert decoded_text == text, "Text coming out of TextDataset must be able to reproduce the original text."


def test_pre_encoded_dataset(text: str):

    tokenizer=GPT2BPETokenizer()

    # Write text to temporary numpy file.
    test_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(test_dir, TMP)
    os.makedirs(tmp_dir, exist_ok=True)
    binary_path = os.path.join(tmp_dir, "test_pre_encoded.npy")
    encoded_text = np.array(tokenizer.encode(text), dtype=np.uint16)
    np.save(binary_path, encoded_text)

    # Read the encoded text as a PreEncodedDataset.
    dset = PreEncodedDataset(binary_file_paths=[binary_path], context_window=1, vocab_size=tokenizer.vocab_size())

    # Yield text from the dataset and decode.
    decoded_text = _text_from_dataset(dset=dset, tokenizer=tokenizer)

    # Clean up files.
    # We do it before the final assert so that no dirty files are left if the test fails.
    os.remove(binary_path)
    os.rmdir(tmp_dir)

    # Verify that the original text can be reproduced.
    assert decoded_text == text, "Text coming out of TextDataset must be able to reproduce the original text."


if __name__ == "__main__":
    shakespeare_text = get_shakespeare_text()
    test_text_dataset(shakespeare_text)
    test_pre_encoded_dataset(shakespeare_text)
