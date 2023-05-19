from text_dataset import TextDataset
from tokenizer import GPT2BPETokenizer
from utils import get_shakespeare_text


def test_text_dataset(text: str):
    """Test whether the outputs of text dataset are correct.

    To do this we will instantiate the dataset with a text, subsequently generate batches to
    reproduce the original text and see that this is indeed equal to the original text.
    """
    tokenizer=GPT2BPETokenizer()
    dset = TextDataset(text=text, tokenizer=tokenizer, context_window=1)
    encoded_text = []

    for i, (x, y) in enumerate(dset):
        encoded_text.append(x)

        # Important to put to break here so another y value does not get set by the loop.
        if i == (len(dset) - 1): break

    # The last token can only be generated as a target.
    encoded_text.append(y)

    decoded_text = tokenizer.decode(encoded_text)
    assert decoded_text == text, "Text coming out of TextDataset must be able to reproduce the original text."


if __name__ == "__main__":
    shakespeare_text = get_shakespeare_text()
    test_text_dataset(shakespeare_text)
