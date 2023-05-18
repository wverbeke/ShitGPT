import os
from constants import SHAKESPEARE_PATH

def get_shakespeare_text() -> str:
    """Get a string containing all of Shakespeare's works."""

    # Download the text if it is not locally available.
    if not os.path.isfile(SHAKESPEARE_PATH):
        # TODO Generalize this to other operating systems than linux.
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        os.system(f"mv input.txt {SHAKESPEARE_PATH}")

    with open(SHAKESPEARE_PATH) as f:
        return f.read()
