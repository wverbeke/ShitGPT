from datasets import load_dataset
from tqdm import tqdm
from tokenizer import END_OF_TEXT

dataset = load_dataset("Skylion007/openwebtext")
dataset = dataset["train"]

with open("webtext.txt", "w") as f:
    for d in tqdm(dataset):
        d = d["text"]

        # Should a newline be added here as well or is end of text enough?
        # Feels like we can fit more content without a redundant newline.
        f.write(d + END_OF_TEXT)

