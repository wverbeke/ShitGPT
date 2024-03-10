from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("Skylion007/openwebtext")
dataset = dataset["train"]

with open("webtext.txt", "w") as f:
    for d in tqdm(dataset):
        d = d["text"]
        f.write(d + "<|endoftext|>\n")

