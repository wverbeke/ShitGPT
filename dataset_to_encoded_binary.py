"""Script to convert datasets to a numpy binary.

TODO: Clean this up and make command line arguments!
"""
import os
import sys
import numpy as np
from text_dataset import BPETokenizer

N_SHARDS = 10
TMP_DIRECTORY = "tmp"


def clean_temp():
    for f in os.listdir(TMP_DIRECTORY):
        os.remove(os.path.join(TMP_DIRECTORY, f))

def shard_text_file(path):
    os.system(f"split -n {N_SHARDS} {path} {TMP_DIRECTORY}/")


def shards_to_binary(tokenizer):
    for i, f in enumerate(os.listdir(TMP_DIRECTORY)):
        arrays = []
        with open(os.path.join(TMP_DIRECTORY, f)) as f:
            for l in f.readlines():
                arrays.append(np.array(tokenizer.encode(l)).astype(np.uint16))
        arrays = np.concatenate(arrays, axis=0)
        np.save(os.path.join(TMP_DIRECTORY, f"binary_shard_{i}.npy"), arrays)


def merge_binaries(target_path):
    arrays = []
    for f in os.listdir(TMP_DIRECTORY):
        if not f.endswith(".npy"): continue
        arrays.append(np.load(os.path.join(TMP_DIRECTORY, f)))
    np.save(target_path, np.concatenate(arrays, axis=0))

if __name__ == "__main__":
    os.makedirs(TMP_DIRECTORY, exist_ok=True)
    clean_temp()
    input_path = sys.argv[1]

    shard_text_file(input_path)

    tokenizer = BPETokenizer()
    shards_to_binary(tokenizer)
    merge_binaries(os.path.join("cleaned_datasets/", os.path.basename(input_path.replace(".txt", ".npy"))))
    clean_temp()
