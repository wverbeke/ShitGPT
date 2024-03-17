"""Script to convert datasets to a numpy binary.

Currently only works in Linux because the 'split' command is used. The text file is tokienized with
the GPT2BPETokenizer class.
"""

import argparse
import os
import sys
import numpy as np
from tokenizer import GPT2BPETokenizer, TokenizerBase

TMP = "tmp_binary_shards"

def _parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Command line arguments for converting large txt files into numpy binaries. Currently the GPT2 tokenizer will always be used.")
    parser.add_argument("--input-path", type=str, help="Path to the txt file that will be converted to a binary.")
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--shard-size", type=int, default=10000000, help="Before making a large binary file with the entire encoded text, we write binary shards. The shard size specified the number of words being written. Python has a large memory overhead for strings, so this number can not be too large.")
    parser.add_argument("--test", action="store_true", help="Use to test whether the script is working. This will generate a binary, and decode it and check it is equal to the original text. Will not work for very large text files that do not fit into memory.")
    # TODO: Potentially add other tokenizers and appropriate data types for their resulting binaries.

    return parser.parse_args()

def _tmp_dir(output_directory: str) -> None:
    """Temporary directory for binary shards."""
    return os.path.join(output_directory, TMP)

def _clean_dir(path: str) -> None:
    """Clean files from a temporary directory.

    WARNING: Do not call this on any directory you are not sure of since all files will be removed.
    """
    if not TMP in path:
        return
    if not os.path.isdir(path):
        return
    for root, _, files in os.walk(path):
        for f in files:
            os.remove(os.path.join(root, f))
    os.rmdir(path)

def _shard_path(output_path: str, index: int):
    return os.path.join(output_path, f"shard_{index}.npy")


def _write_binary(output_path, word_list, tokenizer):
    tokens = np.array(tokenizer.encode(" ".join(word_list)), dtype=tokenizer.smallest_int_type())
    np.save(output_path, tokens)


def _txt_to_binary_shards(txt_path: str, output_path: str, tokenizer: TokenizerBase, shard_size: int) -> None:
    """Convert a directory containing a number of txt files to binary files.

    This will be run on the directory where the split shards of a big txt file are stored.
    """
    with open(txt_path, encoding="utf-8") as txt_file:
        shard_counter = 0
        current_words = []
        for l in txt_file:
            new_words = l.split()
            if len(current_words) + len(new_words) > shard_size:
                _write_binary(_shard_path(output_path, shard_counter), current_words, tokenizer)
                shard_counter += 1
                current_words = []
            else:
                current_words += new_words
        if len(current_words):
            _write_binary(_shard_path(output_path, shard_counter), current_words, tokenizer)


def _merge_binaries(input_directory: str, merged_path: str) -> None:
    """Merge a all numpy binaries in a temporary directory into a single file."""
    # WARNING: It is crucial that the order of the files is maintained here.
    def _extract_shard_index(path):
        return int(os.path.splitext(path)[0].split("shard_")[-1])

    files = [(f, _extract_shard_index(f)) for f in os.listdir(input_directory) if f.endswith(".npy")]
    files = sorted(files, key=lambda x: x[1])
    files = (f[0] for f in files)

    # Read all array chunks and concatenate them at the end.
    arrays = []
    for f in files:
        arrays.append(np.memmap(os.path.join(input_directory, f)))

    total_len = sum(len(a) for a in arrays)
    result = np.memmap(merged_path, dtype=arrays[0].dtype, mode="w+", shape=(total_len))
    start = 0
    for a in arrays:
        result[start:start + len(a)] = a
        start += len(a)
    result.flush()

if __name__ == "__main__":
    args = _parse_args()

    # Ensure the temporary directory for the binary shards exists.
    tmp_dir = _tmp_dir(args.output_directory)

    # Clean up the current temporary directory.
    _clean_dir(tmp_dir)

    # Make the temporary directory
    os.makedirs(tmp_dir, exist_ok=True)

    # Tokenize the text files and store them as binaries.
    tokenizer = GPT2BPETokenizer()

    # Shard the input text file into token binaries.
    _txt_to_binary_shards(args.input_path, tmp_dir, tokenizer, args.shard_size)

    # Merge the binaries into a single file.
    output_name = os.path.splitext(os.path.basename(args.input_path))[0] + ".npy"
    output_path = os.path.join(args.output_directory, output_name)
    _merge_binaries(tmp_dir, output_path)

    # Clean the temporary directory.
    _clean_dir(tmp_dir)

    # Test the functionality if requested.
    # Verify that decoding the binary file results in the original text.
    if not args.test:
        sys.exit()

    with open(args.input_path, encoding="utf-8") as f:
        original = f.read()
    decoded = tokenizer.decode(np.load(output_path))
    assert decoded == original, "Decoded and original text must be the same!"
    print("Test successful!")
