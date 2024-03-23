"""Script to convert datasets to a numpy binary.

Currently only works in Linux because the 'split' command is used. The text file is tokienized with
the GPT2BPETokenizer class.
"""

import argparse
import os
import sys
import numpy as np
from tokenizer import GPT2BPETokenizer, TokenizerBase
from io_utils import BIN_EXT, write_binary, binary_memmap

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
    """Path to a file shard."""
    return os.path.join(output_path, f"shard_{index}{BIN_EXT}")

def _txt_to_binary_shards(txt_path: str, output_path: str, tokenizer: TokenizerBase, shard_size: int) -> None:
    """Convert a directory containing a number of txt files to binary files.

    This will be run on the directory where the split shards of a big txt file are stored.
    """
    with open(txt_path, encoding="utf-8") as txt_file:
        shard_counter = 0
        current_words = ""
        for l in txt_file:
            current_words += l

            if len(current_words) > shard_size:
                write_binary(_shard_path(output_path, shard_counter), current_words, tokenizer)
                shard_counter += 1
                current_words = ""
        if len(current_words):
            write_binary(_shard_path(output_path, shard_counter), current_words, tokenizer)


def _merge_binaries(input_directory: str, merged_path: str, tokenizer: TokenizerBase) -> None:
    """Merge a all numpy binaries in a temporary directory into a single file."""
    # WARNING: It is crucial that the order of the files is maintained here.
    def _extract_shard_index(path):
        return int(os.path.splitext(path)[0].split("shard_")[-1])

    files = [(f, _extract_shard_index(f)) for f in os.listdir(input_directory) if f.endswith(BIN_EXT)]
    files = sorted(files, key=lambda x: x[1])
    files = (f[0] for f in files)

    # Read all array chunks and concatenate them at the end.
    arrays = []
    for f in files:
        path = os.path.join(input_directory, f)
        arrays.append(binary_memmap(path, tokenizer))

    total_len = sum(len(a) for a in arrays)
    result = np.memmap(merged_path, dtype=arrays[0].dtype, mode="w+", shape=(total_len))
    start = 0
    for a in arrays:
        result[start:start + len(a)] = a
        start += len(a)
    result.flush()


def encode_text_to_disk(txt_path: str, output_directory: str, tokenizer: TokenizerBase, shard_size: int) -> str:
    """Encode a potentially very large text file to a binary on disk.

    To deal with large files it happens in two steps. First shards are written to disk.
    Subsequently the shards are streamed into a single large binary.
    """
    # Temporary directory for shards.
    tmp_dir = _tmp_dir(output_directory)

    # Clean up the current temporary directory.
    _clean_dir(tmp_dir)

    # Make the temporary directory
    os.makedirs(tmp_dir, exist_ok=True)

    # Shard the input text file into token binaries.
    _txt_to_binary_shards(txt_path, tmp_dir, tokenizer, shard_size)

    # Merge the binaries into a single file.
    output_name = os.path.splitext(os.path.basename(args.input_path))[0] + ".bin"
    output_path = os.path.join(args.output_directory, output_name)
    _merge_binaries(tmp_dir, output_path, tokenizer)

    # Clean the temporary directory.
    _clean_dir(tmp_dir)

    return output_path



if __name__ == "__main__":
    args = _parse_args()

    # Tokenize the text files and store them as binaries.
    tokenizer = GPT2BPETokenizer()

    # Write the encoded binary.
    output_path = encode_text_to_disk(args.input_path, args.output_directory, tokenizer, args.shard_size)

    # Test the functionality if requested.
    # Verify that decoding the binary file results in the original text.
    if not args.test:
        sys.exit()

    with open(args.input_path, encoding="utf-8") as f:
        original = f.read()

    decoded = tokenizer.decode(binary_memmap(output_path, tokenizer))
    print("decoded init = ", decoded[:10])
    print("original init = ", original[:10])
    print("decoded end = ", decoded[-10:])
    print("original end = ", original[-10:])
    print("len(decoded) = ", len(decoded))
    print("len(original) = ", len(original))
    assert decoded == original, "Decoded and original text must be the same!"
    print("Test successful!")
