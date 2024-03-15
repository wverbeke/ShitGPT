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
    parser.add_argument("--n-shards", type=int, default=10, help="Number of shards the txt file is split in before conversion to binaries and subsequent merging. Python has a large memory overhead for strings, so for very large txt files a lot of shards might be needed during the conversion.")
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

def _shard_txt_file(txt_path: str, num_shards: int, dir_path: str) -> None:
    """Shard a txt file into multiple small files with the 'split' command in linux."""
    os.system(f"split -n {num_shards} {txt_path} {dir_path}/")


def _convert_txt_shards_to_binary(tokenizer: TokenizerBase, dir_path: str) -> None:
    """Convert a directory containing a number of txt files to binary files.

    This will be run on the directory where the split shards of a big txt file are stored.
    """
    # WARNING: It is crucial to keep the correct order here.
    # Here assumptions are made on the filenames coming out of the split command, namely that it
    # creates filenames in an alphabetical order when splitting a text file.
    for i, f in enumerate(sorted(os.listdir(dir_path))):
        fp = os.path.join(dir_path, f)
        
        with open(fp, encoding="utf-8") as txt_file:
            array = np.array(tokenizer.encode(txt_file.read()), dtype=tokenizer.smallest_int_type())

        # Save the full encoded array as a numpy binary.
        out_name = (os.path.splitext(f)[0] + f"_binary_shard_{i}.npy")
        out_path = os.path.join(dir_path, out_name)
        np.save(out_path, array)


def _merge_binaries(input_directory: str, merged_path: str) -> None:
    """Merge a all numpy binaries in a temporary directory into a single file."""
    # WARNING: It is crucial that the order of the files is maintained here.
    def _extract_shard_index(path):
        return int(os.path.splitext(path)[0].split("_")[-1])

    files = [(f, _extract_shard_index(f)) for f in os.listdir(input_directory) if f.endswith(".npy")]
    files = sorted(files, key=lambda x: x[1])
    files = (f[0] for f in files)

    # Read all array chunks and concatenate them at the end.
    arrays = []
    for f in files:
        arrays.append(np.load(os.path.join(input_directory, f)))
    np.save(merged_path, np.concatenate(arrays, axis=0))

if __name__ == "__main__":
    args = _parse_args()

    # Ensure the temporary directory for the binary shards exists.
    tmp_dir = _tmp_dir(args.output_directory)

    # Clean up the current temporary directory.
    _clean_dir(tmp_dir)

    # Make the temporary directory
    os.makedirs(tmp_dir, exist_ok=True)

    # Shard the input text file.
    _shard_txt_file(args.input_path, args.n_shards, tmp_dir)

    # Tokenize the text files and store them as binaries.
    tokenizer = GPT2BPETokenizer()
    _convert_txt_shards_to_binary(tokenizer, tmp_dir)

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
