import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from tokenizer import END_OF_TEXT
from datasets import load_dataset

TXT_DATASET_DIR = "raw_text_datasets"

def get_datasets():
    return {
        "webtext": load_dataset("Skylion007/openwebtext"),
        "wikipedia_english": load_dataset("wikipedia", "20220301.en"),
        "wikipedia_simple_english": load_dataset("wikipedia", "20220301.simple"),
        "bookcorpus": load_dataset("bookcorpus"),
        "open-orca": load_dataset("open-orca"),
    }


def process_wikipedia_page(text):
    """Remove references at the end of page."""
    text = text.strip()
    if text.endswith("References"):
        text = text[:-len("References")]
    text = text.strip()
    text += END_OF_TEXT
    return text



def dump_dataset(dataset, name):
    print(dataset)
    dataset = dataset["train"]
    output_path = os.path.join(TXT_DATASET_DIR, f"{name}.txt")
    with open(output_path, "w") as f:
        for d in tqdm(dataset):
            d = d["text"]
            f.write(d + END_OF_TEXT)


def dump_all_datasets():
    for k, v in get_datasets().items():
        dump_dataset(v, k)


if __name__ == "__main__":
    os.makedirs(TXT_DATASET_DIR, exist_ok=True)
    dump_all_datasets()
