import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from tokenizer import END_OF_TEXT
from datasets import load_dataset
import re

TXT_DATASET_DIR = "raw_text_datasets"

def filter_word_webtext(w):
    if "pic.twitter" in w:
        return False
    if "http:" in w:
        return False
    if "https:" in w:
        return False
    if "tinyurl" in w:
        return False
    return True


def dump_webtext():
    dataset = load_dataset("Skylion007/openwebtext")
    dataset = dataset["train"]
    output_path = os.path.join(TXT_DATASET_DIR, "open_webtext.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for d in tqdm(dataset):
            text = d["text"]
            text = re.sub('\s+', ' ', text)
            clean_text = []
            for w in text.split():
                if not filter_word_webtext(w):
                    continue
                clean_text.append(w)
            clean_text = " ".join(clean_text)
            f.write(clean_text + "\n" + END_OF_TEXT)

def dump_bookcorpus():
    dataset = load_dataset("bookcorpus")
    dataset = dataset["train"]
    output_path = os.path.join(TXT_DATASET_DIR, "book_corpus.txt")
    with open(output_path, "w") as f:
        for d in tqdm(dataset):
            text = d["text"]
            f.write(text)


def filter_question_openorca(q):
    if "translate" in q.lower():
        return False
    if "translation" in q.lower():
        return False
    return True

def dump_openorca():
    dataset = load_dataset("Open-Orca/OpenOrca")
    dataset = dataset["train"]
    output_path = os.path.join(TXT_DATASET_DIR, "open_orca.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for d in tqdm(dataset):
            prompt = d["system_prompt"]
            question = d["question"]
            if not filter_question_openorca(question):
                continue
            response = d["response"]
            out_text = f"{prompt} {question} {response}\n{END_OF_TEXT}"
            f.write(out_text)

def dump_c4():
    dataset = load_dataset("allenai/c4", "en")
    print(dataset.keys())


if __name__ == "__main__":
    os.makedirs(TXT_DATASET_DIR, exist_ok=True)
    #dump_webtext()
    #dump_openorca()
    dump_c4()
