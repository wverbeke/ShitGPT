import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from tokenizer import END_OF_TEXT
from datasets import load_dataset
#from torchtext.datasets import CC100
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


def dump_cc100():
    #dataset = CC100(root="./torchtext_datasets/", language_code="en")
    dataset = load_dataset("cc100", "en")
    dataset = dataset["train"]
    for i, d in enumerate(dataset):
        t = d["text"]
        if len(t) >= 2000:
            print(t)


def dump_alpaca_gpt4():
    dataset = load_dataset("vicgalle/alpaca-gpt4")
    dataset = dataset["train"]
    output_path = os.path.join(TXT_DATASET_DIR, "alpaca.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for d in tqdm(dataset):
            instruction = d["instruction"]
            inp = d["input"]
            outp = d["output"]
            out_text = f"{instruction} {inp} {outp}\n{END_OF_TEXT}"
            f.write(out_text)

#dataset = load_dataset("Open-Orca/SlimOrca-Dedup")




if __name__ == "__main__":
    os.makedirs(TXT_DATASET_DIR, exist_ok=True)
    #dump_webtext()
    #dump_openorca()
    #dump_c4()
    dump_cc100()
    #dump_alpaca_gpt4()

