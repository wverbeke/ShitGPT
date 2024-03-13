import os
import re
from html2text import html2text
import wikitextparser as wtp
import urllib.request
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import END_OF_TEXT

WIKI_EN_URL = "https://dumps.wikimedia.org/enwiki/20231201/enwiki-20231201-pages-articles-multistream.xml.bz2"
WIKI_SIMPLE_URL = "https://dumps.wikimedia.org/simplewiki/20231020/simplewiki-20231020-pages-meta-current.xml.bz2"


def download_path(url):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    out_name = url.split("/")[-1]
    out_path = os.path.join(current_dir, out_path)
    return out_path

def download(url):
    urllib.request.urlretrieve(url, download_path(url))


def unpack(file_path):
    assert file_path.endswith(".bz2"), "Expected to unpack a .bz2 file."
    os.system(f"bz2 -d {file_path}")


def to_raw_text(wiki_text):
    text = wtp.parse(wiki_text).plain_text()
    text = html2text(text)

    # Is this needed?
    text = text.replace('\\n',' ')

    # Remove excess whitespace.
    text = re.sub('\s+', ' ', text)
    return text


def filter(s, sub):
    if sub in s:
        return s[:s.find(sub)]
    return s


def get_title(wiki_text):
    title = wiki_text.split('<title>')[1].split('</title>')[0].strip()
    return html2text(title)


def get_body(wiki_text):
    body = wiki_text.split('</text')[0].split('<text')[1].split('>', maxsplit=1)[1]
    body = to_raw_text(body)

    # Cut off everything after references, which is usually crap.
    # Note that this search makes the processing a lot slower.
    body=filter(body, "== References ==")
    body=filter(body, "==References==")
    body=filter(body, "== Related pages ==")
    body=filter(body, "==Related pages==")
    return body.strip()

def bad_page(wiki_text):
    """Find if page is useless."""
    if '<redirect title="' in wiki_text:
        return True
    if '(disambiguation)' in wiki_text:
        return True
    if ":" in get_title(wiki_text):
        return True
    return False


def page_generator(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:

        # Skip xml start
        start = False
        for l in f.readlines():
            # begin
            if "<page>" in l:
                current_page = ""
                start = True
            if not start:
                continue

            # end
            if "</page>" in l:
                current_page += l[:l.find("</page>")]
                yield current_page
            # add line to page
            else:
                current_page += l


def process_wikipedia_xml(xml_path, output_path):
    with open(output_path, "w") as f:
        for p in tqdm(page_generator(xml_path)):
            if bad_page(p):
                continue
            title = get_title(p)
            body = get_body(p)
            f.write(f"{title}\n{body}{END_OF_TEXT}")


if __name__ == "__main__":
    process_wikipedia_xml("/home/willem/Downloads/simplewiki-20231020-pages-meta-current.xml", "dump.txt")
