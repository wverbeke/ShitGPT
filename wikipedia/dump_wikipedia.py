import os
import re
from html2text import html2text
import wikitextparser as wtp
import urllib.request
from tqdm import tqdm
import progressbar

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import END_OF_TEXT

WIKI_EN_URL = "https://dumps.wikimedia.org/enwiki/20231201/enwiki-20231201-pages-articles-multistream.xml.bz2"
WIKI_SIMPLE_URL = "https://dumps.wikimedia.org/simplewiki/20231020/simplewiki-20231020-pages-meta-current.xml.bz2"

class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def _download_path(url):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    out_name = url.split("/")[-1]
    out_path = os.path.join(current_dir, out_name)
    return out_path

def download(url):
    urllib.request.urlretrieve(url, _download_path(url), MyProgressBar())
    return _download_path(url)

def unpack(file_path):
    assert file_path.endswith(".bz2"), "Expected to unpack a .bz2 file."
    os.system(f"bz2 -d {file_path}")


def to_raw_text(wiki_text):
    # Somehow tables seem to cause errors.
    # We can use replace_tables=False, but instead we'll just avoid
    # pages with tables.
    #text = wtp.parse(wiki_text).plain_text(replace_tables=False)
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
    body = wiki_text.strip().split('</text')[0].split('<text')[1].split('>', maxsplit=1)[1]
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
        for l in f:
            # begin
            if "<page>" in l:
                current_page = ""
                start = True
            if not start:
                continue

            # end
            if "</page>" in l:
                current_page += l[:l.find("</page>")]
                start = False
                yield current_page
            # add line to page
            else:
                current_page += l


def process_wikipedia_xml(xml_path, output_path):
    with open(output_path, "w") as f:
        for p in tqdm(page_generator(xml_path)):
            if bad_page(p):
                continue
            try:
                title = get_title(p)
                body = get_body(p)
                f.write(f"{title}\n{body}{END_OF_TEXT}")
            except:
                pass

def main():

    # TODO: Add conversion here.
    #en_wiki_path = "enwiki-20231201-pages-articles-multistream.xml"
    #process_wikipedia_xml(en_wiki_path, "wiki_en.txt")

    #simple_wiki_path = download(WIKI_SIMPLE_URL)
    simple_wiki_path = "simplewiki-20231020-pages-meta-current.xml"
    process_wikipedia_xml(simple_wiki_path, "wiki_simple.txt")



if __name__ == "__main__":
    main()
