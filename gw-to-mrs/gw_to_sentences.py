"""
Usage: python run.py

Extracts sentences and writes them to a new file, one per line, with a unique id.

After the run:

Get a random subset of 1M:
> cat sentences.txt | shuf | head -n 1000000 > sentences1M.txt

Check vocab size with:
> cat sentences1M.txt | awk -F'\t' '{print $2}' | tr -d '[:punct:]' | tr -d '0123456789' | tr '[:upper:]' '[:lower:]' | tr ' ' '\n' | sort | uniq | wc -l

Check number of words with:
> cat sentences1M.txt | awk -F'\t' '{print $2}' | tr ' ' '\n' | wc -l

Write ids and sentences to separate files:
> cat sentences1M.txt | awk -F'\t' '{print $1}' > sentences1M_ids.txt
> cat sentences1M.txt | awk -F'\t' '{print $2}' > sentences1M_text.txt

"""

from collections import deque
from lxml import etree

import glob
import gzip
import nltk
import os
import re
import sys

GIGAWORD_BASE_DIR='/data/corpora/gigaword_eng_5/data'


def extract_sentences(base_dir, outfilename='sentences.txt'):
    html_parser = etree.HTMLParser()
    with open (outfilename, 'w') as outfile:
        num_sentences = 0
        num_files = 0
        # use files published before 2000 since these are closer to wsj corpus
        for filename in glob.glob(os.path.join(base_dir, "*/*_19*.gz")):  # 1900s
            doc_id = os.path.basename(filename)
            doc_file = gzip.open(filename, 'rb')
            doc_html_tree = etree.parse(doc_file, html_parser)
            paragraph_nodes = doc_html_tree.xpath('//p')
            for pnum, pnode in enumerate(paragraph_nodes):
                sentences = nltk.sent_tokenize(pnode.text.strip())  # FIXME: use Moses instead
                for snum, sent in enumerate(sentences):
                    sent_id = '{}_{}_{}'.format(doc_id, pnum, snum)
                    sent = re.sub('\s+', ' ', sent)  # make sure no tabs or newlines
                    outfile.write('{}\t{}\n'.format(sent_id, sent))
                    num_sentences += 1
                    if num_sentences % 100000 == 0:
                        sys.stderr.write('Wrote {} sentences\n'.format(num_sentences))
            num_files += 1
            if num_files % 10 == 0:
                sys.stderr.write('Processed {} files\n'.format(num_files))

if __name__ == '__main__':
    extract_sentences(GIGAWORD_BASE_DIR)
