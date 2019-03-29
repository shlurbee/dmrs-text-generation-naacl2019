from collections import Counter
from nltk.tokenize import moses
import argparse
import json
import os
import sys

detokenizer = moses.MosesDetokenizer()  # must match what's used in preprocessing.py


def postprocess(infilename, outfilename, replacements_filename, replacements_map_filename):
    """De-anonymize and detokenize results (reverses what was done by preprocessing.py)

    :infilename: Model predictions (which are tokenized and anonymized)
    :outfilename: Location where detokenized, de-anonymized final text will be written
    :replacements_filename: *-anon.txt file created by preprocessing.py in which each
        line is a json-serialized dict mapping anonymization placeholders to the
        predicate values they replaced in the corresponding input line.
    :replacements_map_filename: file with a mapping from DMRS predicate values of named
        nodes (e.g., named0, card0) to the surface form they should be replaced with

    """
    # Load mapping from predicate values to surface form most often seen in training data
    rmap = json.load(open(replacements_map_filename))
    # Generate list of replacements
    replacements = []
    with open(replacements_filename) as infile:
        for line in infile:
            replacement_dicts = json.loads(line.strip())
            temp = {}
            # If replacements map has a "best" value for the given placeholder, use it,
            # otherwise copy the predicate exactly. Note that for special cases like _UNK0,
            # placeholder is not added to the replacement map so predicate is always copied.
            for d in replacement_dicts:
                temp[d['ph']] = rmap[d['value']] if d['value'] in rmap else d['value']
            replacements.append(temp)
    # De-anonymize and detokenize each line of input file and write to outfile
    with open(infilename) as infile, open(outfilename, 'w') as outfile:
        num_written = 0
        for i, line in enumerate(infile):
            repdict = replacements[i]
            anonymized_tokens = line.strip().split()
            tokens = [repdict.get(t, t) for t in anonymized_tokens]
            s = detokenizer.detokenize(tokens, return_str=True)
            outfile.write('{}\n'.format(s))
            num_written += 1
        sys.stderr.write(
            'Wrote {} deanonymized, detokenized lines to {}\n'.format(
                num_written, os.path.abspath(outfile.name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help='File to post process')
    parser.add_argument('--outfile', help='Post-processed text will be written here')
    parser.add_argument('--replacements', help='Location of token->placeholder replacements created during anonymization')
    parser.add_argument('--replacements_map', help='Location of predicate->surface form replacements from training data')
    args = parser.parse_args()
    postprocess(args.infile, args.outfile, args.replacements, args.replacements_map)
