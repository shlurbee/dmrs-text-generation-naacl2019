"""
Interface for anon replacements.
Helpful, e.g., for programming different preprocessing steps.

e.g. To get vocab counts from all training data and then do replacements
in both training and dev data:

# anonymize all unknowns
> python scripts/replace_rare.py vocab --vocabfile data_v2_all_anon/vocab.txt --infile data_v2_all_anon/train-tgt.txt
> python scripts/replace_rare.py replace --vocabfile data_v2_all_anon/vocab.txt --infile data_v2_all_anon/train --min_freq 10000 
> python scripts/replace_rare.py replace --vocabfile data_v2_all_anon/vocab.txt --infile data_v2_all_anon/dev --min_freq 10000

# anonymize rare unknowns
> python scripts/replace_rare.py vocab --vocabfile data_v2_rare_anon/vocab.txt --infile data_v2_rare_anon/train-tgt.txt
> python scripts/replace_rare.py replace --vocabfile data_v2_rare_anon/vocab.txt --infile data_v2_rare_anon/train --min_freq 2 
> python scripts/replace_rare.py replace --vocabfile data_v2_rare_anon/vocab.txt --infile data_v2_rare_anon/dev --min_freq 2

"""
import argparse
import preprocessing
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['vocab', 'replace'], help='One of "vocab" or "replace"')
    parser.add_argument('--vocabfile', help='filename where vocab is or will be stored')
    parser.add_argument('--infile', help='name or prefix file where tokens will be counted or replaced (usually a -tgt file)')
    parser.add_argument('--min_freq', type=int, help='vocab tokens that appear fewer than this many times will be replaced')
    args = parser.parse_args()

    if not (args.vocabfile and args.infile):
        sys.stderr.write('--vocabfile and --infile are required arguments\n')
        sys.exit(1)
    if args.mode == 'replace' and not args.min_freq:
        sys.stderr.write('--min_freq is required for replace mode\n')
        sys.exit(1)
    if args.mode == 'vocab':
        sys.stderr.write('Building vocab from anon file {}\n'.format(args.infile))
        preprocessing.build_vocab(args.infile, args.vocabfile)
    elif args.mode == 'replace':
        sys.stderr.write('Using vocab counts in {} to replace tokens in {} with '
                         'freq less than {}\n'.format(args.vocabfile, args.infile, args.min_freq))
        preprocessing.replace_rare_tokens(args.infile, args.vocabfile, args.min_freq)
