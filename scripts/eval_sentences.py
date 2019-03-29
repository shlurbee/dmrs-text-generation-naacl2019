"""
Generate sacrebleu scores for individual sentences.

Useful for identifying best and worst matching sentence in a set.

Usage:
python scripts/eval_sentences.py \
    results/silver_exp_results/data_vs6_acc_92.52_ppl_1.69_e22.pt.data_dev_dev.pred.text \
    data/dev/dev-orig.txt results/per_sent_bleu.txt --sort

"""

import argparse
import os
import sacrebleu
import sys


def compare_files(hyp_filename, ref_filename, out_filename, output_sorted=False):
    with open(hyp_filename) as hyp_infile:
        hyp_lines = [line.strip() for line in hyp_infile.readlines()]
    with open(ref_filename) as ref_infile:
        ref_lines = [line.strip() for line in ref_infile.readlines()]
    scored_lines = compare_lines(hyp_lines, ref_lines, return_sorted=output_sorted)
    num_written = 0
    num_exact_matches = 0
    with open(out_filename, 'w') as outfile:
        for score, hyp, ref in scored_lines:
            is_exact_match = hyp == ref
            if is_exact_match:
                num_exact_matches += 1
            outfile.write('{:.4f}\t{}\t{}\t{}\n'.format(
                score, '(exact)' if is_exact_match else '(diff)', hyp, ref))
            num_written += 1
        sys.stderr.write(
            'Wrote {} scored sentences to {}\n'.format(num_written, os.path.abspath(outfile.name)))


def compare_lines(hyp_lines, ref_lines, return_sorted=False):
    scores = [sacrebleu.sentence_bleu(hyp, ref) for hyp, ref in zip(hyp_lines, ref_lines)]
    result = zip(scores, hyp_lines, ref_lines)
    return sorted(result, reverse=True) if return_sorted else result
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp', help='Filename of hypothesis texts to be evaluated.')
    parser.add_argument('ref', help='Filename of reference texts to evaluate against.')
    parser.add_argument('outfile', help='Results will be written here')
    parser.add_argument('--sort', action='store_true', default=False, help='Output sorted by score')
    args = parser.parse_args()
    compare_files(args.hyp, args.ref, args.outfile, output_sorted=args.sort)
