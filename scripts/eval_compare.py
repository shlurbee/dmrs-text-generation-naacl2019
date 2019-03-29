###
# Compute sacrebleu for just those lines that appear in both NN output and ACE output.
###

import os


def get_overlap_filename(infilename):
    return infilename + '.overlap'


def get_overlapping_lines(eval_filename, nn_filename, ace_filename):
    """Create dev and test sets including only lines for which both methods succeeded."""
    infile_eval = open(eval_filename)
    infile_nn = open(nn_filename)
    infile_ace = open(ace_filename)
    outfile_eval = open(get_overlap_filename(eval_filename), 'w')
    outfile_nn = open(get_overlap_filename(nn_filename), 'w')
    outfile_ace = open(get_overlap_filename(ace_filename), 'w')
    line_tups = zip(infile_eval.readlines(), infile_nn.readlines(), infile_ace.readlines())
    for tup in line_tups:
        is_complete = True
        for source in tup:
            line = source.strip()
            if not line or line == 'BLANK':
                is_complete = False
                break
        if is_complete:
            line_eval, line_nn, line_ace = tup
            outfile_eval.write(line_eval)
            outfile_nn.write(line_nn)
            outfile_ace.write(line_ace)
    for f in [infile_eval, infile_nn, infile_ace, outfile_eval, outfile_nn, outfile_ace]:
        f.close()


if __name__ == '__main__':
    ace_filename = 'results/ace.pred.test.text'
    eval_filename = 'data/test/test-orig.txt'
    nn_filename = 'results/v10_acc_92.76_ppl_1.49_e22.pt.data_test_test.pred.text'
    get_overlapping_lines(eval_filename, nn_filename, ace_filename)
    cmd1 = "cat {} | sacrebleu {} > results/{}.overlap.sacrebleu".format(get_overlap_filename(nn_filename), get_overlap_filename(eval_filename), 'nn')
    cmd2 = "cat {} | sacrebleu {} > results/{}.overlap.sacrebleu".format(get_overlap_filename(ace_filename), get_overlap_filename(eval_filename), 'ace')
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
