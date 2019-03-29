"""
Helpers for removing lines from training data set that overlap with test data.

Apply after preprocessing.py is complete so comparisons are done on anonymized lines.

See print_parallel_file_instructions() for expected training filename convention.

Example usage one training set and one test set:
> python remove_overlap.py \
    --train_files data/train/train-tgt.txt \
    --test_files data/dev/dev-tgt.txt \
    --blacklist_file data/dev/blacklist.txt

Example usage multiple training sets and test sets:
> python remove_overlap.py \
    --train_files data/train/train-tgt.txt data_gw/train/train-tgt.txt \
    --test_files data/dev/dev-tgt.txt data/test/test-tgt.txt \
    --blacklist_file data/blacklist.txt

"""

import argparse
import os
import sys

DEBUG = False
PARALLEL_FILE_SUFFIXES = ['src.txt', 'tgt.txt', 'anon.txt', 'orig.txt']


def find_overlapping_lines(filename1, filename2, blacklist_filename='blacklist.txt', append=False):
    """Find lines that appear in both filename1 and filename2.

    Useful, e.g., for identifying lines that need to be removed from the training data because
    they also appear in the test data.

    :filename1: smaller file (assumes all lines can fit in memory as a set)
    :filename2: larger file
    :blacklist_filename: common lines will be written here
    :append: if True, append to blacklist file instead of overwriting it

    """
    num_overlapping = 0
    num_lines = 0
    lines1 = set()
    with open(filename1) as infile1:
        for line1 in infile1:
            line1 = line1.strip()
            if line1:
                lines1.add(line1)
    with open(filename2) as infile2, open(blacklist_filename, 'a' if append else 'w') as blacklist:
        for line2 in infile2:
            line2 = line2.strip()
            if line2 and (line2 in lines1):
                num_overlapping += 1
                if DEBUG:
                    print("Overlaps: {} of {} - {}\n".format(num_overlapping, num_lines, line2))
                blacklist.write(line2 + '\n')
            num_lines += 1
            if num_lines % 50000 == 0:
                print("Found {} of {} overlapping\n".format(num_overlapping, num_lines))
    print("Found {} of {} overlapping between {} and {}\n".format(num_overlapping, num_lines, filename1, filename2))


def apply_blacklist(fileprefix, blacklist_filename, output_blank=False):
    """Replaces blacklisted lines with blank lines in parallel files.

    :fileprefix: prefix of parallel files (see preprocessing.py)
    :blacklist_filename: location of blacklist, usually created with find_overlapping_lines()
    :output_blank: if True, output a blank line in place of blacklisted line, otherwise
        skip the line and output nothing.

    """
    src_filename = fileprefix + '-src.txt'
    tgt_filename = fileprefix + '-tgt.txt'
    orig_filename = fileprefix + '-orig.txt'
    anon_filename = fileprefix + '-anon.txt'
    blacklist = set(open(blacklist_filename).readlines())
    # identify line numbers of sentences that should be removed
    bad_line_nums = set()
    with open(tgt_filename) as infile:
        for i, line in enumerate(infile):
            if line in blacklist:
                bad_line_nums.add(i)
    # replace bad line numbers with blank lines in all parallel files
    for filename in [src_filename, tgt_filename, orig_filename, anon_filename]:
        with open(filename, 'r') as infile:
            lines = infile.readlines()
        with open(filename, 'w') as outfile:
            for i, line in enumerate(lines):
                if i in bad_line_nums:
                    if DEBUG:
                        sys.stderr.write('Removing line {} from {}: {}\n'.format(i, filename, line))
                    if output_blank:
                        outfile.write('[]\n' if filename == anon_filename else '\n')
                else:
                    outfile.write(line)


def print_parallel_file_instructions():
    print("""
This script assumes parallel training files are named with a common prefix like so:
1. myfile-src.txt (input)
2. myfile-tgt.txt (expected model output)
3. myfile-anon.txt (anonymization dicts)
4. myfile-orig.txt (original/gold sentences)
    """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Removes lines from training files if they overlap with test file.")
    required_named = parser.add_argument_group('Required named arguments')
    required_named.add_argument(
        '--test_files',
        nargs='+',
        required=True,
        help='Test data file. Any line that appears in this file should not appear in training data.'
    )
    required_named.add_argument(
        '--train_files',
        nargs='+',
        required=True,
        help='One or more training data files. Any line from the test file that appears in one of these '
             'will be added to the blacklist and removed from all four parallel training files.'
    )
    required_named.add_argument(
        '--blacklist_file',
        required=True,
        help='File where lines that appear in both test and train data will be stored.'
    )
    args = parser.parse_args()

    print('Preparing to remove lines that appear in test data from training set...')

    # check that parallel files have expected name format
    for train_file in args.train_files:
        print('Checking parallel files for {}.'.format(train_file))
        prefix = train_file.rsplit('-', 1)[0]
        for suffix in PARALLEL_FILE_SUFFIXES:
            filename = '-'.join([prefix, suffix])
            if os.path.exists(filename):
                print('Found {}'.format(filename))
            else:
                print('\nERROR: Required file {} not found.'.format(filename))
                print_parallel_file_instructions()
                exit(1)

    # find lines that appear in both test and train data and store them in blacklist file
    for i, train_file in enumerate(args.train_files):
        for test_file in args.test_files:
            append = True if i == 0 else False  # create new list on first file, append for others
            print('Checking {} for lines that overlap with {}'.format(train_file, test_file))
            find_overlapping_lines(test_file, train_file, args.blacklist_file, append=append)

    # for any -tgt line that was found in the test set, remove it from parallel files
    for train_file in args.train_files:
        train_prefix = train_file.rsplit('-', 1)[0]
        apply_blacklist(train_prefix, args.blacklist_file)

