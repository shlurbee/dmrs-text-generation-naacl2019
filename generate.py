"""
Use parser to generate text from MRS representation.

If you want to generate using the NN model, you probably want scripts/run.sh instead

Usage (if using default args):
> python generate.py

Usage (if not using defaults):
> python generate.py \
   --grammar mrs-to-penman/erg-1214-x86-64-0.9.25.dat \
   --ace_binary mrs-to-penman/ace-0.9.25/ace

Note: you currently need to change the list of profiles in code to switch
from dev to test.

"""
import sys
import os
import re
import argparse
import json

from delphin.interfaces import ace
from delphin.mrs import xmrs, simplemrs, penman
from delphin.mrs.components import var_sort
from delphin import itsdb

# Parser hangs on these ids
BAD_IDS = set(['1000009000880'])

def run(input_dirs, args, output_filename):
    """Run for each input dir."""
    with open(output_filename, 'w') as outfile:
        for profile in input_dirs:
            process(read_profile(profile, args), args, outfile)


def run_debug(input_dirs, args, output_filename):
    """Print ids and sentences from each profile, for troubleshooting."""
    with open(output_filename, 'w') as outfile:
        for profile in input_dirs:
            outfile.write('{}\n'.format(profile))
            for item_id, snt, mrss in read_profile(profile, args):
                outfile.write('{}\n'.format(snt))


def process(items, args, outfile):
    """Generate text from MRS."""
    i = 0
    for item_id, snt, mrss in items:
        print('# ::id {}\n# ::snt {}'.format(item_id, snt))
        try:
            if mrss is None:
                raise ValueError("mrss was None")
            if item_id in BAD_IDS:
                raise ValueError("Skipping problematic id {}".format(item_id))
            simple_mrs = simplemrs.serialize(mrss)
            response = ace.generate(args.grammar, simple_mrs, executable=args.ace_binary, cmdargs=['-n', '5'])
            outfile.write(response.result(0)['surface'] + '\n')
        except Exception as ex:
            outfile.write('\n')
            print('Item {}\t{}'.format(item_id, str(ex)), file=sys.stderr)
        print()
        i += 1
        if i % 100 == 0:
            outfile.flush()


def read_profile(f, args):
    """Load MRS from tsdb. (Copied from mrs_to_penman.py)"""
    p = itsdb.ItsdbProfile(f)
    inputs = dict((r['i-id'], r['i-input']) for r in p.read_table('item'))
    cur_id, mrss = None, []
    for row in p.join('parse', 'result'):
        try:
            mrs = simplemrs.loads_one(row['result:mrs'])

            if cur_id is None:
                cur_id = row['parse:i-id']

            if cur_id == row['parse:i-id']:
                mrss.append(mrs)
            else:
                yield (cur_id, inputs[cur_id], mrss)
                cur_id, mrss = row['parse:i-id'], [mrs]
        except Exception as ex:
            print('Could not read profile from file {}, row: {}\n'.format(f, row))
            mrss = None  # error case, must be handled by caller

    yield (cur_id, inputs[cur_id], mrss)


def get_test_profiles():
    """Copied from convert_redwoods.sh"""
    return [
        "cb",
        "cf04",
        "cf06",
        "cf10",
        "cf21",
        "cg07",
        "cg11",
        "cg21",
        "cg25",
        "cg32",
        "cg35",
        "ck11",
        "ck17",
        "cl05",
        "cl14",
        "cm04",
        "cn03",
        "cn10",
        "cp15",
        "cp26",
        "cr09",
        "ecpr",
        "jhk",
        "jhu",
        "petet",
        "psk",
        "psu",
        "rondane",
        "tgk",
        "tgu",
        "vm32",
        "ws13",
        "ws214",
        "wsj21a",
        "wsj21b",
        "wsj21c",
        "wsj21d",
    ]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g', '--grammar', default='mrs-to-penman/erg-1214-x86-64-0.9.25.dat',
                           help='path to a grammar file compiled with ACE')
    argparser.add_argument('--ace-binary', default='mrs-to-penman/ace-0.9.25/ace',
                           help='path to the ACE binary (default: mrs-to-penman/ace-0.9.25/ace)')
    args = argparser.parse_args()
    #dev_profiles = ["ecpa", "jh5", "tg2", "ws12", "wsj20a", "wsj20b", "wsj20c", "wsj20d", "wsj20e"]
    test_profiles = get_test_profiles()
    input_dirs = ["mrs-to-penman/profiles/%s" % prof for prof in test_profiles]
    output_filename = "results/ace.pred.test.text"
    run(input_dirs, args, output_filename)
    #generate_parallel_text(input_dirs, args, 'data/test/ace-test-orig.debug.txt')
