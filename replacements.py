from collections import Counter
import argparse
import json
import sys


def build_replacement_map_most_common(anon_filenames, outfilename):
    """Create mapping from predicate to surface form to use in replacement.

    Iterate through one or more *-anon files and count times a particular
    surface form is seen for each predicate. During de-anonymization, the
    postprocessor will use this map to look up the most likely surface form
    for each predicate it is de-anonymizing.

    Note that "most common" value will be affected by repeated training data,
    e.g., if gold data is repeated in training.

    """
    replacement_counts = {}
    for replacements_filename in anon_filenames:
        with open(replacements_filename) as infile:
            for line in infile:
                replacement_dicts = json.loads(line.strip())
                # d is like: [{"ph": "mofy0", "realized": "November", "value": "Nov", "span": [0, 8]}]
                for d in replacement_dicts:
                    # rare unk tokens get special treatment
                    if d['ph'].startswith('_UNK'):
                        continue
                    if 'realized' in d:  # could be missing if two replacements overlapped
                        if d['value'] not in replacement_counts:
                            replacement_counts[d['value']] = Counter()
                        replacement_counts[d['value']].update([d['realized']])
    # when predicate is seen at test time, it will be replaced with the surface form it
    # was most commonly associated with during training
    replacement_map = {}
    for predicate, surface_form_counts in replacement_counts.items():
        most_common_surface_form, count = surface_form_counts.most_common(1)[0]
        if most_common_surface_form != predicate:
            replacement_map[predicate] = most_common_surface_form
    with open(outfilename, 'w') as outfile:
        json.dump(replacement_map, outfile, sort_keys=True, indent=4)
    sys.stderr.write('%d predicate replacements based on most common surface form written to %s\n' %
        (len(replacement_map), outfilename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infiles', nargs='+', help='One or more *-anon.txt files (as written by preprocessing.py)')
    parser.add_argument('--outfile', help='Map from predicates to most common surface form will be written here.')
    args = parser.parse_args()
    build_replacement_map_most_common(args.infiles, args.outfile)
