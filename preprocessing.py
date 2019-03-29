"""
Convert Penman-serialized graphs to a format that can be used with OpenNMT.

* Anonymizes named nodes by replacing their label with, e.g., "named0", "named1", etc
* Combines attributes into a single node
* Strips some extra characters
* Uses pre-order traversal to output node and edge labels on a single line, space-delimited

Usage:
python preprocessing.py data/sample/sample.txt data/sample

Expected input data format is the output of mrs-to-penman/convert-redwoods.sh, with some
modifications to the parameters file. (See setup.sh and https://github.com/goodmami/mrs-to-penman)

"""

from collections import Counter
import argparse
import io
import json
import os
import re
import shutil
import sys

from nltk.tokenize.moses import MosesTokenizer

from delphin.mrs import xmrs, simplemrs, penman
from penman import PENMANCodec, Triple


class PenmanToLinearCodec(PENMANCodec):
    """Reads PENMAN-serialized DMRS graph and serializes to simplified linear representation.

    Useful when you need a simple sequential representation, e.g., for input to a NN.

    Usage:
    > codec = PenmanToLinearCodec()
    > g = codec.decode(penman_serialized_str)
    > codec.encode(g)

    Example: "The window opened."

    Penman serialization:
    (10002 / _open_v_1
      :lnk "<11:18>"
      :sf PROP
      :tense PAST
      :mood INDICATIVE
      :perf -
      :ARG1-NEQ (10001 / _window_n_1
        :lnk "<4:10>"
        :pers 3
        :num SG
        :ind +
        :RSTR-H-of (10000 / _the_q
          :lnk "<0:3>")))

    Linear serialization:
    ( _open_v_1 sf=PROP|tense=PAST|mood=INDICATIVE|perf=- ARG1-NEQ ( _window_n_1 pers=3|num=SG|ind=+ RSTR-H-of ( _the_q ) ) )

    """
    def _layout(self, g, src, offset, seen):
        seen = seen or {}
        if src in seen:
            return seen[src]
        if (src not in g) or not g.get(src, False) or (src in seen):
            return src
        seen[src] = None
        branches = []
        outedges = self.relation_sort(g[src])
        variables = set(g.keys())
        head = '('
        for t in outedges:
            if t.relation == self.TYPE_REL:
                if t.target is not None:
                    # node types always come first
                    branches = ['{}'.format(t.target)] + branches
                    seen[src] = '{}'.format(t.target)
            else:
                if t.inverted:
                    tgt = t.source
                    rel = self.invert_relation(t.relation)
                else:
                    tgt = t.target
                    rel = t.relation or ''
                branch = self._layout(g, tgt, offset, seen)
                if t.source in variables and t.target in variables:  # edge
                    branches.append('{} {}'.format(rel, branch))
                elif t.relation == 'attributes':  # special node of combined attributes
                    branches.append('{}'.format(branch))
                else:  # simple attribute
                    branches.append('{}={}'.format(rel, branch))
        items = []
        for b in branches:
            if b.startswith('('):
                items.append(b)
            else:
                items.append(' ' + b)
        # branches may contain spaces, so join first then split on space to get all tokens
        linearized = ' '.join([head] + branches + [')'])
        tokens = linearized.split()
        # Tokens with "￨" were already featurized by combine_attributes. For anything else,
        # add empty feature _ to fill requirement that all tokens have same number of features.
        featurized_tokens = [token if u'￨' in token else token + u'￨_' for token in tokens]
        return ' '.join(featurized_tokens)


def get_tgt_filename(prefix):
    return prefix + '-tgt.txt'


def get_src_filename(prefix):
    return prefix + '-src.txt'


def get_anon_filename(prefix):
    return prefix + '-anon.txt'


def get_orig_filename(prefix):
    return prefix + '-orig.txt'


def anonymize_graph(g):
    """Anonymize graph by replacing nodes of certain named types with tokens like "named0".

    Modifies original graph. (Gotcha: accesses private member var)

    Returns dict that can be used to recover the original values.

    """
    replacements = []
    id_counters = {}
    carg_triples = g.attributes(relation='carg')
    # anonymize each instance that has a cargs value, storing the mapping from value to token
    for carg_triple in carg_triples:
        named_triple = g.triples(
            relation='instance', source=carg_triple.source)[0]  # assumes exactly 1
        named_type = named_triple.target.replace("_", "")  # _ causes tokenization issues
        value = carg_triple.target.strip('"')
        # extract char location of the word in original (untokenized) sentence
        span_triple = g.triples(relation="lnk", source=carg_triple.source)[0]
        span = [int(pos) for pos in span_triple.target[2:-2].split(':')]  # '"<5:10>"'
        # create data struct to store mapping of this type and create an id counter
        if named_type not in id_counters:
            id_counters[named_type] = 0
        # generate annonymized token and store it with span it should replace
        placeholder = '{}{}'.format(named_type, id_counters[named_type])
        replacements.append({'ph': placeholder, 'span': span, 'value': value})
        id_counters[named_type] += 1
        new_triple = Triple(
            named_triple.source,
            named_triple.relation,
            placeholder,
            inverted=named_triple.inverted
        )
        # gotcha: accessing private member var
        g._triples.insert(g._triples.index(named_triple), new_triple)
        g._triples.remove(named_triple)
        g._triples.remove(carg_triple)
    return replacements


def combine_attributes(g):
    """Group all attribute nodes into one.

    Attribute list is normalized by uppercasing the value and sorting
    the list by attribute name. Concatenated attributes are appended
    to the instance (predicate) target value so OpenNMT will interpret
    them as word features.

    Note that OpenNMT expects all tokens to have the same number of word
    features, but only predicate tokens have attributes, so an extra step
    will be required to make sure all tokens have a feature. (See _layout
    in PenmanToLinearCodec)

    """
    for variable in g.variables():
        old_attributes = [
            attr for attr in g.attributes(source=variable) if attr.relation != 'instance'
        ]
        new_targets = []
        for old_attr in old_attributes:
            old_relation = old_attr.relation
            old_target = old_attr.target.upper() if isinstance(old_attr.target, str) else old_attr.target
            # don't store span info (only needed for anonymization) or untensed (doesn't provide much info)
            if old_relation != 'lnk' and (old_relation, old_target) != ('tense', 'UNTENSED'):
                new_targets.append('{}={}'.format(old_relation, old_target))
            g._triples.remove(old_attr)
        if new_targets:
            attr_features = '|'.join(sorted(new_targets))  # sort by attribute name
            instance = g.attributes(source=variable, relation='instance')[0]
            new_instance = Triple(
                source=instance.source,
                relation=instance.relation,
                target=instance.target + '￨' + attr_features  # N.B. '￨' not '|'
            )
            g._triples.insert(g._triples.index(instance), new_instance)
            g._triples.remove(instance)


def load_serialized_from_file(infilename):
    """Read serialized graphs from a file.

    Stores concatenated comment lines (lines starting with "#") as the graph label.

    Returns list of (label, serialized_graph) tuples.

    """
    serialized = []
    with open(infilename) as infile:
        heading = ''
        partial = []
        for line in infile:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if partial:
                    serialized_graph = ' ' .join(partial)
                    serialized.append((heading, serialized_graph))
                    partial = []
                    heading = line.strip()
                else:
                    heading = heading + ' ' +  line.strip()
            else:
                partial.append(line.strip())
        serialized_graph = ' '.join(partial)
        serialized.append((heading, serialized_graph))
        print('Loaded {} serialized graphs from {}'.format(len(serialized),
                                                           os.path.abspath(infile.name)))
    return serialized


def preprocess_penman(serialized):
    """Given a Penman-serialized graph, simplify, anonymize, and linearize it.

    Anonymization replaces nodes of specific classes with placeholders like named0, named1
    and stores a mapping that can be used to recover original values.

    Returns tuple of (preprocessed_graph, anonymization_mapping)

    """
    codec = preprocess_penman.codec
    g = codec.decode(serialized)
    anon_map = anonymize_graph(g)
    combine_attributes(g)
    linearized = codec.encode(g)
    return linearized, anon_map
preprocess_penman.codec = PenmanToLinearCodec()


def preprocess_sentence(sentence, anon_map):
    """Tokenize sentence and replace known tokens with placeholders."""
    # spans correspond to detokenized position so do anonymization before tokenization
    to_replace = sorted(anon_map, key=lambda x: x['span'], reverse=True)
    start, end = [sys.maxsize - 1, sys.maxsize]
    for i, anon_dict in enumerate(to_replace):
        prev_start, prev_end = start, end
        start, end = anon_dict['span']
        # Special case multi-part entities like number sequences or compounds (555-5555 or Ekorråa/Ikornåa) -
        # the lnk field of each word in a compound is set to span of the full compound and we don't want to
        # accidentally learn that every instance of "555" should be realized as "555-5555".
        if sorted([prev_end, prev_start, end, start], reverse=True) != [prev_end, prev_start, end, start]:
            # Omit "realized" field for all components. Postprocessing will fall back to using predicate.
            del to_replace[i - 1]['realized']
            # Don't replace anything when inserting placeholder (since full span has already been replaced.)
            sentence = sentence[:start] + anon_dict['ph'] + sentence[start:]
            sys.stderr.write('Handled overlapping replacement: {}\n'.format(anon_dict))
            continue
        # If named node contains a hyphen, do search-replace within span
        if anon_dict['value'].endswith('-'):
            full_span = sentence[start:end]
            updated_span = full_span.replace(anon_dict['value'], anon_dict['ph'] + " ")
            if full_span == updated_span:
                sys.stderr.write('Substr with hyphen ({}) not found in span: {}\n'.format(anon_dict['value'], full_span))
            sentence = sentence[:start] + updated_span + sentence[end:]
            continue
        # Otherwise, handle common replacement case
        # adjust span so it doesn't include punctuation
        _adjust_span_boundaries(sentence, anon_dict)
        start, end = anon_dict['span']
        # strip wikipedia links from realized text (later preprocessing removes them and this should match)
        anon_dict['realized'] = re.sub("(\[|\])", "", sentence[start:end])
        # replace contents of adjusted span with placeholder
        sentence = sentence[:start] + anon_dict['ph'] + sentence[end:]
    # clean up sentence (same normalization must be applied the original used in eval)
    sentence = _normalize_sentence(sentence)
    # tokenize
    raw_tokens = preprocess_sentence.tokenizer.tokenize(sentence, escape=False)
    return ' '.join(raw_tokens)
preprocess_sentence.tokenizer = MosesTokenizer()  # must match what's used in postprocessing


def _adjust_span_boundaries(sentence, anon_dict):
    """Shrinks span boundary so it doesn't include punctuation.

    Gotcha: assumes adjustments are being done in order from last to first, so
    one adjustment doesn't affect the next.
    """
    start, end = anon_dict['span']
    # adjust beginning of replacement span to exclude punc
    while start < end and sentence[start] in {'(', '[', '"', '`', "'"}:
        if sentence[start] == "'":  # allow single quote but not double
            if end - start > 1 and sentence[start + 1] == "'":
                start += 2
                continue
            else:
                break
        start += 1
    # adjust end of replacement span to exclude punctuation
    while end > start and sentence[end - 1] in {'.', ',', '!', '?', '"', '`', "'", '(', ')', '[', ']', ';', ':'}:
        # don't remove period after acronym
        if end - start > 1 and re.match(r'[A-Z]', sentence[end - 2]) and sentence[end - 1] == '.':
            break
        # walk back until final char is not punc
        end -= 1
    # update span info
    anon_dict['span'][0] = start
    anon_dict['span'][1] = end


def _normalize_sentence(sentence):
    """Make formatting consistent."""
    # remove wikipedia link brackets
    sentence = re.sub("(\[|\])", "", sentence)
    # normalize double quotes
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


def create_parallel_files(infilename, outfile_prefix, output_blank_for_failure=False):
    """Convert Penman serialized graphs to format that can be used for training.

    Reads Penman-serialized graphs from infilename, where infile was created by
    running mrs-to-penman.py (or has the same format.)

    Writes linear serializations to {outfile_prefix}-src.txt, corresponding
    tokenized sentences to {outfile_prefix}-tgt.txt, and anonymization map
    (map of placeholders to original strings) to {outfile_prefix}-anon.txt

    """
    data = load_serialized_from_file(infilename)
    sys.stderr.write('Deserializing and processing {} graphs.'.format(len(data)))
    sys.stderr.write('Using Moses tokenization from the nltk package.\n')
    with io.open(get_src_filename(outfile_prefix), 'w', encoding='utf8') as outfile_src, \
         io.open(get_tgt_filename(outfile_prefix), 'w', encoding='utf8') as outfile_tgt, \
         io.open(get_anon_filename(outfile_prefix), 'w', encoding='utf8') as outfile_anon, \
         io.open(get_orig_filename(outfile_prefix), 'w', encoding='utf8') as outfile_orig:
        sys.stderr.write(
            'Writing serialized graphs to {}.\n'.format(os.path.abspath(outfile_src.name)))
        sys.stderr.write(
            'Writing tokenized sentences to {}.\n'.format(os.path.abspath(outfile_tgt.name)))
        sys.stderr.write(
            'Writing anonymization map to {}.\n'.format(os.path.abspath(outfile_anon.name)))
        sys.stderr.write(
            'Writing original sentences to {}.\n'.format(os.path.abspath(outfile_orig.name)))
        num_written = 0
        num_skipped = 0
        for label, penman_serialized in data:
            try:
                # treat unknowns same as named tokens so they'll be copied exactly
                penman_serialized = re.sub(r'_([^\s]+)\/(.*?_unknown)', r'UNK\1 :carg "\1"', penman_serialized)
                # simplify, linearize, and anonymize graphs
                linearized, anon_map = preprocess_penman(penman_serialized)
                # tokenize and anonymize sentences (assumes last comment is sentence)
                sentence = label.split('# ::snt ')[-1].strip()
                outfile_tgt.write('{}\n'.format(preprocess_sentence(sentence, anon_map)))  # modifies anon_map
                outfile_src.write('{}\n'.format(linearized))
                # store anonymization info for use in  postprocessing
                outfile_anon.write('{}\n'.format(json.dumps(anon_map)))
                # also write original sentence, which will be compared against during eval
                outfile_orig.write('{}\n'.format(_normalize_sentence(sentence)))
                num_written += 1
            except Exception as e:
                sys.stderr.write(
                    'Deserialization failed for {}, skipping. Error was: {}\n'.format(label, e))
                num_skipped += 1
                if output_blank_for_failure:
                    outfile_src.write('\n')
                    outfile_tgt.write('\n')
                    outfile_anon.write('[]\n')
                    outfile_orig.write('\n')
        ratio_skipped = float(num_skipped) / num_written
        sys.stderr.write(
            'Linearized {} graphs. Skipped {} due to deserialization errors ({}).\n'.format(
                num_written, num_skipped, ratio_skipped))

def build_vocab(target_filename, vocab_filename):
    vocab = Counter()
    with open(target_filename) as infile:
        for line in infile:
            vocab.update(line.strip().split())
    sorted_vocab = vocab.most_common(1000000)
    with open(vocab_filename, 'w') as outfile:
        for word, freq in sorted_vocab:
            outfile.write('{}\t{}\n'.format(word, freq))
        sys.stderr.write(
            '{} vocab words and their frequencies written to {}\n'.format(
            len(vocab), os.path.abspath(outfile.name)))


def load_vocab(vocab_filename, min_word_freq=0):
    vocab = {}
    with open(vocab_filename) as infile:
        for line in infile:
            word, freq_str = line.strip().split()
            freq = int(freq_str)
            if freq >= min_word_freq:
                vocab[word] = freq
    return vocab


def replace_rare_tokens(parallel_files_prefix, vocab_filename, min_word_freq=2):
    # matching updates will need to be made in all parallel files
    anon_filename = get_anon_filename(parallel_files_prefix)
    src_filename = get_src_filename(parallel_files_prefix)
    tgt_filename = get_tgt_filename(parallel_files_prefix)
    # load list of valid vocab words
    vocab = load_vocab(vocab_filename, min_word_freq=min_word_freq)
    # make copies of files that will be modified
    backup_anon_filename = anon_filename + '.full'
    backup_src_filename = src_filename + '.full'
    backup_tgt_filename = tgt_filename + '.full'
    sys.stderr.write('Backing up anon file {} to {}\n'.format(anon_filename, backup_anon_filename))
    shutil.copyfile(anon_filename, backup_anon_filename)
    sys.stderr.write('Backing up src file {} to {}\n'.format(src_filename, backup_src_filename))
    shutil.copyfile(src_filename, backup_src_filename)
    sys.stderr.write('Backing up tgt file {} to {}\n'.format(tgt_filename, backup_tgt_filename))
    shutil.copyfile(tgt_filename, backup_tgt_filename)
    # iterate through lines and replace rare UNK___# placeholders with simpler UNK# in all files
    num_replaced = 0
    num_lines_processed = 0
    with open(backup_anon_filename) as anon_file_orig, \
         open(anon_filename, 'w') as anon_file_new, \
         open(backup_tgt_filename) as tgt_file_orig, \
         open(tgt_filename, 'w') as tgt_file_new, \
         open(backup_src_filename) as src_file_orig, \
         open(src_filename, 'w') as src_file_new:
        for i, line in enumerate(tgt_file_orig):
            tgt_line = line.strip()
            tokens_orig = tgt_line.split()
            anon_serialized = anon_file_orig.readline()
            anon_dicts = json.loads(anon_serialized)
            src_line = src_file_orig.readline().strip()
            ph_num = 0
            for token in tokens_orig:
                if token not in vocab:
                    # replace rare unknown token like UNKfoxes0 with _UNK0
                    if token.startswith('UNK'):
                        old_placeholder = token
                        new_placeholder = '_UNK{}'.format(ph_num)
                        ph_num += 1
                        # update anon dict
                        for d in anon_dicts:
                            if d['ph'] == old_placeholder:
                                d['ph'] = new_placeholder
                        # update target line
                        # note: assumes placeholder is unique enough to never be substring of another token
                        tgt_line = tgt_line.replace(old_placeholder, new_placeholder)
                        # update src line
                        src_line = src_line.replace(old_placeholder, new_placeholder)
                        # update placeholder counter
                        num_replaced += 1
            anon_file_new.write(json.dumps(anon_dicts) + '\n')
            src_file_new.write(src_line + '\n')
            tgt_file_new.write(tgt_line + '\n')
            num_lines_processed += 1
            if num_lines_processed % 50000 == 0:
                sys.stderr.write('processed {} lines\n'.format(num_lines_processed))
    sys.stderr.write('Replaced {} rare placeholder tokens\n'.format(num_replaced))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Name of Penman-serialized graph file to preprocess')
    parser.add_argument(
        'outfile_prefix',
        help='Output files will be named using this prefix. Please make sure dir already exists.'
    )
    parser.add_argument(
        '--with_blanks', action='store_true',
        help='If True, output blank line when deserialization fails. (Useful for preserving line '
        'positions so output from different sources can be compared.)')
    args = parser.parse_args()
    create_parallel_files(args.infile, args.outfile_prefix, output_blank_for_failure=args.with_blanks)

