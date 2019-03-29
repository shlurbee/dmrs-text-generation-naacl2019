#!/bin/bash

###
# Extracts sentences that were used in Neural AMR evaluation (Konstas, 2017)
# and parses them to MRS so BLEU scores can be compared on the same data.
#
# Run from the amr-to-mrs directory:
# > cd amr-to-mrs
# > bash run.sh
###

# Parser/grammar versions
ACE_BINARY=ace-0.9.25
ERG_GRAMMAR=erg-1214-x86-64-0.9.25.dat

# Print commands when executing them
set -o xtrace

# Download files needed to parse text to DMRS
setup_MRS_parser() {
    mkdir -p ace
    cd ace
    wget -O $ACE_BINARY.tar.gz  http://sweaglesw.org/linguistics/ace/download/$ACE_BINARY-x86-64.tar.gz
    tar -zxvf $ACE_BINARY.tar.gz
    wget -O $ERG_GRAMMAR.bz2 http://sweaglesw.org/linguistics/ace/download/$ERG_GRAMMAR.bz2
    bzip2 -dk $ERG_GRAMMAR.bz2
    cd ..
}

# Unpack test sentences and parse them to DMRS
parse_NeuralAMR_test_data() {
	# Unpack data and combine files
	unzip amr-dev-test.zip
	cat test/* > test.amr.txt

	# Extract sentences from penman formatted data into new file, one sentence per line
    cat test.amr.txt | grep "::snt " | sed s/"# ::snt "// > test.sents.txt

	# Parse test sentences to DMRS and output in penman format
	python3 ../mrs-to-penman/mrs_to_penman.py \
		--ace-binary ace/$ACE_BINARY/ace \
		-g ace/$ERG_GRAMMAR \
		-i test.sents.txt \
		-p ../mrs-to-penman/parameters.json \
		> test.sents.txt.mrs
}

setup_MRS_parser
parse_NeuralAMR_test_data
