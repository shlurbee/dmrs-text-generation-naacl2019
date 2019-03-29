#!/bin/bash

###
# Experiments training on news domain and testing on other domains.
# NOTE: Requires bash (not sh) for pushd
###

echo "Usage: From root dir run:"
echo "bash scripts/run_domain.sh"

# Counts used to identify rare words (so they can be anonymized) and most
# frequent surface form (so they can be stored in anon-replacements and used
# in de-anonymization) are computed over one copy of gold *news* data plus
# all of silver data. This means de-anonymization of other domains would
# use the most freq surface form found in news domain.

mkdir -p data
VOCAB_FILE=data/vocab-news.txt

preprocess_data() {
    # Set up directories
    mkdir -p data_wsj/train  # gold news
    mkdir -p data_gw/train  # silver (which is also news)
    mkdir -p data_news/train  # gold + silver
    mkdir -p data_wsj/dev
    mkdir -p data_wsj/test
    mkdir -p data_ws/dev
    mkdir -p data_ws/test
    mkdir -p data_brown/dev
    mkdir -p data_brown/test

    # Prep just the news domain (wsj) subset of gold data.
    cat data_penman/train/wsj*.txt > data_penman/train_wsj.txt
    python preprocessing.py data_penman/train_wsj.txt data_wsj/train/train

    # Prep silver (gigaword) data, which is also news domain.
    python preprocessing.py data_gw/parsed.txt data_gw/train/train

    # Create dev and test sets from only news (in-domain) data
    pushd data_penman/dev; ls | grep -P "wsj.*txt" | xargs cat > ../dev_wsj.txt; popd
    pushd data_penman/test; ls | grep -P "wsj.*txt" | xargs cat > ../test_wsj.txt; popd
    python preprocessing.py data_penman/dev_wsj.txt data_wsj/dev/dev
    python preprocessing.py data_penman/test_wsj.txt data_wsj/test/test

    # Create dev and test sets from only wikipedia data
    pushd data_penman/dev; ls | grep -P "ws\d+.txt" | xargs cat > ../dev_ws.txt; popd
    pushd data_penman/test; ls | grep -P "ws\d+.txt" | xargs cat > ../test_ws.txt; popd
    python preprocessing.py data_penman/dev_ws.txt data_ws/dev/dev
    python preprocessing.py data_penman/test_ws.txt data_ws/test/test

    # Create dev and test sets from only brown corpus excerpts
    # Prefixes are: cf*, cg*, ck*, cl*, cm*, cn*, cp*, cr*
    pushd data_penman/dev; ls | grep -P "c[f-r]\d\d.txt" | xargs cat > ../dev_brown.txt; popd
    pushd data_penman/test; ls | grep -P "c[f-r]\d\d.txt" | xargs cat > ../test_brown.txt; popd
    python preprocessing.py data_penman/dev_brown.txt data_brown/dev/dev
    python preprocessing.py data_penman/test_brown.txt data_brown/test/test

    # Combine gold news data with silver news data and use combined file to
    # generate vocab counts.
    cat data_wsj/train/train-tgt.txt data_gw/train/train-tgt.txt > data/combined.txt
    python replace_rare.py vocab --vocabfile $VOCAB_FILE --infile data/combined.txt

    # Use generated vocab counts to anonymize rare unknowns
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_wsj/train/train --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_wsj/dev/dev --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_wsj/test/test --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_gw/train/train --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_ws/dev/dev --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_ws/test/test --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_brown/dev/dev --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_brown/test/test --min_freq 2

    # Remove lines from gold and gold+silver training data that overlap with any dev/test set
    ALL_DEV_TEST=data/combined-dev-test.txt
    cat data_wsj/dev/dev-tgt.txt data_wsj/test/test-tgt.txt data_ws/dev/dev-tgt.txt \
        data_ws/test/test-tgt.txt data_brown/dev/dev-tgt.txt data_brown/test/test-tgt.txt \
        > $ALL_DEV_TEST
    python scripts/remove_overlap.py \
        --train_files data_wsj/train/train-tgt.txt \
        --test_files $ALL_DEV_TEST \
        --blacklist_file data_wsj/blacklist.txt
    python scripts/remove_overlap.py \
        --train_files data_gw/train/train-tgt.txt \
        --test_files $ALL_DEV_TEST \
        --blacklist_file data_gw/blacklist.txt

    # Combine gold and silver data to create news training set
    cat data_gw/train/train-tgt.txt > data_news/train/train-tgt.txt
    cat data_gw/train/train-src.txt > data_news/train/train-src.txt
    cat data_gw/train/train-orig.txt > data_news/train/train-orig.txt
    cat data_gw/train/train-anon.txt > data_news/train/train-anon.txt
    cat data_wsj/train/train-tgt.txt >> data_news/train/train-tgt.txt  # append
    cat data_wsj/train/train-src.txt >> data_news/train/train-src.txt
    cat data_wsj/train/train-orig.txt >> data_news/train/train-orig.txt
    cat data_wsj/train/train-anon.txt >> data_news/train/train-anon.txt

    # Put gold news training data into opennmt format
    rm data_wsj/opennmt.*
    python OpenNMT-py/preprocess.py -train_src data_wsj/train/train-src.txt \
        -train_tgt data_wsj/train/train-tgt.txt \
        -valid_src data_wsj/dev/dev-src.txt \
        -valid_tgt data_wsj/dev/dev-tgt.txt \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data data_wsj/opennmt -report_every 1000

    # Put gold + silver news training data into opennmt format
    rm data_news/opennmt.*
    python OpenNMT-py/preprocess.py -train_src data_news/train/train-src.txt \
        -train_tgt data_news/train/train-tgt.txt \
        -valid_src data_wsj/dev/dev-src.txt \
        -valid_tgt data_wsj/dev/dev-tgt.txt \
        -src_vocab_size 100000 \
        -tgt_vocab_size 100000 \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data data_news/opennmt -report_every 1000

    # Combine all non-news dev sets so "non-news" can be evaluated with one score
    mkdir -p data_non_news/dev
    cat data_ws/dev/dev-src.txt data_brown/dev/dev-src.txt > data_non_news/dev/dev-src.txt
    cat data_ws/dev/dev-tgt.txt data_brown/dev/dev-tgt.txt > data_non_news/dev/dev-tgt.txt
    cat data_ws/dev/dev-orig.txt data_brown/dev/dev-orig.txt > data_non_news/dev/dev-orig.txt
    cat data_ws/dev/dev-anon.txt data_brown/dev/dev-anon.txt > data_non_news/dev/dev-anon.txt
}

train_gold_news() {
    # Train model on wsj subset of news data
    MODEL_VERSION=news_gold  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1
    python OpenNMT-py/train.py -data data_wsj/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_gold_and_silver_news() {
    # Train model on combination of gold and silver news data
    MODEL_VERSION=news_gold_silver  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1
    # Use settings from train_v4 in gold/silver architecture experiments
    python OpenNMT-py/train.py -data data_news/opennmt \
        -layers 2 -dropout 0.3 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

check_unks() {
    FILENAME=$1
    echo "UNKword# token occurrences in $FILENAME (should be > 0)"
    cat $FILENAME | tr ' ' '\n' | grep -P "(^| )UNK.*\d" | sort | uniq -c | sort -n | wc -l
    echo "_UNK# anonymized rare token occurrences in $FILENAME (should be > 0)"
    cat $FILENAME | tr ' ' '\n' | grep -P "_UNK\d" | sort | uniq -c | sort -n
    echo "----------------"
}

run_data_checks() {
    # Print counts of UNK# (rare) and UNKsomeword# (not rare) unknowns in each dataset
    # to verify that replace_rare worked as expected
    check_unks data_wsj/train/train-tgt.txt
    check_unks data_wsj/dev/dev-tgt.txt
    check_unks data_wsj/test/test-tgt.txt
    check_unks data_ws/dev/dev-tgt.txt
    check_unks data_ws/test/test-tgt.txt
    check_unks data_brown/dev/dev-tgt.txt
    check_unks data_brown/test/test-tgt.txt
    check_unks data_gw/train/train-tgt.txt

    echo "Num lines in wsj data (gold data) (should be ~35k)"
    cat data_wsj/train/train-src.txt | wc -l

    echo "Num lines in data_news (should be ~870k)"
    cat data_news/train/train-src.txt | wc -l
}

preprocess_data
run_data_checks

train_gold_news 0
train_gold_and_silver_news 0

