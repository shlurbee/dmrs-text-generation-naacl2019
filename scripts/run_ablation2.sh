#!/bin/bash

###
# Script to train on data with all semantic features (node feats and edge feats) removed.
# Similar to run_ablation.sh, but this script removes all node and edge features at once
#   instead of a group at a time, and trains on silver data as well as gold.
# The motivation was to get a feature set more similar to what's available in AMR for
#   use in amr comparison. (see scripts/run_amr_compare.sh)
###

echo "Usage: From root dir run:"
echo "sh scripts/run_ablation2.sh"

# Regex for concatenated attributes attached to predicate with ￨
RE_REMOVE_ALL_NODE_FEAT="s/￨.*?( |$)/ /g"

# Regex for features embedded in edge labels. (Note HEQ before H to prevent partial match)
RE_REMOVE_EDGE_FEAT="s/([A-Z0-9\-])-(HEQ|H|EQ|NEQ)/\1/g"

prep_data() {
    # Make sure expected directories exist
    mkdir -p data/train
    mkdir -p data/dev
    mkdir -p data/test
    mkdir -p data_gw/train

    # Linearize train, dev, test data
    python preprocessing.py data_penman/train.txt data/train/train
    python preprocessing.py data_penman/dev.txt data/dev/dev
    python preprocessing.py --with_blanks data_penman/test.txt data/test/test
    python preprocessing.py data_penman/dev_wsj.txt data_wsj/dev/dev
    python preprocessing.py --with_blanks data_penman/test_wsj.txt data_wsj/test/test
    python preprocessing.py data_gw/parsed.txt data_gw/train/train
    echo 'done linearizing'

    # Compute word counts over one copy of the gold training data and one copy of
    # all the silver data.
    VOCAB_FILE=data/vocab-combined.txt
    cat data/train/train-tgt.txt data_gw/train/train-tgt.txt > data/combined.txt
    python replace_rare.py vocab --vocabfile $VOCAB_FILE --infile data/combined.txt

    # Use computed word counts to identify tokens that appear only once in the
    # gold + silver data and replace them with UNK# placeholders.
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data/train/train --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data/dev/dev --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data/test/test --min_freq 2
    python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_gw/train/train --min_freq 2
    echo 'done anonymizing rare tokens'

    # Overwrite lines in gold and silver training data that overlap with dev set,
    # to avoid leaking answers to the model. The script first creates a blacklist
    # of sentences that appear in both, then rewrites the training data, omitting
    # lines found in the blacklist.
    python scripts/remove_overlap.py \
        --train_files data/train/train-tgt.txt \
        --test_files data/dev/dev-tgt.txt data/test/test-tgt.txt \
        --blacklist_file data/blacklist.txt
    python scripts/remove_overlap.py \
        --train_files data_gw/train/train-tgt.txt \
        --test_files data/dev/dev-tgt.txt data/test/test-tgt.txt \
        --blacklist_file data/blacklist_gw.txt

    # Create file mapping anonymization tokens to most common surface form, computed
    # across one copy of gold + one copy of silver data. The --outfile value here will
    # be passed as an argument to the eval.sh script and is used during de-anonymization.
    python replacements.py \
        --infiles data/train/train-anon.txt data_gw/train/train-anon.txt \
        --outfile data/anon-replacements.json
}

preprocess_vs() {
    NUM_GOLD_COPIES=$1
    NUM_SILVER_EXAMPLES=$2
    DATA_DIR=data_vs$NUM_GOLD_COPIES
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n $NUM_SILVER_EXAMPLES data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n $NUM_SILVER_EXAMPLES data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n $NUM_SILVER_EXAMPLES data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n $NUM_SILVER_EXAMPLES data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # append gold data. (assumes opennmt will shuffle)
    i=0
    while [ $i -lt  $NUM_GOLD_COPIES ]; do
        echo $i
        cat data/train/train-tgt.txt >> $DATA_DIR/train-tgt.txt
        cat data/train/train-src.txt >> $DATA_DIR/train-src.txt
        cat data/train/train-orig.txt >> $DATA_DIR/train-orig.txt
        cat data/train/train-anon.txt >> $DATA_DIR/train-anon.txt
        i=`expr $i + 1`
    done

    # delete any leftover data so new files can be written
    rm $DATA_DIR/opennmt.*

    # preprocess data files with all semantic features included
    python OpenNMT-py/preprocess.py -train_src $DATA_DIR/train-src.txt \
        -train_tgt $DATA_DIR/train-tgt.txt \
        -valid_src data/dev/dev-src.txt \
        -valid_tgt data/dev/dev-tgt.txt \
        -src_vocab_size 100000 \
        -tgt_vocab_size 100000 \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data $DATA_DIR/opennmt -report_every 1000

    # create an anon replacement map from the training data
    python replacements.py \
        --infiles data/train/train-anon.txt data_gw/train/train-anon.txt \
        --outfile data/anon-replacements.json
}

# Remove all node and edge features from the combined gold+silver training dataset
prep_ablate_sem_feats() {
    RAW_DATA_DIR=$1  #data_vs6
    DATA_DIR="$RAW_DATA_DIR"_ablate_sem_feats
    mkdir -p $DATA_DIR
    rm $DATA_DIR/opennmt.*

    # copy gold+silver training data to dir, removing node and edge features from source file
    cat $RAW_DATA_DIR/train-tgt.txt > $DATA_DIR/train-tgt.txt
    cat $RAW_DATA_DIR/train-src.txt | perl -p -e "$RE_REMOVE_ALL_NODE_FEAT" | perl -p -e "$RE_REMOVE_EDGE_FEAT" > $DATA_DIR/train-src.txt
    cat $RAW_DATA_DIR/train-orig.txt > $DATA_DIR/train-orig.txt
    cat $RAW_DATA_DIR/train-anon.txt > $DATA_DIR/train-anon.txt

    # apply same preprocessing to dev data
    cat data/dev/dev-tgt.txt > $DATA_DIR/dev-tgt.txt
    cat data/dev/dev-src.txt | perl -p -e "$RE_REMOVE_ALL_NODE_FEAT" | perl -p -e "$RE_REMOVE_EDGE_FEAT" > $DATA_DIR/dev-src.txt
    cat data/dev/dev-orig.txt > $DATA_DIR/dev-orig.txt
    cat data/dev/dev-anon.txt > $DATA_DIR/dev-anon.txt

    # create opennmt training set
    python OpenNMT-py/preprocess.py -train_src $DATA_DIR/train-src.txt \
        -train_tgt $DATA_DIR/train-tgt.txt \
        -valid_src $DATA_DIR/dev-src.txt \
        -valid_tgt $DATA_DIR/dev-tgt.txt \
        -src_vocab_size 100000 \
        -tgt_vocab_size 100000 \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data $DATA_DIR/opennmt -report_every 1000

}

run_data_checks() {
    echo "Original data set should have ~70k lines"
    cat data/train/train-src.txt | wc -l
    echo "Combined and ablated data sets should each have ~1.2M lines"
    cat data_vs6/train-src.txt | wc -l
    cat data_vs6_ablate_sem_feats/train-src.txt | wc -l
    echo "Gold data set and ablated gold data set should have ~70k lines"
    cat data_vs1/train-src.txt | wc -l
    cat data_vs1_ablate_sem_feats/train-src.txt | wc -l
}


train() {
    DATA_DIR=$1  # used in naming model files
    MODEL_PREFIX=models/$DATA_DIR
    GPU_ID=$2

    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data $DATA_DIR/opennmt \
        -layers 2 -dropout 0.3 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$DATA_DIR.log"
}

prep_data

# generate same gold+silver training data that was used in silver experiments
# version vs6, but with node and edge features removed
preprocess_vs 6 850000
prep_ablate_sem_feats data_vs6

# generate gold-only training data with node and edge features removed
preprocess_vs 1 0  # gold data only
prep_ablate_sem_feats data_vs1

# make sure each training data set has the expected number of lines
run_data_checks

# train
train data_vs6_ablate_sem_feats 1
train data_vs1_ablate_sem_feats 2

