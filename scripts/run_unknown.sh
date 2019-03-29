#!/bin/bash

###
# Helpers to prepare and train with different methods of handling unknowns.
# Note these experiments are run on gold data only.
# Please run from repo root.
###

echo "Usage: From root dir run:"
echo "sh scripts/run_unknown.sh"

preprocess_include_all_unknowns() {
    # This is the default behavior in the main preprocessing script, but in
    # this case we operate only on gold data instead of gold+silver.
    # If a token is not recognized by the grammar, its predicate token (which
    # looks like "xx_y_unknown") is replaced by the surface form of the word
    # as extracted from the carg field.
    # (In reality, if the surface form is "cat" we replace it with "UNKcat" to
    # make it easier to track parser-unknowns in final output, but the effect
    # on the model is the same since there is a unique token for each word.)
    DATA_DIR=data_unk_include
    mkdir -p $DATA_DIR

    sh scripts/unknowns/prep_include_all_unknowns.sh $DATA_DIR

    # preprocess data files
    rm $DATA_DIR/opennmt.*
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

preprocess_anonymize_all_unknowns() {
    # For all tokens not recognized by the grammar, extracts the surface form
    # from the carg field, then anonymizes the input token by replacing it with
    # generic tokens like UNK0, UNK1. For a given sentence, the surface form
    # associated with the generic token will be remembered and copied back into
    # place during post-processing.
    DATA_DIR=data_unk_all
    mkdir -p $DATA_DIR

    sh scripts/unknowns/prep_anonymize_all_unknowns.sh $DATA_DIR

    # preprocess data files
    rm $DATA_DIR/opennmt.*
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

preprocess_anonymize_rare_unknowns() {
    # Tokens not recognized by the parser are handled in one of two ways:
    # (1) If the token appears more than once in the training data, its
    #   predicate token (which looks like "nn_u_unknown", e.g., is replaced
    #   by the surface form of the word as extracted from the carg field.
    # (2) If the token appears only once in the training data, it is
    #   anonymized by replacing it with a generic token like UNK0, UNK1.
    #   Occurrences of these tokens in the generated text will be replaced
    #   with the original surface form during post-processing (pointing the
    #   unknown words)
    DATA_DIR=data_unk_rare
    mkdir -p $DATA_DIR

    sh scripts/unknowns/prep_anonymize_rare_unknowns.sh $DATA_DIR

    # preprocess data files
    rm $DATA_DIR/opennmt.*
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

train_unk() {
    DATA_DIR=$1  # used in naming model files
    MODEL_PREFIX=models/$DATA_DIR
    GPU_ID=$2

    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data $DATA_DIR/opennmt \
        -layers 2 -dropout 0.3 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$DATA_DIR.log"
}

run_data_checks() {
    # Check counts of different types of anon tokens in both train and test
    # data to make sure preprocessing completed as expected
    echo "------------- Checking all -------------"
    echo "Num lines in data_unk_all (should be ~70K)"
    cat data_unk_all/train-src.txt | wc -l
    echo "UNK rare token occurrences in data_unk_all (should be 0)"
    cat data_unk_all/train-tgt.txt | tr ' ' '\n' | grep -P "(^| )UNK.*\d" | sort | uniq -c | sort -n | wc -l
    echo "_UNK anonymized token occurrences in data_unk_all (should be large)"
    cat data_unk_all/train-tgt.txt | tr ' ' '\n' | grep -P "_UNK\d" | sort | uniq -c | sort -n
    echo "Num lines in data_unk_all/dev (should be ~5K)"
    cat data_unk_all/dev-src.txt | wc -l
    echo "UNK rare token occurrences in data_unk_all/dev (should be 0)"
    cat data_unk_all/dev-tgt.txt | tr ' ' '\n' | grep -P "(^| )UNK.*\d" | sort | uniq -c | sort -n | wc -l
    echo "_UNK anonymized token occurrences in data_unk_all/dev (should be large)"
    cat data_unk_all/dev-tgt.txt | tr ' ' '\n' | grep -P "_UNK\d" | sort | uniq -c | sort -n
    echo
    echo "------------- Checking rare -------------"
    echo "Num lines in data_unk_rare (should be ~70K)"
    cat data_unk_rare/train-src.txt | wc -l
    echo "UNK rare token occurrences in data_unk_rare (should be > 0 but < data_unk_include)"
    cat data_unk_rare/train-tgt.txt | tr ' ' '\n' | grep -P "(^| )UNK.*\d" | sort | uniq -c | sort -n | wc -l
    echo "_UNK anonymized token occurrences in data_unk_rare (should be > 0 but < data_unk_all)"
    cat data_unk_rare/train-tgt.txt | tr ' ' '\n' | grep -P "_UNK\d" | sort | uniq -c | sort -n
    echo "Num lines in data_unk_rare/dev (should be ~5K)"
    cat data_unk_rare/dev-src.txt | wc -l
    echo "UNK rare token occurrences in data_unk_rare/dev (should be > 0 but < data_unk_include)"
    cat data_unk_rare/dev-tgt.txt | tr ' ' '\n' | grep -P "(^| )UNK.*\d" | sort | uniq -c | sort -n | wc -l
    echo "_UNK anonymized token occurrences in data_unk_rare/dev (should be > 0 but < data_unk_all)"
    cat data_unk_rare/dev-tgt.txt | tr ' ' '\n' | grep -P "_UNK\d" | sort | uniq -c | sort -n
    echo
    echo "------------- Checking include -------------"
    echo "Num lines in data_unk_include (should be ~70K)"
    cat data_unk_include/train-src.txt | wc -l
    echo "UNK rare token occurrences in data_unk_include (should be large)"
    cat data_unk_include/train-tgt.txt | tr ' ' '\n' | grep -P "(^| )UNK.*\d" | sort | uniq -c | sort -n | wc -l
    echo "_UNK token occurrences in data_unk_include (should be 0)"
    cat data_unk_include/train-tgt.txt | tr ' ' '\n' | grep -P "_UNK\d" | sort | uniq -c | sort -n
    echo "Num lines in data_unk_include/dev (should be ~5K)"
    cat data_unk_include/dev-src.txt | wc -l
    echo "UNK rare token occurrences in data_unk_include/dev (should be large)"
    cat data_unk_include/dev-tgt.txt | tr ' ' '\n' | grep -P "(^| )UNK.*\d" | sort | uniq -c | sort -n | wc -l
    echo "_UNK token occurrences in data_unk_include/dev (should be 0)"
    cat data_unk_include/dev-tgt.txt | tr ' ' '\n' | grep -P "_UNK\d" | sort | uniq -c | sort -n
}

# preprocess data for all experiments
preprocess_include_all_unknowns
preprocess_anonymize_all_unknowns
preprocess_anonymize_rare_unknowns

# visually inspect output of this fn to make sure preprocessing worked
run_data_checks
echo
echo "Make sure counts above look correct, then press any key to continue"
read kp

# train models on the different datasets
train_unk data_unk_include 0 &
train_unk data_unk_all 1 &
train_unk data_unk_rare 2

# Example commands for BLEU eval after training has completed:
# (Copy name of best-performing model from each experiment into arguments below)
#bash scripts/eval.sh models/data_unk_include_acc_87.14_ppl_3.02_e30.pt data_unk_include/dev data_unk_include/anon-replacements.json 0
#bash scripts/eval.sh  models/data_unk_all_acc_86.86_ppl_2.78_e30.pt data_unk_all/dev data_unk_all/anon-replacements.json 1
#bash scripts/eval.sh  models/data_unk_rare_acc_87.30_ppl_2.77_e30.pt data_unk_rare/dev data_unk_rare/anon-replacements.json 2

