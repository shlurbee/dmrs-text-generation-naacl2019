#!/bin/bash

####
# Script for evaluating several of our trained models on a standard
# AMR test set (LDC2015E86). ACE is used to parse as many of the test examples
# as possible to DMRS, then the trained models are downloaded from S3 and used
# to generate text for the parsed lines.
#
# Note: should be run from root dir
####

# Info for fetching and using trained models
S3_BASE_LOCATION=https://s3-us-west-1.amazonaws.com/dmrs-text-generation/models
GOLD_DIR=vgold
GOLD_SILVER_DIR=vs6
ABLATED_DIR=sem_feat_ablation

# Trained on gold data with all features included:
GOLD_MODEL=$GOLD_DIR/data_unk_rare_acc_87.30_ppl_2.77_e29.pt

# Trained on gold data with node and edge features removed:
GOLD_ABLATED_MODEL=$ABLATED_DIR/data_vs1_ablate_sem_feats_acc_79.32_ppl_4.47_e32.pt

# Trained on gold+silver data with all features included (our best model, "vs6" in silver experiments):
GOLD_SILVER_MODEL=$GOLD_SILVER_DIR/data_vs6_acc_92.52_ppl_1.69_e22.pt

# Trained on gold+silver data with node and edge features removed:
GOLD_SILVER_ABLATED_MODEL=$ABLATED_DIR/data_vs6_ablate_sem_feats_acc_88.21_ppl_1.94_e24.pt

# Deanonymization data from each training set (needed for postprocessing after prediction)
ANON_REPLACEMENTS_GOLD=anon-replacements-gold.json  # anon data for gold models
ANON_REPLACEMENTS=anon-replacements.json  # anon data for gold+silver models

# Create the test data files
setup_AMR_test_data() {
    if [ ! -f amr-to-mrs/test.sents.txt.mrs ]; then
        cd amr-to-mrs
        sh run.sh
        cd ..
    fi
}

# Linearize and preprocess data so it can be input to trained models
prep_data() {
    mkdir -p data_amr/test
    python preprocessing.py amr-to-mrs/test.sents.txt.mrs data_amr/test/test
}

# Linearize, preprocess, and ablate features from data so it can be input to models trained on ablated data
prep_data_ablated() {
    # Check for dependencies
    if [ ! -d data_amr ]; then
        echo "data_amr directory not found - please run prep_data"
    fi
    # Note: regexes here must match what was used in scripts/run_silver_ablation.sh
    RE_REMOVE_ALL_NODE_FEAT="s/￨.*?( |$)/ /g"  # removes word features after ￨
    RE_REMOVE_EDGE_FEAT="s/([A-Z0-9\-])-(HEQ|H|EQ|NEQ)/\1/g"
    # Copy data into new dir with node and edge features removed
    mkdir -p data_amr_ablated/test
    cat data_amr/test/test-tgt.txt > data_amr_ablated/test/test-tgt.txt
    # Note both edge and node features are removed here
    cat data_amr/test/test-src.txt | perl -p -e "$RE_REMOVE_ALL_NODE_FEAT" | \
        perl -p -e "$RE_REMOVE_EDGE_FEAT" > data_amr_ablated/test/test-src.txt
    cat data_amr/test/test-orig.txt > data_amr_ablated/test/test-orig.txt
    cat data_amr/test/test-anon.txt > data_amr_ablated/test/test-anon.txt
}

# Download trained model and run sacrebleu
evaluate() {
    MODEL=$1
    EVAL_FILE_PREFIX=$2
    ANON_FILE=$3
    MODEL_DIR=$(dirname "${MODEL}")
    TARGET_FILE="$EVAL_FILE_PREFIX"-orig.txt
    # Download model from s3 if it's not already there
    if [ ! -f models/$MODEL ]; then
        mkdir -p models/$MODEL_DIR
        echo "Downloading $S3_BASE_LOCATION/$MODEL to models/$MODEL_DIR"
        wget -O models/$MODEL $S3_BASE_LOCATION/$MODEL
    fi
    # Download anon file from s3 if it's not already there
    if [ ! -f data/$ANON_FILE ]; then
        mkdir -p data
        echo "Downloading $S3_BASE_LOCATION/$ANON_FILE to data/$ANON_FILE"
        wget -O data/$ANON_FILE $S3_BASE_LOCATION/$ANON_FILE
    fi
    # Use model to predict sentences and evaluate
    bash scripts/eval.sh models/$MODEL $EVAL_FILE_PREFIX data/$ANON_FILE 0
}

setup_AMR_test_data
prep_data
prep_data_ablated
evaluate $GOLD_MODEL data_amr/test/test $ANON_REPLACEMENTS_GOLD
evaluate $GOLD_ABLATED_MODEL data_amr_ablated/test/test $ANON_REPLACEMENTS_GOLD
evaluate $GOLD_SILVER_MODEL data_amr/test/test $ANON_REPLACEMENTS
evaluate $GOLD_SILVER_ABLATED_MODEL data_amr_ablated/test/test $ANON_REPLACEMENTS

