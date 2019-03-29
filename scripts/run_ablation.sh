#!/bin/bash

###
# Experiments removing semantic features from linearization to see which ones
# contribute most to good results. Please run from repo root.
#
# N.B. This set of experiments is run on gold data only.
###

echo "Usage: From root dir run:"
echo "sh scripts/run_ablation.sh"

RE_GEND="gend=[A-Z\-]+"
RE_IND="ind=[\+\-]"
RE_MOOD="mood=[A-Z]+"
RE_NUM="num=[A-Z]+"
RE_PERF="perf=[\+\-]"
RE_PERS="pers=\d"
RE_SF="sf=[A-Z]+"
RE_TENSE="tense=[A-Z]+"
RE_PT="pt=[A-Z]+"
# Regex to remove everything except tense, number
RE_REMOVE_MOST_NODE_FEAT="s/(($RE_GEND|$RE_IND|$RE_MOOD|$RE_PERF|$RE_PERS|$RE_SF|$RE_PT)(\|)?|\|)//g"
# Regex for concatenated attributes attached to predicate with ￨
RE_REMOVE_ALL_NODE_FEAT="s/￨.*?( |$)/ /g"

# Regex for features embedded in edge labels. (Note HEQ before H to prevent partial match)
RE_REMOVE_EDGE_FEAT="s/([A-Z0-9\-])-(HEQ|H|EQ|NEQ)/\1/g"
RE_REMOVE_EDGE_ARGNUM="s/ARG\d/ARG/g"


# First, do preprocessing that is the same for all versions. Later functions
# will make copies of this data with certain features removed.
prep_data() {
	mkdir -p data/train
    mkdir -p data/dev

	# Linearize train, dev data
	python preprocessing.py data_penman/train.txt data/train/train
	python preprocessing.py data_penman/dev.txt data/dev/dev
	echo 'done linearizing'

	# Build vocab from training data
	python replace_rare.py vocab --vocabfile data/vocab.txt --infile data/train/train-tgt.txt

	# Use vocab to replace rare tokens in train/dev/test sets, anonymizing only those unknown tokens with freq 1
	python replace_rare.py replace --vocabfile data/vocab.txt --infile data/train/train --min_freq 2
	python replace_rare.py replace --vocabfile data/vocab.txt --infile data/dev/dev --min_freq 2

	# Overwrite lines in training data that overlap with dev set
	python scripts/remove_overlap.py \
		--train_files data/train/train-tgt.txt \
		--test_files data/dev/dev-tgt.txt \
		--blacklist_file data/blacklist.txt

	# Create file mapping anonymization tokens to most common surface form
	python replacements.py \
		--infiles data/train/train-anon.txt \
		--outfile data/anon-replacements-gold.json
}


# Include all semantic features (original data.)
prep_no_ablation() {
    # create opennmt training set
    rm data/opennmt.*
    python OpenNMT-py/preprocess.py -train_src data/train/train-src.txt \
        -train_tgt data/train/train-tgt.txt \
        -valid_src data/dev/dev-src.txt \
        -valid_tgt data/dev/dev-tgt.txt \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data data/opennmt -report_every 1000
}

# Remove all node attributes.
prep_ablate_node_feats() {
    DATA_DIR=data_ablate_node_feats
    mkdir -p $DATA_DIR
    rm $DATA_DIR/opennmt.*

    # copy gold data to dir, removing node features from source file
    cat data/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    cat data/train/train-src.txt | perl -p -e "$RE_REMOVE_ALL_NODE_FEAT" > $DATA_DIR/train-src.txt
    cat data/train/train-orig.txt > $DATA_DIR/train-orig.txt
    cat data/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # apply same preprocessing to dev data
    cat data/dev/dev-tgt.txt > $DATA_DIR/dev-tgt.txt
    cat data/dev/dev-src.txt | perl -p -e "$RE_REMOVE_ALL_NODE_FEAT" > $DATA_DIR/dev-src.txt
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

# Remove node attributes except a few that are obviously important.
prep_ablate_most_node_feats() {
    # everything except tense, num, person
    DATA_DIR=data_ablate_most_node_feats
    mkdir -p $DATA_DIR
    rm $DATA_DIR/opennmt.*

    # copy gold data to dir, removing node features from source file
    cat data/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    cat data/train/train-src.txt | perl -p -e "$RE_REMOVE_MOST_NODE_FEAT" > $DATA_DIR/train-src.txt
    cat data/train/train-orig.txt > $DATA_DIR/train-orig.txt
    cat data/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # apply same preprocessing to dev data
    cat data/dev/dev-tgt.txt > $DATA_DIR/dev-tgt.txt
    cat data/dev/dev-src.txt | perl -p -e "$RE_REMOVE_MOST_NODE_FEAT" > $DATA_DIR/dev-src.txt
    cat data/dev/dev-orig.txt > $DATA_DIR/dev-orig.txt
    cat data/dev/dev-anon.txt > $DATA_DIR/dev-anon.txt

    # preprocess data files
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

# Remove all occurrences of HEQ, H, EQ, NEQ from edge labels.
prep_ablate_edge_feats() {
    DATA_DIR=data_ablate_edge_feats
    mkdir -p $DATA_DIR
    rm $DATA_DIR/opennmt.*

    # copy gold data to dir, removing edge features from source file
    cat data/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    cat data/train/train-src.txt | perl -p -e "$RE_REMOVE_EDGE_FEAT" > $DATA_DIR/train-src.txt
    cat data/train/train-orig.txt > $DATA_DIR/train-orig.txt
    cat data/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # apply same preprocessing to dev data
    cat data/dev/dev-tgt.txt > $DATA_DIR/dev-tgt.txt
    cat data/dev/dev-src.txt | perl -p -e "$RE_REMOVE_EDGE_FEAT" > $DATA_DIR/dev-src.txt
    cat data/dev/dev-orig.txt > $DATA_DIR/dev-orig.txt
    cat data/dev/dev-anon.txt > $DATA_DIR/dev-anon.txt

    # preprocess data files
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

# Instead of ARG1, ARG2, etc, include only "ARG" (drops subj/obj info.)
prep_ablate_edge_arg_num() {
    # rename edges to, e.g., ARG and ARG-of, keeping direction but removing subj/obj info
    DATA_DIR=data_ablate_edge_arg_num
    mkdir -p $DATA_DIR
    rm $DATA_DIR/opennmt.*

    # copy gold data to dir, removing edge features from source file
    cat data/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    cat data/train/train-src.txt | perl -p -e "$RE_REMOVE_EDGE_ARGNUM" > $DATA_DIR/train-src.txt
    cat data/train/train-orig.txt > $DATA_DIR/train-orig.txt
    cat data/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # apply same preprocessing to dev data
    cat data/dev/dev-tgt.txt > $DATA_DIR/dev-tgt.txt
    cat data/dev/dev-src.txt | perl -p -e "$RE_REMOVE_EDGE_ARGNUM" > $DATA_DIR/dev-src.txt
    cat data/dev/dev-orig.txt > $DATA_DIR/dev-orig.txt
    cat data/dev/dev-anon.txt > $DATA_DIR/dev-anon.txt

    # preprocess data files
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
    echo "Each file should have ~70k lines"
    cat data/train/train-src.txt | wc -l
    cat data_ablate_node_feats/train-src.txt | wc -l
    cat data_ablate_most_node_feats/train-src.txt | wc -l
    cat data_ablate_edge_feats/train-src.txt | wc -l
    cat data_ablate_edge_arg_num/train-src.txt | wc -l
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

prep_no_ablation
prep_ablate_node_feats
prep_ablate_most_node_feats
prep_ablate_edge_feats
prep_ablate_edge_arg_num

run_data_checks

#train data 0  # baseline, all features intact
train data_ablate_node_feats 1
train data_ablate_most_node_feats 1
train data_ablate_edge_feats 1
train data_ablate_edge_arg_num 1

# Fill in best model names below and eval. Note that dev set must have same preprocessing as training set.
bash scripts/eval.sh models/to_keep/data_ablate_node_feats_acc_81.18_ppl_3.63_e29.pt data_ablate_node_feats/dev data/anon-replacements-gold.json 0
bash scripts/eval.sh models/to_keep/data_ablate_most_node_feats_acc_85.17_ppl_2.99_e26.pt data_ablate_most_node_feats/dev data/anon-replacements-gold.json 0
bash scripts/eval.sh models/to_keep/data_ablate_edge_feats_acc_86.73_ppl_2.92_e30.pt data_ablate_edge_feats/dev data/anon-replacements-gold.json 0
bash scripts/eval.sh models/to_keep/data_ablate_edge_arg_num_acc_86.92_ppl_2.86_e29.pt data_ablate_edge_arg_num/dev data/anon-replacements-gold.json 0

