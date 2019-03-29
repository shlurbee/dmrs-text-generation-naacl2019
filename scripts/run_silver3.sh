#!/bin/bash

###
# Helpers to prepare datasets of increasing size, with gold:silver ratio fixed at 1:2.
# Please run from repo root.
###

echo "Usage: From root dir run:"
echo "sh scripts/run_silver3.sh"


prep_data() {
	# Make sure expected directories exist
	mkdir -p data/train
	mkdir -p data/dev
	mkdir -p data/test
	mkdir -p data_gw/train
	mkdir -p data_wsj/dev/
	mkdir -p data_wsj/test/

	# Create dev set of only wsj examples (since these most closely match the silver domain)
	cat data_penman/dev/wsj*.txt > data_penman/dev_wsj.txt
	cat data_penman/test/wsj*.txt > data_penman/test_wsj.txt

	# Linearize train, dev, test data
	# "--with_blanks" inserts blank lines when the input can't be parsed
	# to make sure line numbers don't change (useful for comparison with parser)
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
	python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_wsj/dev/dev --min_freq 2
	python replace_rare.py replace --vocabfile $VOCAB_FILE --infile data_wsj/test/test --min_freq 2
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

    # preprocess data files, increasing max vocab size
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
    check_unks data/train/train-tgt.txt
    check_unks data/dev/dev-tgt.txt
    check_unks data/test/test-tgt.txt
    check_unks data_wsj/dev/dev-tgt.txt
    check_unks data_wsj/test/test-tgt.txt
    check_unks data_gw/train/train-tgt.txt

    # Print num lines in the different data sets to make sure they're correct
    echo "Num lines in data (gold data) (should be ~70K)"
    cat data/train/train-src.txt | wc -l

    echo "Num lines in data_gw (should be ~800K)"
    cat data_gw/train/train-src.txt | wc -l

    echo "Num lines in data_vs1 (should be ~210K)"
    cat data_vs1/train-src.txt | wc -l

    echo "Num lines in data_vs2 (should be ~420K)"
    cat data_vs2/train-src.txt | wc -l

    echo "Num lines in data_vs3 (should be ~630K)"
    cat data_vs3/train-src.txt | wc -l

    echo "Num lines in data_vs4 (should be ~840K)"
    cat data_vs4/train-src.txt | wc -l

    echo "Num lines in data_vs5 (should be ~1.05M)"
    cat data_vs5/train-src.txt | wc -l

    echo "Num lines in data_vs6 (should be 1.25M)"
    cat data_vs6/train-src.txt | wc -l

    echo "Num lines in data_vs0 (should be ~800K)"
    cat data_vs0/train-src.txt | wc -l
}

train_vs() {
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

prep_data

# preprocess_vs $num_gold_copies $num_silver_examples
preprocess_vs 1 140000
preprocess_vs 2 280000
preprocess_vs 3 420000
preprocess_vs 4 560000
preprocess_vs 5 700000
preprocess_vs 6 850000
preprocess_vs 0 850000  # silver only

run_data_checks

# train_vs $data_dir $gpu_id
train_vs data_vs1 0
train_vs data_vs2 1
train_vs data_vs3 2
train_vs data_vs4 1
train_vs data_vs5 2
train_vs data_vs6 2
train_vs data_vs0 0


## ------  Example commands for evaluating results -------
# Replace model name in the commands below with name of best-performing model from training

time bash scripts/eval.sh models/data_vs1_acc_90.51_ppl_2.00_e26.pt data/dev/dev data/anon-replacements.json 0
bash scripts/eval.sh models/data_vs1_acc_90.51_ppl_2.00_e26.pt data_wsj/dev/dev data/anon-replacements.json 0

time bash scripts/eval.sh models/vs2/data_vs2_acc_91.60_ppl_1.83_e25.pt data/dev/dev data/anon-replacements.json 0
bash scripts/eval.sh models/vs2/data_vs2_acc_91.60_ppl_1.83_e25.pt  data_wsj/dev/dev data/anon-replacements.json 0

time bash scripts/eval.sh models/vs3/data_vs3_acc_91.92_ppl_1.82_e23.pt data/dev/dev data/anon-replacements.json 0
bash scripts/eval.sh models/vs3/data_vs3_acc_91.92_ppl_1.82_e23.pt  data_wsj/dev/dev data/anon-replacements.json 0

time bash scripts/eval.sh models/vs4/data_vs4_acc_92.32_ppl_1.72_e22.pt data/dev/dev data/anon-replacements.json 0
bash scripts/eval.sh models/vs4/data_vs4_acc_92.32_ppl_1.72_e22.pt  data_wsj/dev/dev data/anon-replacements.json 0

time bash scripts/eval.sh models/vs5/data_vs5_acc_92.33_ppl_1.71_e21.pt data/dev/dev data/anon-replacements.json 0
bash scripts/eval.sh models/vs5/data_vs5_acc_92.33_ppl_1.71_e21.pt  data_wsj/dev/dev data/anon-replacements.json 0

time bash scripts/eval.sh models/vs6/data_vs6_acc_92.52_ppl_1.69_e22.pt data/dev/dev data/anon-replacements.json 0
bash scripts/eval.sh models/vs6/data_vs6_acc_92.52_ppl_1.69_e22.pt  data_wsj/dev/dev data/anon-replacements.json 0

time bash scripts/eval.sh models/vgold/data_unk_rare_acc_87.30_ppl_2.77_e29.pt data/dev/dev data/anon-replacements.json 1
bash scripts/eval.sh models/vgold/data_unk_rare_acc_87.30_ppl_2.77_e29.pt data_wsj/dev/dev data/anon-replacements.json 1

time bash scripts/eval.sh models/vsilver/data_vs0_acc_90.26_ppl_2.30_e25.pt data/dev/dev data/anon-replacements.json 1
bash scripts/eval.sh models/vsilver/data_vs0_acc_90.26_ppl_2.30_e25.pt data_wsj/dev/dev data/anon-replacements.json 1

bash scripts/eval.sh models/vgold/data_unk_rare_acc_87.30_ppl_2.77_e29.pt data/test/test data/anon-replacements.json 0
bash scripts/eval.sh models/vgold/data_unk_rare_acc_87.30_ppl_2.77_e29.pt data_wsj/test/test data/anon-replacements.json 0

bash scripts/eval.sh models/vsilver/data_vs0_acc_90.26_ppl_2.30_e25.pt data/test/test data/anon-replacements.json 0
bash scripts/eval.sh models/vsilver/data_vs0_acc_90.26_ppl_2.30_e25.pt data_wsj/test/test data/anon-replacements.json 0

bash scripts/eval.sh models/vs6/data_vs6_acc_92.52_ppl_1.69_e22.pt data/test/test data/anon-replacements.json 0
bash scripts/eval.sh models/vs6/data_vs6_acc_92.52_ppl_1.69_e22.pt  data_wsj/test/test data/anon-replacements.json 0

