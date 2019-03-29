#!/bin/bash

###
# Helpers to prepare different combinations of gold and silver data and train.
# Please run from repo root.
###

echo "Usage: From root dir run:"
echo "sh scripts/run_experiments.sh"

preprocess_silver() {
    DATA_DIR=data_gw
    mkdir -p $DATA_DIR

    # preprocess gigaword data files to create silver data (use gold data for validation)
    python OpenNMT-py/preprocess.py -train_src $DATA_DIR/train/train-src.txt \
        -train_tgt $DATA_DIR/train/train-tgt.txt \
        -valid_src data/dev/dev-src.txt \
        -valid_tgt data/dev/dev-tgt.txt \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data $DATA_DIR/opennmt -report_every 1000
}

preprocess_gold() {
    # preprocess gold data files to create gold training data
    python OpenNMT-py/preprocess.py -train_src data/train/train-src.txt \
        -train_tgt data/train/train-tgt.txt \
        -valid_src data/dev/dev-src.txt \
        -valid_tgt data/dev/dev-tgt.txt \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data data/opennmt -report_every 1000
}

preprocess_anon_map() {
    python replacements.py \
        --infiles data/train/train-anon.txt data_gw/train/train-anon.txt \
        --outfile data/anon-replacements.json
}

preprocess_v1() {
    # Same as v19 from prev experiments.
    # Combines 600K silver data examples with 70K x 10 = 700K gold examples
    # Increases max vocab size from default of 50K to 100K (no min freq)
    # Does not replace rare unknowns with UNK0, UNK1, etc
    NUM_COPIES=10
    DATA_DIR=data_v1
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n 600000 data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n 600000 data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n 600000 data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n 600000 data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # append gold data. (assumes opennmt will shuffle)
    i=0
    while [ $i -lt  $NUM_COPIES ]; do
		echo $i
        cat data/train/train-tgt.txt >> $DATA_DIR/train-tgt.txt
        cat data/train/train-src.txt >> $DATA_DIR/train-src.txt
        cat data/train/train-orig.txt >> $DATA_DIR/train-orig.txt
        cat data/train/train-anon.txt >> $DATA_DIR/train-anon.txt
		i=`expr $i + 1`
    done

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
}


preprocess_v2() {
    # Combines 300K silver data examples with 70K x 5 = 350K gold examples
    # Increases max vocab size from default of 50K to 100K (no min freq)
    # Does not replace rare unknowns with UNK0, UNK1, etc
    NUM_COPIES=5
    DATA_DIR=data_v2
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n 300000 data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n 300000 data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n 300000 data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n 300000 data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # append gold data. (assumes opennmt will shuffle)
    i=0
    while [ $i -lt  $NUM_COPIES ]; do
		echo $i
        cat data/train/train-tgt.txt >> $DATA_DIR/train-tgt.txt
        cat data/train/train-src.txt >> $DATA_DIR/train-src.txt
        cat data/train/train-orig.txt >> $DATA_DIR/train-orig.txt
        cat data/train/train-anon.txt >> $DATA_DIR/train-anon.txt
		i=`expr $i + 1`
    done

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
}


preprocess_v3() {
    # Combines 800K silver data examples with 70K x 11 = 770K gold examples
    # Increases max vocab size from default of 50K to 100K (no min freq)
    # Does not replace rare unknowns with UNK0, UNK1, etc
    NUM_COPIES=11
    DATA_DIR=data_v3
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n 800000 data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n 800000 data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n 800000 data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n 800000 data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # append gold data. (assumes opennmt will shuffle)
    i=0
    while [ $i -lt  $NUM_COPIES ]; do
		echo $i
        cat data/train/train-tgt.txt >> $DATA_DIR/train-tgt.txt
        cat data/train/train-src.txt >> $DATA_DIR/train-src.txt
        cat data/train/train-orig.txt >> $DATA_DIR/train-orig.txt
        cat data/train/train-anon.txt >> $DATA_DIR/train-anon.txt
		i=`expr $i + 1`
    done

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
}


preprocess_v4() {
    # Combines 800K silver data examples with 70K x 6 = 420K gold examples
    # Increases max vocab size from default of 50K to 100K (no min freq)
    # Does not replace rare unknowns with UNK0, UNK1, etc
    NUM_COPIES=6
    DATA_DIR=data_v4
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n 800000 data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n 800000 data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n 800000 data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n 800000 data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # append gold data. (assumes opennmt will shuffle)
    i=0
    while [ $i -lt  $NUM_COPIES ]; do
		echo $i
        cat data/train/train-tgt.txt >> $DATA_DIR/train-tgt.txt
        cat data/train/train-src.txt >> $DATA_DIR/train-src.txt
        cat data/train/train-orig.txt >> $DATA_DIR/train-orig.txt
        cat data/train/train-anon.txt >> $DATA_DIR/train-anon.txt
		i=`expr $i + 1`
    done

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
}


preprocess_v5() {
    # Combines 800K silver data examples with 70K x 3 = 210K gold examples
    # Increases max vocab size from default of 50K to 100K (no min freq)
    # Does not replace rare unknowns with UNK0, UNK1, etc
    NUM_COPIES=3
    DATA_DIR=data_v5
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n 800000 data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n 800000 data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n 800000 data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n 800000 data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # append gold data. (assumes opennmt will shuffle)
    i=0
    while [ $i -lt  $NUM_COPIES ]; do
		echo $i
        cat data/train/train-tgt.txt >> $DATA_DIR/train-tgt.txt
        cat data/train/train-src.txt >> $DATA_DIR/train-src.txt
        cat data/train/train-orig.txt >> $DATA_DIR/train-orig.txt
        cat data/train/train-anon.txt >> $DATA_DIR/train-anon.txt
		i=`expr $i + 1`
    done

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
}

train_v1() {
    MODEL_VERSION=v1  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v1/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -train_from models/v1_acc_91.96_ppl_1.66_e15.pt \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v2() {
    MODEL_VERSION=v2  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v2/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -train_from models/v2_acc_91.83_ppl_1.89_e15.pt \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v3() {
    MODEL_VERSION=v3  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v3/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v4() {
    MODEL_VERSION=v4  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v4/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v5() {
    MODEL_VERSION=v5  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v5/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


run_data_checks() {
    echo "Num lines in data (gold data) (should be ~70K)"
    cat data/train/train-src.txt | wc -l

    echo "Num lines in data_gw (should be ~1M)"
    cat data_gw/train/train-src.txt | wc -l

    echo "Num lines in data_v1 (should be ~1.2M)"
    cat data_v1/train-src.txt | wc -l

    echo "Num lines in data_v2 (should be ~600K)"
    cat data_v2/train-src.txt | wc -l

    echo "Num lines in data_v3 (should be ~1.6M)"
    cat data_v3/train-src.txt | wc -l

    echo "Num lines in data_v4 (should be ~1.2M)"
    cat data_v4/train-src.txt | wc -l

    echo "Num lines in data_v5 (should be ~1M)"
    cat data_v5/train-src.txt | wc -l

}

#preprocess_silver
#preprocess_gold
#preprocess_anon_map
#preprocess_v1
#preprocess_v2
#preprocess_v3
#preprocess_v4
#preprocess_v5

#run_data_checks

#train_v1 1
train_v2 0
#train_v3 2
#train_v4 0
#train_v5 1
