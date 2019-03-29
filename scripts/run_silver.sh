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

preprocess_v15() {
    # combines 600K silver data examples with 60K x 10 = 600K gold examples
    NUM_COPIES=10
    DATA_DIR=data_v15
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

    # preprocess data files in the usual way
    python OpenNMT-py/preprocess.py -train_src $DATA_DIR/train-src.txt \
        -train_tgt $DATA_DIR/train-tgt.txt \
        -valid_src data/dev/dev-src.txt \
        -valid_tgt data/dev/dev-tgt.txt \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data $DATA_DIR/opennmt -report_every 1000
}

preprocess_v17() {
    # combines 60K silver data examples with all gold examples
    NUM_COPIES=1
    DATA_DIR=data_v17
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n 60000 data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n 60000 data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n 60000 data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n 60000 data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

    # append gold data. (assumes opennmt will shuffle)
    #for ((i=0;i<$NUM_COPIES;i+=1)); do
    for i in {1..$NUM_COPIES}; do
        echo $i
        cat data/train/train-tgt.txt >> $DATA_DIR/train-tgt.txt
        cat data/train/train-src.txt >> $DATA_DIR/train-src.txt
        cat data/train/train-orig.txt >> $DATA_DIR/train-orig.txt
        cat data/train/train-anon.txt >> $DATA_DIR/train-anon.txt
    done

    # preprocess data files in the usual way
    python OpenNMT-py/preprocess.py -train_src $DATA_DIR/train-src.txt \
        -train_tgt $DATA_DIR/train-tgt.txt \
        -valid_src data/dev/dev-src.txt \
        -valid_tgt data/dev/dev-tgt.txt \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data $DATA_DIR/opennmt -report_every 1000
}

preprocess_v19() {
    # same as v15 but increases max vocab size from default of 50K to 100K (no min freq)
    # combines 600K silver data examples with 60K x 10 = 600K gold examples
    NUM_COPIES=10
    DATA_DIR=data_v19
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

preprocess_v20() {
    # same as v15 but increases max vocab size from default of 50K to 100K (no min freq)
    # combines 60K silver data examples with 60K gold
    NUM_COPIES=1
    DATA_DIR=data_v20
    mkdir -p $DATA_DIR

    # copy silver data to directory
    head -n 60000 data_gw/train/train-tgt.txt > $DATA_DIR/train-tgt.txt
    head -n 60000 data_gw/train/train-src.txt > $DATA_DIR/train-src.txt
    head -n 60000 data_gw/train/train-orig.txt > $DATA_DIR/train-orig.txt
    head -n 60000 data_gw/train/train-anon.txt > $DATA_DIR/train-anon.txt

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


preprocess_v21() {
    # same as v15 but increases max vocab size from default of 50K to 100K (min freq 2)
    # same as v19 but requires min freq of 2 for vocab
    # combines 600K silver data examples with 60K x 10 = 600K gold examples
    NUM_COPIES=10
    DATA_DIR=data_v21
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
        -src_words_min_frequency 2 \
        -tgt_words_min_frequency 2 \
        -src_seq_length 400 -tgt_seq_length 400 \
        -shuffle 1 -data_type text \
        -save_data $DATA_DIR/opennmt -report_every 1000
}


preprocess_v23() {
    # same as v19 but replaces rare unknowns in the training data to reduce vocab size
    # combines 600K silver data examples with 60K x 10 = 600K gold examples
    NUM_COPIES=10
    DATA_DIR=data_v23
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

    # replace rare unknown placeholders with _UNK0 to reduce vocab size
    python -c "import preprocessing; preprocessing.replace_rare_tokens('data_v23/train', 'data/vocab.txt', min_word_freq=2)"
    echo 'Replaced rare unknown placeholders with _UNK0'

    # preprocess data files
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

train_v14() {
    MODEL_VERSION=v14  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on silver data, evaluate on gold dev data
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_gw/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -start_decay_at 100 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v15() {
    MODEL_VERSION=v15  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v15/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v16() {
    MODEL_VERSION=v16
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # load model pretrained on silver data (v14) and train more on gold data to fine-tune
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_gw/opennmt \
        -layers 2 -dropout 0.5 \
        -train_from models/v14_acc_69.44_ppl_5.96_e40.pt \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -start_decay_at 100 -opt adam -epochs 60 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


train_v17() {
    MODEL_VERSION=v17
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different data dir)
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v17/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


train_v18() {
    # train on just gold as a baseline
    MODEL_VERSION=v18
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


train_v19() {
    MODEL_VERSION=v19  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v19/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


train_v20() {
    MODEL_VERSION=v20  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v20/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX -start_checkpoint_at 10 \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v21() {
    MODEL_VERSION=v21  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v19/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


train_v22() {
    # same data as v19 but using larger model
    MODEL_VERSION=v22  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v19/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 800 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 1000 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v23() {
    MODEL_VERSION=v23  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train on combination of silver and gold data (note different train data dir)
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v23/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 35 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


run_data_checks() {
    echo "Num lines in data (gold data) (should be ~60K)"
    cat data/train/train-src.txt | wc -l

    echo "Num lines in data_gw (used for train_v14) (should be ~800K)"
    cat data_gw/train/train-src.txt | wc -l

    echo "Num lines in data_v15 (should be ~1.2M)"
    cat data_v15/train-src.txt | wc -l

    echo "Num lines in data_v17 (should be ~120K)"
    cat data_v17/train-src.txt | wc -l

    echo "Num lines in data_v19 (should be ~1.2M)"
    cat data_v19/train-src.txt | wc -l

    echo "Num lines in data_v20 (should be ~120K)"
    cat data_v20/train-src.txt | wc -l

    echo "Num lines in data_v21 (should be ~1.2M)"
    cat data_v21/train-src.txt | wc -l

    echo "Num lines in data_v23 (should be ~1.2M)"
    cat data_v23/train-src.txt | wc -l


}

#preprocess_silver
#preprocess_gold
#preprocess_anon_map
#preprocess_v15
#preprocess_v17
#preprocess_v19
#preprocess_v20
#preprocess_v21
#preprocess_v23

#run_data_checks

#train_v14 2
#train_v15 0
#train_v16 2
#train_v17 2 &
#train_v18 1
#train_v19 2
#train_v20 2
#train_v21 1
#train_v22 0
#train_v23 2
