#!/bin/bash

###
# Helper to train with different parameters.
# Please run from repo root after preprocessing data splits.
#
# This script was used to generate "Experiment 2" results, comparing
# architectures and hyperparams trained on gold data.
###

echo "Usage: From root dir run:"
echo "sh scripts/run_experiments.sh"

train_v1() {
    # train model using files created by preprocess()
    MODEL_VERSION=v1  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 1 -dropout 0.3 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type rnn -decoder_type rnn -rnn_type LSTM -rnn_size 500 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v2() {
    MODEL_VERSION=v2  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 1 -dropout 0.3 \
        -word_vec_size 250 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type rnn -decoder_type rnn -rnn_type LSTM -rnn_size 500 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v3() {
    MODEL_VERSION=v3  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 1 -dropout 0.3 \
        -word_vec_size 250 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type rnn -decoder_type rnn -rnn_type LSTM -rnn_size 250 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}


train_v4() {
    # train model using files created by preprocess()
    MODEL_VERSION=v4  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type rnn -decoder_type rnn -rnn_type LSTM -rnn_size 500 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v5() {
    MODEL_VERSION=v5  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.3 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type rnn -decoder_type rnn -rnn_type LSTM -rnn_size 500 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v6() {
    MODEL_VERSION=v6  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type rnn -decoder_type rnn -rnn_type LSTM -rnn_size 500 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v7() {
    MODEL_VERSION=v7  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.3 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v8() {
    MODEL_VERSION=v8  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v9() {
    MODEL_VERSION=v9  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 250 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v10() {
    MODEL_VERSION=v10  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 250 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v11() {
    MODEL_VERSION=v11  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -start_decay_at 15 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v12() {
    MODEL_VERSION=v12  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 250 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 400 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -start_decay_at 25 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

train_v13() {
    MODEL_VERSION=v13  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    # train model using files created by preprocess()
    rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data/opennmt \
        -layers 2 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001 -start_decay_at 25 -opt adam -epochs 40 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

# Start multiple jobs at once, running in the background
#train_v7 0 &
#train_v8 1 &
#train_v9 2 &
#train_v10 0
#train_v11 1 &
train_v12 1 &
train_v13 2
