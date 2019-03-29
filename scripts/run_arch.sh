#!/bin/bash

###
# Helpers to prepare different combinations of gold and silver data and train.
# Please run from repo root.
#
# This set of experiments tests different architectures on the best-performing
# silver+gold augmented training data set.
###

echo "Usage: From root dir run:"
echo "sh scripts/run_arch.sh"


# decrease dropout
train_v10() {
    MODEL_VERSION=v10  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v4/opennmt \
        -layers 2 -dropout 0.3 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

# three layers
train_v11() {
    MODEL_VERSION=v11  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v4/opennmt \
        -layers 3 -dropout 0.5 \
        -word_vec_size 500 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 800 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

# increase hidden layer and embedding size
train_v12() {
    MODEL_VERSION=v12  # used in naming model files
    MODEL_PREFIX=models/$MODEL_VERSION
    GPU_ID=$1

    #rm $MODEL_PREFIX*
    python OpenNMT-py/train.py -data data_v4/opennmt \
        -layers 3 -dropout 0.5 \
        -word_vec_size 800 -batch_type sents -max_grad_norm 5 -param_init_glorot \
        -encoder_type brnn -decoder_type rnn -rnn_type LSTM -rnn_size 1000 \
        -save_model $MODEL_PREFIX \
        -learning_rate 0.001  -start_decay_at 100 -opt adam -epochs 30 -gpuid $GPU_ID > "logs/train_$MODEL_VERSION.log"
}

#train_v10 0
#train_v11 2
train_v12 1
