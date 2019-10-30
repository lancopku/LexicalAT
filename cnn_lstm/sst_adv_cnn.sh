#!/usr/bin/env sh
set -e

NAME=sst_adv_cnn
TRAIN_DIR=./train/$NAME
INPUT_DIR=./data/sst
mkdir $TRAIN_DIR || true

python3 ./src/main.py \
    --mode=train_adv \
    --action=all \
    --model_type=cnn \
    --input_dir=$INPUT_DIR \
    --train_dir=$TRAIN_DIR \
    --test_steps=50 \
    --least_freq=2 \
    --num_classes=2 \
    --dis_warm_up_step=500 \
    --gene_warm_up_step=500 \
    --max_vocab_size=100000 \
    --embedding_dims=300 \
    --rnn_cell_size=300 \
    --batch_size=64 \
    --learning_rate=0.0001 \
    --generator_learning_rate=0.001 \
    --max_steps=20000 \
    --max_grad_norm=1.0 \
    --num_timesteps=100 \
    --keep_prob_emb=0.6 \
    --keep_prob_dense=0.9 \
