#!/usr/bin/env sh
set -e

BERT_BASE_DIR=./uncased_L-24_H-1024_A-16
TASK_NAME=rt
RESULT_FILE=large_result_$TASK_NAME.txt

ACTION=no_updownsame
NAME=$TASK_NAME
python3 src/msr.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --action=$ACTION \
  --generator_learning_rate 0.001 \
  --data_dir=data/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=24 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --train_dir=./tmp/$NAME \
  --output_dir=./tmp/$NAME 2>&1 | tail -10 | tee -a $RESULT_FILE
