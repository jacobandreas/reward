#!/bin/sh

#kernprof -l ../../reward.py \
python -u ../../reward.py \
  --run train \
  --gpu \
  --lr 1e-3 \
  --discount 0.97 \
  --w_value 0.1 \
  --w_entropy 0.01 \
  > train.out \
  2> train.err
