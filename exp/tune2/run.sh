#!/bin/sh

#kernprof -l ../../reward.py \
python -u ../../reward.py \
  --run train \
  --gpu \
  --lr 1e-4 \
  --discount 0.97 \
  --w_value 0.05 \
  --w_entropy 0.05 \
  > train.out \
  2> train.err
