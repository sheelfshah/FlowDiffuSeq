#!/bin/bash

mkdir -p script_logs
# steps=(2000 1000 500 250 125 50 25 10 5 2 1)
steps=(1 2 5 10 25 50 125 250 500 1000 2000)
for step in ${steps[@]}; do
    python -u run_decode.py \
    --model_dir pretrained/QQP/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test_ori20221113-20:27:29/ \
    --seed 123 \
    --split test \
    --bsz 64 \
    --step $step > script_logs/decode_$step.log 2>&1
    
    python eval_seq2seq.py \
    --folder ../generation_outputs/$step/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test_ori20221113-20\:27\:29/ema_0.9999_050000.pt.samples/ \
    > script_logs/eval_$step.log 2>&1

done
