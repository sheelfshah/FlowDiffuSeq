#!/bin/bash
steps=(1 2 5 10 25 50 125 250 500 1000 2000)
bleu_values=()
rouge_values=()
for step in ${steps[@]}; do
    bleu=$(grep "BLEU" script_logs/eval_${step}.log | rev | cut -d' ' -f1 | rev)
    bleu_values+=(${bleu})
    rouge=$(grep "ROUGE" script_logs/eval_${step}.log | rev | cut -d' ' -f1 | rev)
    rouge_values+=(${rouge})
done

# make comma separated string
steps_str=$(IFS=,; echo "${steps[*]}")
bleu_values_str=$(IFS=,; echo "${bleu_values[*]}")
rouge_values_str=$(IFS=,; echo "${rouge_values[*]}")
echo "steps = [${steps_str}]"
echo "bleu_values = [${bleu_values_str}]"
echo "rouge_values = [${rouge_values_str}]"

